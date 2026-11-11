"""A selected fallback is not the target originally resolved from a model slot."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_model_failover_recovery import _RecoveryProvider

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.providers.retry_policy import RetryPolicy
from cayu.sessions import ModelFailoverPolicy, ModelTarget
from cayu.sessions.outcomes import run_to_completion


@pytest.mark.parametrize("fallback_model", ["other-model", "shared-model"])
def test_fallback_does_not_inherit_primary_slot_for_same_model(fallback_model):
    asyncio.run(_fallback_does_not_inherit_primary_slot(fallback_model))


async def _fallback_does_not_inherit_primary_slot(fallback_model):
    app = CayuApp(enable_logging=False)
    primary = _RecoveryProvider("primary")
    backup = _RecoveryProvider("backup")
    app.register_provider(primary, default=True)
    app.register_provider(backup)
    app.require_model("answer", default="shared-model")
    app.register_agent(
        AgentSpec(name="assistant", model="unused", provider_name="primary"),
        model_slot="answer",
    )
    try:
        outcome = await run_to_completion(
            app,
            RunRequest(
                session_id="fallback-slot",
                agent_name="assistant",
                messages=[Message.text("user", "answer")],
                retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                failover=ModelFailoverPolicy(
                    fallbacks=(ModelTarget(provider_name="backup", model=fallback_model),),
                    max_total_attempts=2,
                ),
            ),
        )
        assert outcome.ok, outcome.error
        assert len(primary.requests) == len(backup.requests) == 1
        assert primary.requests[0].model_slot == "answer"
        assert backup.requests[0].model == fallback_model
        assert backup.requests[0].model_slot is None, (
            "Fallback requests must not be attributed to the primary slot merely "
            "because a different provider uses the same model ID"
        )
    finally:
        await app.aclose()
