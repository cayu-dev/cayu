"""Internal evaluation apps must not publish attention under the deployed identity."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.evals.judges import _isolated_structured_judge_app
from cayu.evals.runtime_replay import _isolated_app
from cayu.evals.testing import ScriptedModelProvider
from cayu.human_attention_handoff import HumanAttentionHandoff, HumanAttentionHandoffSink
from cayu.project_control_plane import ProjectEvalJudgeConfiguration
from cayu.providers import ModelStreamEvent
from cayu.server.evals_registry import _generated_project_model_judges


@pytest.mark.parametrize("kind", ["structured_judge", "project_judge", "replay"])
@pytest.mark.parametrize("complete_environment", [True, False])
def test_internal_apps_do_not_inherit_attention(monkeypatch, kind, complete_environment):
    async def run():
        monkeypatch.setenv("CAYU_ATTENTION_HANDOFF_URL", "https://receiver.invalid/attention")
        if complete_environment:
            monkeypatch.setenv("CAYU_ATTENTION_HANDOFF_TOKEN", "test-token")
        else:
            monkeypatch.delenv("CAYU_ATTENTION_HANDOFF_TOKEN", raising=False)
        source = CayuApp(enable_logging=False, human_attention_handoff=HumanAttentionHandoff(None))
        source.register_provider(
            ScriptedModelProvider(
                [(ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed())]
            ),
            default=True,
        )
        source.register_agent(AgentSpec(name="judge", model="test"))
        if complete_environment:
            deployed = CayuApp(enable_logging=False)
            assert deployed.human_attention_handoff is not None
            assert any(isinstance(s, HumanAttentionHandoffSink) for s in deployed._event_sinks)
            await deployed.aclose()
        if kind == "structured_judge":
            isolated = _isolated_structured_judge_app(source, "judge")
            agent_name = "judge"
        elif kind == "project_judge":
            (target,) = _generated_project_model_judges(
                source,
                ProjectEvalJudgeConfiguration(
                    provider_name=source.describe().agents[0].resolved_provider,
                    model="test",
                    privacy_policy="public-only",
                    allow_same_model=True,
                ),
                price_book=None,
            )
            isolated, agent_name = target.app, target.agent_name
        else:
            isolated = _isolated_app(
                source,
                provider_batches=None,
                request_billing_identities=None,
                completion_billing_identities=None,
                tool_tracker=None,
                clock_instant=datetime.now(UTC),
            )
            agent_name = "judge"
        assert isolated.human_attention_handoff is None
        assert not any(isinstance(s, HumanAttentionHandoffSink) for s in isolated._event_sinks)
        events = [
            event
            async for event in isolated.run(
                RunRequest(agent_name=agent_name, messages=[Message.text("user", "hello")])
            )
        ]
        assert any(str(event.type) == "session.completed" for event in events)
        await isolated.aclose()
        await source.aclose()

    asyncio.run(run())
