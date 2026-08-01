from __future__ import annotations

import asyncio

from tests.core._workload_secret_support import FakeProvider, collect_events

from cayu.core import AgentSpec, EventType, Message
from cayu.providers import ModelStreamEvent
from cayu.runtime import CayuApp, RunRequest, SessionQuery
from cayu.tools import SubagentSpec, SubagentTool
from cayu.vaults import SecretRedactor


def test_foreground_subagent_generated_lineage_survives_short_secret_collision() -> None:
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "Review the change."},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("review complete"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(secret_redactor=SecretRedactor("-"), enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="model"),
        tools=[SubagentTool(app, agents={"reviewer": SubagentSpec(agent_name="reviewer")})],
    )
    app.register_agent(AgentSpec(name="reviewer", model="model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="parent",
                messages=[Message.text("user", "delegate")],
            ),
        )
    )
    sessions = asyncio.run(app.session_store.list_sessions()).sessions
    roots = [session for session in sessions if session.parent_session_id is None]
    assert len(roots) == 1
    root = roots[0]
    children = asyncio.run(
        app.session_store.list_sessions(SessionQuery(parent_session_id=root.id))
    ).sessions

    assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
    assert len(children) == 1
    assert children[0].id.startswith(f"{root.id}_subagent_")
    assert children[0].parent_session_id == root.id
    assert children[0].causal_budget_id == root.id
    assert children[0].status.value == "completed"
