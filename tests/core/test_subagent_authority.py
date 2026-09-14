from __future__ import annotations

import asyncio
import re

import pytest
from tests.core._workload_secret_support import FakeProvider, collect_events

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import RunRequest, SessionQuery
from cayu.tools.subagents import SubagentSpec, SubagentTool
from cayu.vaults.redaction import SecretRedactor


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
    assert re.fullmatch(r"cayu-child:v1:subagent:[0-9a-f]{64}", children[0].id)
    assert children[0].parent_session_id == root.id
    assert children[0].causal_budget_id == root.id
    assert children[0].status.value == "completed"


@pytest.mark.parametrize("provenance", ["attested", "raw", "serialized", "mutated"])
def test_subagent_lineage_redaction_requires_exact_private_provenance(provenance):
    from cayu.runtime._child_session_identity import run_request_with_subagent_lineage
    from cayu.runtime._session_request_boundary import prepare_run_request
    from cayu.sessions.base import run_request_with_runtime_generated_authority

    request = RunRequest(
        session_id="child",
        parent_session_id="parent-generated",
        causal_budget_id="budget",
        agent_name="reviewer",
        messages=[Message.text("user", "review")],
        metadata={
            "subagent": {
                "parent_session_id": "parent-generated",
                "idempotency_key": "runtime-generated",
                "spawn_fingerprint": "sha256:1234",
                "tool_call_id": "provider-controlled",
            },
            "note": "caller-controlled",
        },
    )
    if provenance != "raw":
        request = run_request_with_subagent_lineage(request)
    if provenance == "serialized":
        request = RunRequest.model_validate(request.model_dump())
    if provenance == "mutated":
        request.metadata["subagent"]["idempotency_key"] = "replacement-generated"
    # Top-level identity authority is independent of nested lineage authority.
    request = run_request_with_runtime_generated_authority(request, "parent_session_id")
    prepared = prepare_run_request(request, redactor=SecretRedactor("-"))
    lineage = prepared.metadata["subagent"]
    if provenance == "attested":
        assert lineage["parent_session_id"] == "parent-generated"
        assert lineage["idempotency_key"] == "runtime-generated"
    else:
        assert "[REDACTED_SECRET]" in lineage["parent_session_id"]
        assert "[REDACTED_SECRET]" in lineage["idempotency_key"]
    assert "[REDACTED_SECRET]" in lineage["tool_call_id"]
    assert "[REDACTED_SECRET]" in prepared.metadata["note"]
