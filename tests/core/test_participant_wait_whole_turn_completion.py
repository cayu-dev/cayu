"""A participant execution-to-wait parks at every whole-turn result, not only plain text."""

import asyncio

import pytest
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_collaboration_waits import wait_for
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_tool_completion import FinalTool

from cayu import AgentSpec, EventType, Message, RunRequest, SessionStatus
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.context import StructuredOutputSpec
from cayu.evals import ScriptedModelProvider
from cayu.evals.testing import scripted_structured_output
from cayu.providers import ModelStreamEvent
from cayu.runtime import ToolCompletionPolicy
from cayu.runtime._session_continuation import continuation_digest
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)
from cayu.sessions.memory import InMemorySessionStore


@pytest.mark.parametrize("completion", ["text", "native", "tool", "final_tool"])
def test_participant_execution_to_wait_parks_all_whole_turn_results(completion):
    asyncio.run(_scenario(completion))


async def _scenario(completion):
    source = InMemoryCollaborationStore()
    store = InMemorySessionStore()
    try:
        app, resolver, values = await public_setup(source, session_store=store)
        initialized = values[1]
        receipt = await app.accept_collaboration_request(values[4], context=resolver.context)
        wait = wait_for(receipt, initialized).model_copy(
            update={"service_policy": "clarification", "failure_policy": "return_and_report"}
        )
        if completion == "tool":
            script = list(scripted_structured_output({"answer": "OK"}, id="answer"))
        elif completion == "final_tool":
            script = [
                ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="question"),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        else:
            script = [
                ModelStreamEvent.text_delta('{"answer":"OK"}'),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        provider = ScriptedModelProvider(
            [script], supports_native_structured_output=completion == "native"
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="root", model="model"),
            tools=[FinalTool()] if completion == "final_tool" else [],
        )
        run = RunRequest(
            agent_name="root",
            messages=[Message.text("user", "Answer, then wait.")],
            structured_output=(
                StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    strategy=completion,
                )
                if completion in {"tool", "native"}
                else None
            ),
            tool_completion=(
                ToolCompletionPolicy(tool_names=["ask_customer"])
                if completion == "final_tool"
                else None
            ),
        )
        participant = values[2].reference
        session, _ = await app.create_participant_session(
            ParticipantSessionCreationRequest(creation_key="root-with-wait", request=run),
            participant=participant,
            context=CONTEXT,
        )
        execution = ParticipantSessionExecutionRequest(
            request=run.model_copy(update={"session_id": session.id}),
            session_instance_id=session.instance_id,
            execution_key="execute-and-wait",
        )
        events = [
            event
            async for event in app.execute_participant_session_to_wait(
                execution,
                wait,
                participant=participant,
                context=CONTEXT,
                wait_context=resolver.context,
            )
        ]
        assert len(provider.requests) == 1
        assert not [e.payload for e in events if e.type == EventType.SESSION_FAILED]
        retained = await store.load_continuation_ticket(
            session.id,
            session_instance_id=session.instance_id,
            registration_key="execution-wait:" + continuation_digest(wait.operation),
        )
        assert retained is not None
        assert retained.ticket.state == "WAITING", (
            completion,
            retained.ticket.state,
            (await store.load(session.id)).status,
        )
        assert (await store.load(session.id)).status == SessionStatus.INTERRUPTED
        assert not any(e.type == EventType.SESSION_COMPLETED for e in events)
        # The exact execution replays the parked wait: no second model call, no failure.
        replayed = [
            event
            async for event in app.execute_participant_session_to_wait(
                execution,
                wait,
                participant=participant,
                context=CONTEXT,
                wait_context=resolver.context,
            )
        ]
        assert len(provider.requests) == 1
        assert not [e for e in replayed if e.type == EventType.SESSION_FAILED]
        assert (await store.load(session.id)).status == SessionStatus.INTERRUPTED
    finally:
        await source.close()
