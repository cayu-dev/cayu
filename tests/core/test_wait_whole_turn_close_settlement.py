"""An execution wait must finish publication when its observer closes at the pause."""

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
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("completion", ["text", "native", "tool", "final_tool"])
@pytest.mark.parametrize(
    "close_at", [None, EventType.INTERACTION_INTERRUPTED, EventType.TURN_COMPLETED]
)
def test_close_at_whole_turn_wait_finishes_terminal_evidence(
    tmp_path, backend, completion, close_at
):
    asyncio.run(_scenario(tmp_path, backend, completion, close_at))


async def _scenario(tmp_path, backend, completion, close_at):
    source = InMemoryCollaborationStore()
    store = (
        InMemorySessionStore()
        if backend == "memory"
        else SQLiteSessionStore(tmp_path / "wait-close.sqlite")
    )
    app = None
    try:
        app, resolver, values = await public_setup(source, session_store=store)
        receipt = await app.accept_collaboration_request(values[4], context=resolver.context)
        wait = wait_for(receipt, values[1]).model_copy(
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
            ParticipantSessionCreationRequest(creation_key="wait-close", request=run),
            participant=participant,
            context=CONTEXT,
        )
        execution = ParticipantSessionExecutionRequest(
            request=run.model_copy(update={"session_id": session.id}),
            session_instance_id=session.instance_id,
            execution_key="execute-and-wait",
        )
        stream = app.execute_participant_session_to_wait(
            execution,
            wait,
            participant=participant,
            context=CONTEXT,
            wait_context=resolver.context,
        )
        try:
            async for event in stream:
                if event.type is close_at:
                    break
            else:
                assert close_at is None, f"Execution did not reach {close_at}"
        finally:
            await stream.aclose()
        assert await app.drain_recovery_cleanups(timeout_s=5)
        assert await app.drain_environment_cleanups(timeout_s=5)
        # Give async-generator finalizers a chance too: the release must not
        # depend on an untracked background finalizer racing this public replay.
        await asyncio.sleep(0.05)
        durable = await store.load_events(session.id)
        assert sum(e.type is EventType.SESSION_INTERRUPTED for e in durable) == 1
        assert not any(e.type is EventType.SESSION_FAILED for e in durable)
        assert (await store.load(session.id)).status is SessionStatus.INTERRUPTED
        parked = await store.load_continuation_ticket(
            session.id,
            session_instance_id=session.instance_id,
            registration_key="execution-wait:" + continuation_digest(wait.operation),
        )
        assert parked.ticket.state == "WAITING"
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
        assert not any(e.type is EventType.SESSION_FAILED for e in replayed)
        assert len(provider.requests) == 1
    finally:
        if app is not None:
            await app.aclose()
        await source.close()
        if backend == "sqlite":
            await store.close()
