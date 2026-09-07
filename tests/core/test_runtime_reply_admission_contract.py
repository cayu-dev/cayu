"""Runtime-first integration must distinguish draft events from accepted replies."""

from __future__ import annotations

import asyncio

import pytest

from cayu import (
    AgentSpec,
    BeforeStopDecision,
    CayuApp,
    EventType,
    InMemorySessionStore,
    LoopPolicy,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SessionStatus,
    SQLiteSessionStore,
    StructuredOutputSpec,
    run_to_completion,
    scripted_structured_output,
)


class _RejectFirstDraft(LoopPolicy):
    def __init__(self):
        self.calls = 0

    async def before_stop(self, context):
        self.calls += 1
        if self.calls == 1:
            return BeforeStopDecision.continue_with(
                Message.text("user", "Produce a reply supported by the available evidence."),
                reason="unsupported_reply",
            )
        return BeforeStopDecision.complete()


@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "sqlite"])
@pytest.mark.parametrize("max_steps", [1, 2])
def test_reply_admission_requires_successful_invocation_not_nonempty_text(
    tmp_path, persistent, max_steps
):
    async def scenario():
        store = (
            SQLiteSessionStore(tmp_path / "reply.sqlite3") if persistent else InMemorySessionStore()
        )
        policy = _RejectFirstDraft()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta(text),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                    for text in ("unsupported draft", "accepted reply")
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="agent", model="scripted"), loop_policies=[policy])
        try:
            async with asyncio.timeout(15):
                outcome = await run_to_completion(
                    app,
                    RunRequest(
                        agent_name="agent",
                        session_id="reply",
                        max_steps=max_steps,
                        messages=[Message.text("user", "Please help.")],
                    ),
                )
            # Streaming model output is observable before the completion gate.
            draft_index = next(
                i
                for i, e in enumerate(outcome.events)
                if e.type == EventType.MODEL_TEXT_DELTA
                and e.payload["delta"] == "unsupported draft"
            )
            gate_index = next(
                i
                for i, e in enumerate(outcome.events)
                if e.type == "custom.loop.before_stop.selected"
            )
            assert draft_index < gate_index
            assert policy.calls == max_steps
            if max_steps == 1:
                assert outcome.status is SessionStatus.INTERRUPTED
                assert not outcome.ok
                # A diagnostic final_text is not an accepted customer response.
                assert outcome.final_text == "unsupported draft"
                assert not any(e.type == EventType.SESSION_COMPLETED for e in outcome.events)
                transcript = await store.load_transcript("reply")
                assert len(transcript) == 2  # no correction appended beyond the limit
            else:
                assert outcome.ok
                assert outcome.final_text == "accepted reply"
                transcript = await store.load_transcript("reply")
                assert len(transcript) == 4
        finally:
            if persistent:
                await store.close()

    asyncio.run(scenario())


def test_structured_output_owns_completion_instead_of_generic_before_stop_gate():
    async def scenario():
        policy = _RejectFirstDraft()
        app = CayuApp(enable_logging=False)
        app.register_provider(
            ScriptedModelProvider([scripted_structured_output({"answer": "schema-valid"})]),
            default=True,
        )
        app.register_agent(AgentSpec(name="agent", model="scripted"), loop_policies=[policy])
        async with asyncio.timeout(15):
            outcome = await run_to_completion(
                app,
                RunRequest(
                    agent_name="agent",
                    session_id="structured",
                    messages=[Message.text("user", "Help.")],
                    structured_output=StructuredOutputSpec(
                        json_schema={
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        }
                    ),
                ),
            )
        assert outcome.ok
        assert outcome.structured_output.output == {"answer": "schema-valid"}
        assert policy.calls == 0

    asyncio.run(scenario())
