from __future__ import annotations

import asyncio

from cayu import (
    AgentSpec,
    BeforeStopContext,
    BeforeStopDecision,
    CayuApp,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    LoopPolicy,
    Message,
    ModelStreamEvent,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SessionIdentity,
    SessionStatus,
)


def test_public_resume_preserves_non_deepcopyable_request_loop_policy() -> None:
    class StatefulPolicy(LoopPolicy):
        def __init__(self) -> None:
            self.calls = 0

        def __deepcopy__(self, _memo):
            raise AssertionError("request loop policies must not be deep-copied")

        async def before_stop(self, _context: BeforeStopContext) -> BeforeStopDecision:
            self.calls += 1
            return BeforeStopDecision.complete()

    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("continued"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
        initial = Message.text("user", "start")
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="stateful-policy-resume",
                messages=[initial],
            ),
            identity=SessionIdentity(
                provider_name=provider.name,
                model="scripted-model",
            ),
        )
        await store.append_transcript_messages("stateful-policy-resume", [initial])
        await store.update_status("stateful-policy-resume", SessionStatus.RUNNING)
        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="stateful-policy-resume")
        )
        assert recovered.status is SessionStatus.INTERRUPTED

        policy = StatefulPolicy()
        events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="stateful-policy-resume",
                    messages=[Message.text("user", "continue")],
                    loop_policies=(policy,),
                )
            )
        ]

        assert events[-1].type == "session.completed"
        assert policy.calls == 1

    asyncio.run(scenario())
