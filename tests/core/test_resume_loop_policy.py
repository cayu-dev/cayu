from __future__ import annotations

import asyncio

from tests.core._execution_profile_fixtures import (
    checkpoint_with_rebound_test_invocation_profile,
    profiled_session_identity,
    runtime_interaction_started_event,
)

from cayu import (
    AgentSpec,
    BeforeStopContext,
    BeforeStopDecision,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    LoopPolicy,
    Message,
    ModelStreamEvent,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SessionStatus,
)


def test_public_resume_preserves_non_deepcopyable_request_loop_policy() -> None:
    class StatefulPolicy(LoopPolicy):
        def __init__(self) -> None:
            self.calls = 0

        def __deepcopy__(self, _memo):
            raise AssertionError("request loop policies must not be deep-copied")

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:resume-loop-policy:stateful-policy",
                behavior_version="1",
                implementation_version="1",
            )

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
        policy = StatefulPolicy()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="stateful-policy-resume",
                messages=[initial],
            ),
            identity=profiled_session_identity(
                provider_name=provider.name,
                model="scripted-model",
                invocation_loop_policies=(policy,),
            ),
        )
        interaction_id = "interaction_stateful_policy_resume"
        await store.transition_status_and_checkpoint(
            "stateful-policy-resume",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
            checkpoint_transform=lambda session, checkpoint: (
                checkpoint_with_rebound_test_invocation_profile(
                    session,
                    checkpoint,
                    interaction_id=interaction_id,
                )
            ),
            interaction_started_event=runtime_interaction_started_event(
                app,
                session_id="stateful-policy-resume",
                interaction_id=interaction_id,
                agent_name="assistant",
            ),
            interaction_source_messages=[initial],
        )
        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="stateful-policy-resume")
        )
        assert recovered.status is SessionStatus.INTERRUPTED

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
