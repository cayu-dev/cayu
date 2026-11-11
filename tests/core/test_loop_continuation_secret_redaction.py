"""Runtime-authored loop continuations must obey transcript secret redaction."""

import asyncio
import json

import pytest
from tests.core.test_targeted_tool_grants import _codec

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime.loop_policies import BeforeStopDecision, LoopPolicy
from cayu.sessions import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.vaults import REDACTED_SECRET, SecretRedactor


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("with_secret", [False, True])
def test_before_stop_continuation_is_redacted_before_transcript_write(
    tmp_path, backend, with_secret
):
    async def scenario():
        secret = "loop-continuation-private-canary-42979"
        reminder = "Retry the operation for credential " + secret
        store = (
            InMemorySessionStore(public_authority_alias_codec=_codec())
            if backend == "memory"
            else SQLiteSessionStore(
                tmp_path / "loop-redaction.sqlite", public_authority_alias_codec=_codec()
            )
        )
        provider = ScriptedModelProvider(
            [
                [ModelStreamEvent.text_delta("Not done"), ModelStreamEvent.completed({})],
                [ModelStreamEvent.text_delta("Still not done"), ModelStreamEvent.completed({})],
            ]
        )

        class ContinueOnce(LoopPolicy):
            async def before_stop(self, context):
                if context.step == 1:
                    return BeforeStopDecision.continue_with(Message.text("user", reminder))
                return BeforeStopDecision.complete()

        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret if with_secret else None),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"))
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="loop-continuation-redaction",
                        messages=[Message.text("user", "Perform the operation")],
                        max_steps=3,
                        loop_policies=(ContinueOnce(),),
                    )
                )
            ]
            assert len(provider.requests) == 2
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert (
                await store.load("loop-continuation-redaction")
            ).status is SessionStatus.COMPLETED
            expected = reminder.replace(secret, REDACTED_SECRET) if with_secret else reminder
            # The provider boundary already masks this message, but the durable
            # transcript must not keep the original secret either.
            assert provider.requests[1].messages[2].content[0].text == expected
            transcript = await store.load_transcript("loop-continuation-redaction")
            assert transcript[2].content[0].text == expected
            if with_secret:
                assert secret not in json.dumps([m.model_dump(mode="json") for m in transcript])
        finally:
            await app.aclose()

    asyncio.run(scenario())
