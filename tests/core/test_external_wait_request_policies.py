"""Request-scoped policies remain real collaborators across external-wait restart."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu import AgentSpec, CayuApp, Message, ModelStreamEvent, RunRequest, ScriptedModelProvider
from cayu.external_waits import ExternalEventWaits
from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import ExecutionProfileMismatchError
from cayu.runtime.loop_policies import BeforeStopDecision, LoopPolicy
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions._session_continuation import ContinuationUnavailable
from cayu.sessions.external_waits import (
    ExternalEventDelivery,
    ExternalWaitConflict,
    ExternalWaitUnavailable,
)


class RequestStopPolicy(LoopPolicy):
    def __init__(self, version="1"):
        self.version = version
        self.calls = 0
        self.interrupted = False

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:external-request-stop",
            behavior_version=self.version,
            implementation_version="1",
        )

    async def before_stop(self, context):
        self.calls += 1
        if self.interrupted:
            return BeforeStopDecision.interrupt("Qualification stop gate")
        return BeforeStopDecision.complete()


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_request_policy_reconstructs_for_recovery_and_event_service(
    backend, tmp_path, request, monkeypatch
):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            provider = ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta(text),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                    for text in ("Submitted", "Received")
                ]
            )

            def app_for(native):
                app = CayuApp(session_store=native, enable_logging=False)
                app.register_provider(provider, default=True)
                app.register_agent(AgentSpec(name="root", model="model"))
                return app

            app = app_for(store)
            policy = RequestStopPolicy()
            adapter = SessionExternalWaitAdapter(app, waits, request_loop_policies=(policy,))
            entered = asyncio.Event()

            async def pause(self, invocation):
                entered.set()
                await asyncio.Event().wait()

            session_id = "external-request-policy-" + uuid4().hex
            run = RunRequest(
                agent_name="root",
                session_id=session_id,
                messages=[Message.text("user", "Submit")],
                loop_policies=(policy,),
            )
            with monkeypatch.context() as patch:
                patch.setattr(_ExternalExecutionToWait, "park", pause)
                task = asyncio.create_task(
                    adapter.run_to_wait(
                        run,
                        registered,
                        context=CONTEXT,
                    )
                )
                try:
                    await asyncio.wait_for(entered.wait(), 30)
                finally:
                    task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled() and task.cancelling() == 1
            assert policy.calls == 1 and len(provider.requests) == 1
            with pytest.raises(ExternalWaitConflict):
                await adapter.run_to_wait(
                    run.model_copy(update={"loop_policies": (RequestStopPolicy("2"),)}),
                    registered,
                    context=CONTEXT,
                )
            await app.aclose()
            await waits.aclose()

            restored = reopen()
            restored_waits = ExternalEventWaits(store=restored, access_policy=Policy())
            restored_app = app_for(restored)
            for policies in ((), (RequestStopPolicy("2"),)):
                wrong = SessionExternalWaitAdapter(
                    restored_app, restored_waits, request_loop_policies=policies
                )
                before = await restored.load(session_id)
                with pytest.raises(ExecutionProfileMismatchError):
                    await wrong.recover_to_wait(registered, context=CONTEXT, inactive_for_seconds=0)
                after = await restored.load(session_id)
                # Existing profile-rejection diagnostics may advance activity;
                # they must not admit an epoch or change execution authority.
                assert after.model_dump(exclude={"last_activity_at"}) == before.model_dump(
                    exclude={"last_activity_at"}
                )
                assert len(provider.requests) == 1
            replacement = RequestStopPolicy()
            recovered = SessionExternalWaitAdapter(
                restored_app, restored_waits, request_loop_policies=(replacement,)
            )
            receipt = await recovered.recover_to_wait(
                registered, context=CONTEXT, inactive_for_seconds=0
            )
            assert (
                await recovered.run_to_wait(
                    run.model_copy(update={"loop_policies": (replacement,)}),
                    registered,
                    context=CONTEXT,
                )
                == receipt
            )
            assert replacement.calls == 1 and len(provider.requests) == 1
            await restored_waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="ready", payload_json='{"done":true}'
                ),
                context=CONTEXT,
            )
            for policies in ((), (RequestStopPolicy("2"),)):
                wrong = SessionExternalWaitAdapter(
                    restored_app, restored_waits, request_loop_policies=policies
                )
                with pytest.raises(ExternalWaitUnavailable) as rejected:
                    await wrong.service_wait(registered, context=CONTEXT)
                assert isinstance(rejected.value.__cause__, ContinuationUnavailable)
                assert "invocation_policies" in str(rejected.value.__cause__.__cause__)
                assert len(provider.requests) == 1
                assert (await restored_waits.inspect(correlation, context=CONTEXT)).pending_handoff
            await recovered.service_wait(registered, context=CONTEXT)
            assert replacement.calls == 2 and len(provider.requests) == 2
            await recovered.service_wait(registered, context=CONTEXT)
            assert replacement.calls == 2 and len(provider.requests) == 2
            await restored_app.aclose()
            await restored_waits.aclose()

    asyncio.run(scenario())
