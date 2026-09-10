"""Ordinary admission after browser cleanup and protected human review.

Browser transport is synthetic; app execution, review, cleanup, checkpoint CAS,
projection, and SQLite reopen use the production paths.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from tests._tool_admission_fixtures import (
    SimulatedToolFactory,
    simulated_tool_executables,  # noqa: F401
)
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_session import _FakeBrowserBackend
from tests.core.test_environment_allocation_recovery import _FakeRemoteProvider
from tests.core.test_human_review import (
    CONTEXT,
    QUESTION,
    ApprovalPolicy,
    RecordingTool,
    ReviewPolicy,
    decision,
    identity,
    resolve,
)

from cayu import (
    AgentSpec,
    CayuApp,
    EnvironmentSpec,
    EventType,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
)
from cayu.runtime import (
    AdmitInvocationCommand,
    ResumeRequest,
    SessionRunFenced,
    SessionStatus,
    ToolPolicyDecision,
    ToolPolicyResult,
)
from cayu.runtime import _invocation_lifecycle as lifecycle
from cayu.runtime._browser_control_bootstrap import BrowserGuestBootstrap
from cayu.runtime._browser_control_checkpoint import (
    BrowserControlCheckpointMutation,
    browser_control_checkpoint_mutation_scope,
)
from cayu.runtime._browser_control_service import BrowserControlService
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime.browser_control import BrowserControlCheckpoint
from cayu.runtime.browser_control_config import BrowserControlConfig
from cayu.runtime.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY
from cayu.tools.browser_session import BrowserSessionTool, _RunnerBrowserSessionBackend
from cayu.tools.user_input import UserInputTool

pytestmark = pytest.mark.usefixtures("simulated_tool_executables")


class _ReviewAfterBrowserPolicy(ApprovalPolicy):
    async def authorize(self, request):
        if request.tool_name == "side_effect":
            return await super().authorize(request)
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


@pytest.mark.parametrize("restart", [False, True], ids=["same-worker", "new-worker"])
@pytest.mark.parametrize("approval", [False, True], ids=["user-input", "approval"])
@pytest.mark.parametrize("mutation", [None, "application", "browser"])
def test_completed_browser_session_ordinary_resume(
    tmp_path, monkeypatch, restart, approval, mutation
):
    async def scenario():
        path = tmp_path / "sessions.sqlite"
        store = SQLiteSessionStore(path)
        fake = _FakeBrowserBackend()
        remote = _FakeRemoteProvider()

        async def preflight(backend, ctx, args):
            return await fake.preflight(ctx, args)

        async def execute(backend, ctx, args):
            response = await fake.execute(ctx, args)
            return (
                replace(response, profile_output_protected=True)
                if response.observation is not None
                else response
            )

        bound = False

        async def bootstrap(service, ctx, *, backend, browser_session_id, arguments):
            nonlocal bound
            if bound:
                return
            bound = True
            runtime = app._browser_control_runtime
            assert runtime is not None
            allocation = BrowserGuestBootstrap.allocation_for_invocation(
                ctx,
                purpose=operator_purpose(),
                browser_session_id=browser_session_id,
                arguments=arguments,
            )
            await runtime.coordinator.bind_guest(
                allocation=allocation, worker_instance_id="vw_" + "a" * 32
            )

        monkeypatch.setattr(_RunnerBrowserSessionBackend, "preflight", preflight)
        monkeypatch.setattr(_RunnerBrowserSessionBackend, "execute", execute)
        monkeypatch.setattr(BrowserControlService, "bootstrap", bootstrap)

        class Provider(ScriptedModelProvider):
            execution_profile_identity = identity("browser-resume-provider")
            count = 0

            async def stream(self, request):
                self.count += 1
                if self.count == 1:
                    name, args = (
                        "browser_session",
                        {
                            "operation": "navigate",
                            "url": "https://example.test/",
                            "operation_id": "open",
                        },
                    )
                elif self.count == 2:
                    name, args = (
                        "browser_session",
                        {
                            "operation": "close",
                            "session_id": fake.session_id,
                            "operation_id": "close",
                        },
                    )
                elif self.count == 3:
                    name, args = (
                        ("side_effect", {"value": "morning"})
                        if approval
                        else ("ask_user", {"question": QUESTION})
                    )
                else:
                    yield ModelStreamEvent.text_delta("done")
                    yield ModelStreamEvent.completed({"finish_reason": "stop"})
                    return
                yield ModelStreamEvent.tool_call(id=f"call-{self.count}", name=name, arguments=args)
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

        class Factory(SimulatedToolFactory):
            execution_profile_identity = identity("browser-resume-factory")

        def make_app(*, resumed=False):
            provider = Provider([])
            if resumed:
                provider.count = 4
            app = CayuApp(
                session_store=store,
                enable_logging=False,
                human_review_policy=ReviewPolicy(),
                browser_control=BrowserControlConfig(
                    purpose=operator_purpose(),
                    policy=Policy(True),
                    guest_endpoint="wss://control.test/guest",
                ),
            )
            app.register_provider(provider, default=True)
            app.register_environment_factory(
                EnvironmentSpec(
                    name="browser",
                    execution_profile_identity=identity("browser-resume-environment"),
                ),
                Factory(remote),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="model"),
                tools=[BrowserSessionTool(), RecordingTool() if approval else UserInputTool()],
                tool_policy=_ReviewAfterBrowserPolicy() if approval else None,
            )
            return app, provider

        app, provider = make_app()
        try:
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="review-session",
                        messages=[Message.text("user", "prepare a draft")],
                    )
                )
            ]
            assert len(fake.calls) == 2
            assert provider.count == 3
            assert QUESTION not in str([event.model_dump() for event in paused])
            view = await app.inspect_human_review("review-session", context=CONTEXT)
            assert view.status == "permitted", view
            assert view.reference is not None
            completed = await resolve(app, decision(view, approval=approval))
            assert completed[-1].type == EventType.SESSION_COMPLETED
            assert provider.count == 4
            source_session = await store.load("review-session")
            assert source_session is not None
            assert source_session.status == SessionStatus.COMPLETED
            private_store = runtime_checkpoint_session_store(store)
            source = await private_store.load_checkpoint("review-session")
            assert source is not None
            controls = BrowserControlCheckpoint.model_validate(
                source[BROWSER_CONTROLS_CHECKPOINT_KEY]
            )
            assert controls.records
            assert all(record.state == "closed" for record in controls.records)
            source_events = await store.load_events("review-session")

            if restart:
                await store.close()
                store = SQLiteSessionStore(path)
                app, provider = make_app(resumed=True)
                private_store = runtime_checkpoint_session_store(store)

            # Generic callbacks retain the private root without seeing it.
            observed = []

            def ordinary_transform(_session, checkpoint):
                observed.append(checkpoint)
                return checkpoint

            await store.transform_checkpoint("review-session", ordinary_transform)
            assert BROWSER_CONTROLS_CHECKPOINT_KEY not in observed[0]
            assert await private_store.load_checkpoint("review-session") == source

            apply = lifecycle.apply_invocation_lifecycle_command
            prepared = []
            before_apply = []

            async def change_after_preparation(runtime_store, command):
                if isinstance(command, AdmitInvocationCommand):
                    prepared.append(command)
                    if mutation == "application":
                        await store.transform_checkpoint(
                            "review-session",
                            lambda _session, checkpoint: {
                                **(checkpoint or {}),
                                "concurrent_application_change": True,
                            },
                        )
                    elif mutation == "browser":
                        original = controls.records[0]
                        desired = controls.replace_record(
                            expected=original,
                            desired=original.model_copy(update={"revision": original.revision + 1}),
                        )
                        with browser_control_checkpoint_mutation_scope(
                            BrowserControlCheckpointMutation("review-session", controls, desired)
                        ):
                            await private_store.checkpoint(
                                "review-session",
                                {
                                    **source,
                                    BROWSER_CONTROLS_CHECKPOINT_KEY: desired.model_dump(
                                        mode="json"
                                    ),
                                },
                            )
                    before_apply.append(await store.load("review-session"))
                return await apply(runtime_store, command)

            monkeypatch.setattr(
                lifecycle, "apply_invocation_lifecycle_command", change_after_preparation
            )

            async def resume():
                return [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="review-session",
                            messages=[Message.text("user", "continue")],
                        )
                    )
                ]

            if mutation is not None:
                with pytest.raises(SessionRunFenced, match="source checkpoint changed"):
                    await resume()
                assert provider.count == 4
                assert await store.load("review-session") == before_apply[0]
                assert before_apply[0].status == SessionStatus.COMPLETED
                assert before_apply[0].run_epoch == source_session.run_epoch
                assert await store.load_events("review-session") == source_events
            else:
                events = await resume()
                assert events[-1].type == EventType.SESSION_COMPLETED
                assert provider.count == 5
                current = await store.load("review-session")
                assert current is not None
                assert current.status == SessionStatus.COMPLETED
                assert current.run_epoch > source_session.run_epoch
                assert len(await store.load_events("review-session")) > len(source_events)
                checkpoint = await private_store.load_checkpoint("review-session")
                assert checkpoint is not None
                assert (
                    checkpoint[BROWSER_CONTROLS_CHECKPOINT_KEY]
                    == source[BROWSER_CONTROLS_CHECKPOINT_KEY]
                )
            assert len(prepared) == 1
            # The typed command scope must not leak into subsequent callbacks.
            observed.clear()
            await store.transform_checkpoint("review-session", ordinary_transform)
            assert BROWSER_CONTROLS_CHECKPOINT_KEY not in observed[0]
        finally:
            await store.close()

    asyncio.run(scenario())
