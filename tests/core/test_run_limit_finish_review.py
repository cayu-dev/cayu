from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from tests.core.test_execution_deadlines import clock as clock
from tests.core.test_run_limit_finish import answer, spec
from tests.core.test_targeted_tool_grants import _GatewayRememberTool
from tests.core.test_tool_discovery import _HostedReplayDiscoveryProvider, _RememberKnowledgeTool
from tests.environments.test_factory_secret_scope import _Factory

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest
from cayu.budgets.run_limits import RunLimits
from cayu.configuration import CayuConfig, RunDefaults
from cayu.deadlines import _TIMERS, ExecutionDeadline
from cayu.environments import EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.messages import ProviderStatePart
from cayu.providers import ModelStreamEvent, OpenAIWebSearch
from cayu.providers.openai import build_openai_payload
from cayu.sessions import ModelFailoverPolicy, ModelTarget
from cayu.tools import TargetedToolGrant
from cayu.tools.exposure import StaticToolExposurePolicy


def _expire_owned_deadline(clock):
    clock.wall += timedelta(seconds=3601)
    clock.mono += 3601
    for timer, _ in _TIMERS.get():
        if timer.when() is not None:
            timer.reschedule(asyncio.get_running_loop().time())


@pytest.mark.parametrize(
    "config", [CayuConfig(), CayuConfig(run=RunDefaults(limits=RunLimits(max_output_tokens=100)))]
)
def test_unrelated_application_config_preserves_unbounded_task_default(config):
    app = CayuApp(config=config, enable_logging=False)
    request = RunRequest(agent_name="agent", task_id="task", messages=[])
    assert app._with_application_run_defaults(request).max_steps is None
    assert (
        app._with_application_run_defaults(request.model_copy(update={"max_steps": 64})).max_steps
        == 64
    )


def test_finish_withdraws_hosted_tools_from_serialized_provider_request():
    class Provider(ScriptedModelProvider):
        def preflight_hosted_tools(self, *, model, hosted_tools, options):
            pass

    provider = Provider(answer("tool"))
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="agent", model="fake"), hosted_tools=[OpenAIWebSearch()])

    async def scenario():
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "Answer now")],
                    max_steps=1,
                    limits=RunLimits(on_limit="finish"),
                    structured_output=spec("tool"),
                )
            )
        ]
        assert any(event.type is EventType.STRUCTURED_OUTPUT_VALIDATED for event in events)
        assert not provider.requests[0].hosted_tools
        assert all(
            tool["type"] != "web_search"
            for tool in build_openai_payload(provider.requests[0])["tools"]
        )
        assert events[-1].type is EventType.SESSION_INTERRUPTED

    asyncio.run(scenario())


def test_finish_with_targeted_grants_and_configured_failover():
    primary = ScriptedModelProvider(answer("tool"), name="primary")
    backup = ScriptedModelProvider(answer("tool"), name="backup")
    tool = _GatewayRememberTool()
    app = CayuApp(enable_logging=False)
    app.register_provider(primary, default=True)
    app.register_provider(backup)
    app.register_agent(
        AgentSpec(name="agent", model="fake"),
        tools=[tool],
        targeted_tool_mode="call_tool",
        tool_exposure_policy=StaticToolExposurePolicy(profile_id="targeted-only", tools=()),
    )

    async def scenario():
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "Answer now")],
                    max_steps=1,
                    limits=RunLimits(on_limit="finish"),
                    structured_output=spec("tool"),
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="fact",
                            tool_id="cayu:remember",
                            max_calls=1,
                            lifetime_seconds=60,
                        ),
                    ),
                    failover=ModelFailoverPolicy(
                        fallbacks=(ModelTarget(provider_name="backup", model="fake"),)
                    ),
                )
            )
        ]
        assert len(primary.requests) == 1 and not backup.requests
        assert any(event.type is EventType.STRUCTURED_OUTPUT_VALIDATED for event in events)
        assert events[-1].type is EventType.SESSION_INTERRUPTED
        assert not tool.calls

    asyncio.run(scenario())


def test_finish_projects_native_discovery_history_before_provider_serialization():
    class Provider(_HostedReplayDiscoveryProvider):
        async def stream(self, request):
            if not self.requests:
                async for event in super().stream(request):
                    yield event
                return
            self.requests.append(request)
            assert request.tool_discovery_projection is None
            build_openai_payload(request)
            assert not any(
                isinstance(part, ProviderStatePart)
                for message in request.messages
                for part in message.content
            )
            for event in answer("tool"):
                yield event

    provider = Provider()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="agent", model="fake-model"),
        tools=[_RememberKnowledgeTool()],
        tool_discovery_mode="openai_tool_search_hosted",
        tool_exposure_policy=StaticToolExposurePolicy(profile_id="discovery-only", tools=()),
    )

    async def scenario():
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "Find and answer")],
                    max_steps=2,
                    limits=RunLimits(on_limit="finish"),
                    structured_output=spec("tool"),
                )
            )
        ]
        assert len(provider.requests) == 2
        assert any(event.type is EventType.STRUCTURED_OUTPUT_VALIDATED for event in events)
        assert events[-1].type is EventType.SESSION_INTERRUPTED

    asyncio.run(scenario())


@pytest.mark.parametrize("structured", [False, True])
def test_deadline_during_admitted_factory_setup_ends_normally_after_cleanup(structured, clock):
    class Factory(_Factory):
        def __init__(self):
            super().__init__()
            self.entered = self.cleaned = False

        async def create(self, request):
            self.entered = True
            _expire_owned_deadline(clock)
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0.01)
                self.cleaned = True

    factory = Factory()
    provider = ScriptedModelProvider([ModelStreamEvent.completed()])
    app = CayuApp(enable_logging=False)
    app.register_environment_factory(EnvironmentSpec(name="managed"), factory, default=True)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="agent", model="fake"))

    async def scenario():
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "Wait")],
                    execution_deadline=ExecutionDeadline.after(3600),
                    structured_output=spec("tool") if structured else None,
                )
            )
        ]
        assert factory.entered and factory.cleaned
        assert not provider.requests
        assert not any(event.type is EventType.SESSION_STARTED for event in events)
        assert events[-1].type is EventType.SESSION_INTERRUPTED
        assert events[-1].payload["limit"] == "execution_deadline"
        assert events[-1].payload["structured_output_pending"] is structured

    asyncio.run(scenario())


@pytest.mark.parametrize("structured", [False, True])
def test_deadline_during_admitted_resume_reconnect_yields_its_exact_terminal(structured, clock):
    class Factory(_Factory):
        entered = cleaned = False

        async def create(self, request):
            if not self.operations:
                return await super().create(request)
            self.entered = True
            _expire_owned_deadline(clock)
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0.01)
                self.cleaned = True

    factory = Factory()
    provider = ScriptedModelProvider(
        [answer("tool") if structured else [ModelStreamEvent.completed()]]
    )
    app = CayuApp(enable_logging=False)
    app.register_environment_factory(EnvironmentSpec(name="managed"), factory, default=True)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="agent", model="fake"))

    async def scenario():
        initial = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="resume-setup-deadline",
                    agent_name="agent",
                    messages=[Message.text("user", "First answer")],
                    execution_deadline=ExecutionDeadline.after(3600),
                    structured_output=spec("tool") if structured else None,
                )
            )
        ]
        assert initial[-1].type is EventType.SESSION_COMPLETED
        previous = await app.session_store.load("resume-setup-deadline")
        events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="resume-setup-deadline",
                    messages=[Message.text("user", "Next answer")],
                    structured_output=spec("tool") if structured else None,
                )
            )
        ]
        assert factory.entered and factory.cleaned
        assert len(provider.requests) == 1
        assert not any(event.type is EventType.SESSION_RESUMED for event in events)
        terminal = events[-1]
        assert terminal.type is EventType.SESSION_INTERRUPTED
        assert terminal.payload["limit"] == "execution_deadline"
        assert terminal.payload["structured_output_pending"] is structured
        assert terminal.payload["failure_evidence"]["run_epoch"] == previous.run_epoch + 1
        stored = await app.session_store.load_events("resume-setup-deadline")
        durable = next(event for event in stored if event.type is EventType.SESSION_INTERRUPTED)
        projected = await app._project_emitted_event_for_public_api(durable)
        assert projected.id == terminal.id

    asyncio.run(scenario())


@pytest.mark.parametrize("gate", ["approval", "user_input"])
@pytest.mark.parametrize("structured", [False, True])
def test_deadline_during_admitted_human_continuation_reconnect_ends_normally(
    gate, structured, clock
):
    from tests.core.test_human_pause_run_limits import _resolve
    from tests.runtime.runtime_tool_support import SideEffectTool

    from cayu.tools import AlwaysRequireApprovalToolPolicy, UserInputTool

    class Factory(_Factory):
        entered = cleaned = False

        async def create(self, request):
            if not self.operations:
                return await super().create(request)
            self.entered = True
            _expire_owned_deadline(clock)
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0.01)
                self.cleaned = True

    async def scenario():
        factory = Factory()
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call",
                        name="side_effect" if gate == "approval" else "ask_user",
                        arguments={"value": "approved"}
                        if gate == "approval"
                        else {"question": "Continue?"},
                    ),
                    ModelStreamEvent.completed(),
                ]
            ]
        )
        app = CayuApp(enable_logging=False)
        app.register_environment_factory(EnvironmentSpec(name="managed"), factory, default=True)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="agent", model="fake"),
            tools=[SideEffectTool(), UserInputTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy() if gate == "approval" else None,
        )
        initial = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="human-reconnect",
                    agent_name="agent",
                    messages=[Message.text("user", "Wait")],
                    execution_deadline=ExecutionDeadline.after(3600),
                    structured_output=spec("tool") if structured else None,
                )
            )
        ]
        assert initial[-1].type is EventType.SESSION_INTERRUPTED
        previous = await app.session_store.load("human-reconnect")
        events = await _resolve(app, "human-reconnect", gate)
        assert factory.entered and factory.cleaned
        assert len(provider.requests) == 1
        assert not any(event.type is EventType.SESSION_RESUMED for event in events)
        terminal = events[-1]
        assert terminal.type is EventType.SESSION_INTERRUPTED
        assert terminal.payload["limit"] == "execution_deadline"
        assert terminal.payload["structured_output_pending"] is structured
        assert terminal.payload["failure_evidence"]["run_epoch"] == previous.run_epoch + 1
        stored = await app.session_store.load_events("human-reconnect")
        durable = next(
            event for event in reversed(stored) if event.type is EventType.SESSION_INTERRUPTED
        )
        assert (await app._project_emitted_event_for_public_api(durable)).id == terminal.id
        await app.aclose()

    asyncio.run(scenario())
