"""Disposed failed-run allocations need the same fresh tool namespace as completion."""

import asyncio

import pytest
from tests._tool_admission_fixtures import (
    SimulatedToolFactory,
    simulated_tool_executables,  # noqa: F401
)
from tests.core.browser_session_support import _FakeBrowserBackend
from tests.core.test_environment_allocation_recovery import _FakeRemoteProvider

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest
from cayu.environments import EnvironmentSpec
from cayu.environments._finalization_disposal import checkpoint_finalization_disposal
from cayu.environments.bindings import BoundWorkspace, WorkspaceBinding
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.tools.browser_session import BrowserSessionTool

pytestmark = pytest.mark.usefixtures("simulated_tool_executables")


@pytest.mark.parametrize("first_outcome", ["completed", "failed"])
def test_disposed_terminal_allocation_starts_fresh_browser_accounting(first_outcome):
    async def scenario():
        remote = _FakeRemoteProvider()
        backend = _FakeBrowserBackend()
        disposed = []

        class Factory(SimulatedToolFactory):
            async def is_allocation_disposed(self, request):
                return request.reconnect_metadata["resource_name"] not in remote.resources

            async def recover_finalization_disposal(self, request, state):
                assert state == {"resource_name": request.reconnect_metadata["resource_name"]}
                remote.resources.pop(state["resource_name"], None)

            def _result(self, request, resource, **kwargs):
                result = super()._result(request, resource, **kwargs)

                class Binding(WorkspaceBinding):
                    async def bind(self, workspace, runner, **unused):
                        return BoundWorkspace(workspace=workspace, runner=runner)

                    async def finalize(self, bound, *, outcome=None, metadata=None):
                        if outcome != "interrupted":
                            await checkpoint_finalization_disposal(
                                {
                                    "resource_name": resource.resource_name,
                                }
                            )
                            remote.resources.pop(resource.resource_name, None)
                            disposed.append(resource.resource_name)
                        return None

                    def _completion_requires_successful_finalization(self, bound):
                        return True

                    def _completion_finalization_recovery_state(self, bound):
                        return {"resource_name": resource.resource_name}

                result.environment.binding = Binding()
                return result

        class Provider(ModelProvider):
            name = "fake"

            def __init__(self):
                self.count = 0

            async def stream(self, request):
                self.count += 1
                if self.count in (1, 3):
                    yield ModelStreamEvent.tool_call(
                        id=f"browser-call-{self.count}",
                        name="browser_session",
                        arguments={
                            "operation": "navigate",
                            "url": "https://example.test/form",
                            "operation_id": f"navigate-{self.count}",
                        },
                    )
                    yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                elif self.count == 2 and first_outcome == "failed":
                    raise RuntimeError("model service unavailable")
                else:
                    yield ModelStreamEvent.completed({"finish_reason": "stop"})

        factory = Factory(remote)
        app = CayuApp(enable_logging=False)
        app.register_provider(Provider(), default=True)
        app.register_environment_factory(EnvironmentSpec(name="browser"), factory, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[BrowserSessionTool(_backend=backend)],
        )
        first = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="browser-after-disposal",
                    agent_name="assistant",
                    messages=[Message.text("user", "open a page")],
                )
            )
        ]
        assert first[-1].type is (
            EventType.SESSION_FAILED if first_outcome == "failed" else EventType.SESSION_COMPLETED
        ), first[-1].payload
        first_results = [
            event.payload["result"]
            for event in first
            if event.type is EventType.TOOL_CALL_COMPLETED
        ]
        assert len(first_results) == 1 and not first_results[0]["is_error"], first_results
        assert len(disposed) == 1 and not remote.resources
        private = runtime_checkpoint_session_store(app.session_store)
        checkpoint = await private.load_checkpoint("browser-after-disposal")
        if first_outcome == "failed":
            assert checkpoint["environment_factory_retired_disposals"]["browser"]
            assert not checkpoint.get("environment_factory_reconnect")
        resumed = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="browser-after-disposal",
                    messages=[Message.text("user", "open another page")],
                )
            )
        ]
        assert resumed[-1].type is EventType.SESSION_COMPLETED, resumed[-1].payload
        assert len(remote.create_calls) == 2
        results = [
            event.payload["result"]
            for event in resumed
            if event.type in (EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED)
        ]
        assert len(results) == 1
        after = await private.load_checkpoint("browser-after-disposal")
        assert not results[0]["is_error"], (
            "Fresh allocation was refused before browser dispatch",
            results[0]["structured"].get("error"),
            after.get("environment_factory_allocation_generations"),
            len(backend.calls),
        )
        assert len(backend.calls) == 2

    asyncio.run(scenario())
