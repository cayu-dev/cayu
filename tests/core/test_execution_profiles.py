from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

import cayu.runtime._session_engine as session_engine_module
from cayu import (
    EXECUTION_PROFILE_METADATA_KEY,
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    Event,
    EventType,
    ExecutionProfileComponentClass,
    ExecutionProfileIdentityAvailability,
    ExecutionProfileMismatchError,
    InMemorySessionStore,
    Message,
    ModelTarget,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SessionIdentity,
    SessionStatus,
    StructuredOutputSpec,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelStreamEvent
from cayu.runtime.execution_profiles import (
    build_execution_profile_identity,
    changed_execution_profile_components,
    execution_profile_from_session_metadata,
)


class RecordingTool(Tool):
    def __init__(self, name: str) -> None:
        self.spec = ToolSpec(
            name=name,
            description="Record execution.",
            input_schema={"type": "object", "properties": {}},
        )
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        return ToolResult(content="recorded")


class PreflightRecordingProvider(ScriptedModelProvider):
    supports_native_structured_output = True

    def __init__(self) -> None:
        super().__init__(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop", "model": "fake-model"}),
            ],
            name="fake",
        )
        self.native_preflight_calls = 0

    def preflight_native_structured_output_schema(self, json_schema: dict) -> None:
        self.native_preflight_calls += 1


def _completed_provider() -> ScriptedModelProvider:
    return ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop", "model": "fake-model"}),
        ],
        name="fake",
    )


async def _collect(events: AsyncIterator[Event]) -> list[Event]:
    return [event async for event in events]


def test_public_resume_rejects_changed_direct_tools_before_work() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        original_provider = _completed_provider()
        original_tool = RecordingTool("original_tool")
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(original_provider, default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[original_tool],
        )

        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-tool-drift",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        before = await store.load("execution-profile-tool-drift")
        assert before is not None

        replacement_provider = _completed_provider()
        replacement_tool = RecordingTool("replacement_tool")
        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[replacement_tool],
        )

        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-tool-drift",
                        messages=[Message.text("user", "second")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.DIRECT_TOOLS,
        )
        assert replacement_provider.requests == []
        assert replacement_tool.calls == []

        with pytest.raises(ExecutionProfileMismatchError):
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-tool-drift",
                        messages=[Message.text("user", "retry")],
                    )
                )
            )

        after = await store.load("execution-profile-tool-drift")
        assert after is not None
        assert after.status == before.status
        assert after.run_epoch == before.run_epoch

        events = await store.load_events("execution-profile-tool-drift")
        rejection = events[-1]
        assert rejection.type == EventType.SESSION_EXECUTION_PROFILE_REJECTED
        assert rejection.payload["changed_component_classes"] == ["direct_tools"]
        assert set(rejection.payload) == {
            "candidate_profile_fingerprint",
            "changed_component_classes",
            "expected_profile_fingerprint",
        }
        assert (
            sum(event.type is EventType.SESSION_EXECUTION_PROFILE_REJECTED for event in events) == 1
        )

    asyncio.run(exercise())


def test_public_resume_rejects_reordered_direct_tools_before_work() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        original_provider = _completed_provider()
        original_search = RecordingTool("search")
        original_write = RecordingTool("write")
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(original_provider, default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[original_search, original_write],
        )

        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-tool-order-drift",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        assert [tool["name"] for tool in original_provider.requests[0].tools] == [
            "search",
            "write",
        ]

        replacement_provider = _completed_provider()
        replacement_write = RecordingTool("write")
        replacement_search = RecordingTool("search")
        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[replacement_write, replacement_search],
        )

        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-tool-order-drift",
                        messages=[Message.text("user", "second")],
                    )
                )
            )

        assert caught.value.changed_component_classes == (
            ExecutionProfileComponentClass.DIRECT_TOOLS,
        )
        assert replacement_provider.requests == []
        assert replacement_write.calls == []
        assert replacement_search.calls == []

    asyncio.run(exercise())


def test_changed_tool_profile_rejects_before_replacement_provider_preflight() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        original_app = CayuApp(session_store=store, enable_logging=False)
        original_app.register_provider(_completed_provider(), default=True)
        original_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("original_tool")],
        )
        await _collect(
            original_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-before-provider-preflight",
                    messages=[Message.text("user", "first")],
                )
            )
        )

        replacement_provider = PreflightRecordingProvider()
        replacement_app = CayuApp(session_store=store, enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)
        replacement_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("replacement_tool")],
        )
        with pytest.raises(ExecutionProfileMismatchError):
            await _collect(
                replacement_app.resume(
                    ResumeRequest(
                        session_id="execution-profile-before-provider-preflight",
                        messages=[Message.text("user", "second")],
                        structured_output=StructuredOutputSpec(
                            json_schema={"type": "object"},
                            strategy="native",
                        ),
                    )
                )
            )

        assert replacement_provider.native_preflight_calls == 0
        assert replacement_provider.requests == []

    asyncio.run(exercise())


def test_unavailable_required_component_is_not_compatible_with_itself() -> None:
    profile = build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version=None,
        provider_name="fake",
        model="fake-model",
        durable_system_prompt=None,
        direct_tools=[],
    )

    assert (
        profile.component(ExecutionProfileComponentClass.RUNTIME).availability
        is ExecutionProfileIdentityAvailability.UNAVAILABLE
    )
    assert changed_execution_profile_components(profile, profile) == (
        ExecutionProfileComponentClass.RUNTIME,
    )


def test_public_run_fails_closed_before_work_when_required_identity_is_unavailable(
    monkeypatch,
) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        provider = _completed_provider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        monkeypatch.setattr(session_engine_module, "_runtime_version", lambda: None)

        with pytest.raises(RuntimeError, match="unavailable required components: runtime"):
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="execution-profile-unavailable-run",
                        messages=[Message.text("user", "first")],
                    )
                )
            )

        assert await store.load("execution-profile-unavailable-run") is None
        assert provider.requests == []

    asyncio.run(exercise())


def test_creation_profile_truthfully_identifies_rendered_workspace_system_projection() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        provider = _completed_provider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="workspace"),
                workspace_instructions="Use the durable workspace contract.",
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                system_prompt="Follow the agent contract.",
            )
        )
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-system-projection",
                    messages=[Message.text("user", "first")],
                )
            )
        )

        session = await store.load("execution-profile-system-projection")
        transcript = await store.load_transcript("execution-profile-system-projection")
        assert session is not None
        rendered_system_prompt = transcript[0].content[0].text
        assert "Follow the agent contract." in rendered_system_prompt
        assert "Use the durable workspace contract." in rendered_system_prompt
        expected = build_execution_profile_identity(
            runtime_name=session.runtime_name or "cayu",
            runtime_version=session.runtime_version,
            provider_name=session.provider_name,
            model=session.model,
            durable_system_prompt=rendered_system_prompt,
            direct_tools=[],
        )
        persisted = execution_profile_from_session_metadata(session.metadata)
        assert persisted.component(
            ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION
        ) == expected.component(ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION)

    asyncio.run(exercise())


def test_public_resume_accepts_same_structural_profile_without_transition() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        first_provider = _completed_provider()
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(first_provider, default=True)
        first_app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                system_prompt="private durable instructions",
            ),
            tools=[RecordingTool("stable_tool")],
        )
        await _collect(
            first_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-exact-reuse",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        after_run = await store.load("execution-profile-exact-reuse")
        assert after_run is not None
        profile_record = after_run.metadata[EXECUTION_PROFILE_METADATA_KEY]
        serialized_record = json.dumps(profile_record, sort_keys=True)
        assert "private durable instructions" not in serialized_record
        assert "stable_tool" not in serialized_record

        second_provider = _completed_provider()
        second_app = CayuApp(session_store=store, enable_logging=False)
        second_app.register_provider(second_provider, default=True)
        second_app.register_agent(
            AgentSpec(
                name="assistant",
                model="fake-model",
                # Resumes use the already-durable projection, not this live value.
                system_prompt="different live prompt that is not re-injected",
            ),
            tools=[RecordingTool("stable_tool")],
        )
        resume_events = await _collect(
            second_app.resume(
                ResumeRequest(
                    session_id="execution-profile-exact-reuse",
                    messages=[Message.text("user", "second")],
                )
            )
        )

        assert len(second_provider.requests) == 1
        assert EventType.SESSION_EXECUTION_PROFILE_REJECTED not in {
            event.type for event in resume_events
        }
        after_resume = await store.load("execution-profile-exact-reuse")
        assert after_resume is not None
        assert after_resume.metadata[EXECUTION_PROFILE_METADATA_KEY] == profile_record

    asyncio.run(exercise())


def test_existing_model_switch_advances_expected_profile_for_later_resume() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        source = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("source"),
                ModelStreamEvent.completed({"finish_reason": "stop", "model": "source-model"}),
            ],
            name="source",
        )
        target = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("target first"),
                    ModelStreamEvent.completed({"finish_reason": "stop", "model": "target-model"}),
                ],
                [
                    ModelStreamEvent.text_delta("target second"),
                    ModelStreamEvent.completed({"finish_reason": "stop", "model": "target-model"}),
                ],
            ],
            name="target",
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(source, default=True)
        app.register_provider(target)
        app.register_agent(
            AgentSpec(name="assistant", model="source-model"),
            tools=[RecordingTool("stable_tool")],
        )
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-model-switch",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        switched = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="execution-profile-model-switch",
                    messages=[Message.text("user", "switch")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        )
        resumed = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="execution-profile-model-switch",
                    messages=[Message.text("user", "continue")],
                )
            )
        )

        assert EventType.SESSION_MODEL_SWITCHED in {event.type for event in switched}
        assert EventType.SESSION_EXECUTION_PROFILE_REJECTED not in {event.type for event in resumed}
        assert len(target.requests) == 2

    asyncio.run(exercise())


def test_public_resume_fails_closed_without_a_durable_profile() -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="execution-profile-missing",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.update_status("execution-profile-missing", SessionStatus.COMPLETED)
        provider = _completed_provider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[RecordingTool("stable_tool")],
        )

        with pytest.raises(ValueError, match="no durable execution-profile identity"):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-missing",
                        messages=[Message.text("user", "continue")],
                    )
                )
            )
        assert provider.requests == []
        session = await store.load("execution-profile-missing")
        assert session is not None
        assert session.status is SessionStatus.COMPLETED

    asyncio.run(exercise())


def test_profile_resolution_failure_does_not_admit_resume(monkeypatch) -> None:
    async def exercise() -> None:
        store = InMemorySessionStore()
        provider = _completed_provider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="execution-profile-resolution-failure",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        before = await store.load("execution-profile-resolution-failure")
        assert before is not None
        provider.requests.clear()

        def fail_resolution(**_kwargs):
            raise RuntimeError("profile resolution failed")

        monkeypatch.setattr(
            session_engine_module,
            "_execution_profile_identity",
            fail_resolution,
        )
        with pytest.raises(RuntimeError, match="profile resolution failed"):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="execution-profile-resolution-failure",
                        messages=[Message.text("user", "second")],
                    )
                )
            )

        after = await store.load("execution-profile-resolution-failure")
        assert after is not None
        assert after.status == before.status
        assert after.run_epoch == before.run_epoch
        assert provider.requests == []

    asyncio.run(exercise())
