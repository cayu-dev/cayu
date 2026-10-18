"""Controller-attached fork overlays: instructions, served tools and memory per fork."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.events import Event
from cayu.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
)
from cayu.fork_controllers import ForkController, ForkControllerDenied
from cayu.knowledge.records import KnowledgeStatus
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.mcp import (
    McpInitializeResult,
    McpResourceDefinition,
    McpResourceResult,
    McpServerSpec,
    McpSession,
    McpToolDefinition,
    McpToolResult,
    McpToolset,
)
from cayu.messages import MessageRole
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import ExecutionProfileMismatchError
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import ExecutionProfileAdoptionRejected, ExecutionProfilePolicy
from cayu.sessions import (
    ForkOverlay,
    ForkOverlayMemoryItem,
    ForkOverlayToolGrant,
    ForkSessionRequest,
    SessionStatus,
)
from cayu.sessions.fork_overlays import FORK_OVERLAY_METADATA_KEY, fork_overlay_record_from_metadata
from cayu.sessions.memory import InMemorySessionStore
from cayu.storage import SQLiteSessionStore
from cayu.storage.knowledge_memory import InMemoryKnowledgeStore
from cayu.tools import ToolCapabilityCeiling
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.policy import (
    AlwaysRequireApprovalToolPolicy,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.vaults import SecretRedactor, SecretRef

HOST_AGENT = "finance-monthend"
CONTROLLER = "compass"
SUBJECT = "svc:compass"
LOOKUP = "mcp__compass__lookup"
BODY = f"{HOST_AGENT}@fork-controller:{CONTROLLER}"
INSTRUCTIONS = "Investigate the September accrual variance only. Report one finding."
MEMORY = ForkOverlayMemoryItem(
    id="aug-close",
    content="August closed with a 2% accrual variance.",
    source="compass:findings",
)


class ScriptedProvider(ModelProvider):
    name = "fake"

    def __init__(self, batches: list[list[ModelStreamEvent]]) -> None:
        self.batches = batches
        self.requests: list[ModelRequest] = []

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:fork-overlays:scripted-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        for event in self.batches[len(self.requests) - 1]:
            yield event


class ControllerMcpSession(McpSession):
    """The controller's remote MCP server; it records exactly what reaches it."""

    def __init__(self, definitions: tuple[McpToolDefinition, ...]) -> None:
        self.definitions = definitions
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def initialize_result(self) -> McpInitializeResult:
        return McpInitializeResult(protocol_version="2025-06-18")

    async def list_tools(self) -> tuple[McpToolDefinition, ...]:
        return self.definitions

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        self.calls.append((name, arguments))
        return McpToolResult(content=[{"type": "text", "text": "accrual variance is 3%"}])

    async def list_resources(self) -> tuple[McpResourceDefinition, ...]:
        return ()

    async def read_resource(self, uri: str) -> McpResourceResult:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class PostJournal(Tool):
    spec = ToolSpec(
        name="post_journal",
        description="Post a journal entry to the ledger.",
        input_schema={"type": "object"},
        effect=ToolEffect.EXTERNAL,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:fork-overlays:post-journal",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[ToolContext] = []

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        self.contexts.append(ctx)
        return ToolResult(content="posted")


class VoidJournal(PostJournal):
    spec = PostJournal.spec.model_copy(
        update={
            "name": "void_journal",
            "description": "Void a journal entry.",
            "execution_profile_identity": ExecutionProfileBehaviorIdentity(
                name="tests:fork-overlays:void-journal",
                behavior_version="1",
                implementation_version="1",
            ),
        }
    )


class RecordingToolPolicy(ToolPolicy):
    def __init__(self) -> None:
        self.agent_names: list[str] = []

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        self.agent_names.append(request.agent.name)
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class RejectProfileChanges(ExecutionProfilePolicy):
    @property
    def identity(self) -> str:
        return "test:reject-profile-changes:v1"

    async def decide(self, request: ExecutionProfilePolicyRequest) -> ExecutionProfilePolicyResult:
        return ExecutionProfilePolicyResult(
            action=ExecutionProfilePolicyAction.REJECT,
            reason="The host reviews every child profile change.",
        )


def _definitions(
    description: str = "Look up a controller finding.",
    names: tuple[str, ...] = ("lookup",),
) -> tuple[McpToolDefinition, ...]:
    return tuple(
        McpToolDefinition(
            name=name,
            description=description,
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        for name in names
    )


def _controller_toolset(
    session: ControllerMcpSession | None = None,
    **server: Any,
) -> tuple[McpToolset, ControllerMcpSession]:
    session = session or ControllerMcpSession(_definitions())
    spec = McpServerSpec(
        name=CONTROLLER,
        connection_id=CONTROLLER,
        url="https://compass.example/mcp",
        **server,
    )
    return McpToolset(server=spec, session=session, definitions=session.definitions), session


def _completed(text: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta(text),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def _calls(name: str, arguments: dict[str, Any], call_id: str = "call-1") -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(id=call_id, name=name, arguments=arguments),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def _host_app(
    store,
    provider: ScriptedProvider,
    *,
    toolset: McpToolset | None = None,
    enroll: bool = True,
    extra_tools: tuple[Tool, ...] = (),
    enrolled_agents: tuple[str, ...] = (HOST_AGENT,),
    host_tools: tuple[str, ...] | None = None,
    extra_agents: tuple[str, ...] = (),
    hosted_tools: tuple[Any, ...] = (),
    environment: Environment | None = None,
    allow_model_runs: bool = True,
    **app_options: Any,
) -> tuple[CayuApp, PostJournal]:
    app = CayuApp(session_store=store, enable_logging=False, **app_options)
    app.register_provider(provider, default=True)
    if environment is not None:
        app.register_environment(environment, default=True)
    journal = PostJournal()
    app.register_agent(
        AgentSpec(name=HOST_AGENT, model="fake-model", system_prompt="Close the books."),
        tools=[journal, *extra_tools],
        **({"hosted_tools": hosted_tools} if hosted_tools else {}),
    )
    for name in extra_agents:
        app.register_agent(AgentSpec(name=name, model="fake-model"))
    if enroll:
        # ``host_tools=None`` keeps the enrollment default, which allows no host tool.
        allowance: dict[str, Any] = {} if host_tools is None else {"host_tools": host_tools}
        app.enroll_fork_controller(
            ForkController(
                name=CONTROLLER,
                subjects=(SUBJECT,),
                agents=enrolled_agents,
                mcp_toolsets=() if toolset is None else (toolset,),
                # These scenarios run their forks; the gate's default is tested on its own.
                allow_model_runs=allow_model_runs,
                **allowance,
            )
        )
    return app, journal


def _overlay(**changes: Any) -> ForkOverlay:
    values: dict[str, Any] = {
        "controller": CONTROLLER,
        "instructions": INSTRUCTIONS,
        "tools": (ForkOverlayToolGrant(tool_name=LOOKUP),),
        "memory": (MEMORY,),
    }
    values.update(changes)
    return ForkOverlay(**values)


def _fork_request(
    session_id: str | None = "compass-child",
    *,
    overlay: ForkOverlay | None = None,
    subject: str = SUBJECT,
    **changes: Any,
) -> ForkSessionRequest:
    values: dict[str, Any] = {
        "source_session_id": "trunk",
        "session_id": session_id,
        "tool_capability_ceiling": ToolCapabilityCeiling(tool_names=()),
        "execution_profile_selection": "current_child",
        "profile_adoption": ExecutionProfileAdoptionIntent(
            idempotency_key="compass-investigation-1",
            reason="Compass investigates one month-end question.",
            requested_by=ResolutionActor(subject=subject, source=ResolutionActorSource.REQUEST),
        ),
        "overlay": _overlay() if overlay is None else overlay,
    }
    values.update(changes)
    return ForkSessionRequest(**values)


async def _drain(stream) -> list[Any]:
    return [event async for event in stream]


async def _run_trunk(app: CayuApp) -> None:
    await _drain(
        app.run(
            RunRequest(
                session_id="trunk",
                agent_name=HOST_AGENT,
                messages=[Message.text("user", "Close September.")],
            )
        )
    )


def _text(message: Message) -> str:
    return "".join(getattr(part, "text", "") for part in message.content)


def test_outside_controller_forks_with_instructions_served_tool_and_memory() -> None:
    async def scenario():
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider(
            [
                _completed("September closed."),
                _calls(LOOKUP, {"q": "accruals"}),
                _completed("Finding: accruals drifted by 3%."),
            ]
        )
        store = InMemorySessionStore()
        knowledge = InMemoryKnowledgeStore()
        app, journal = _host_app(
            store,
            provider,
            toolset=toolset,
            knowledge_store=knowledge,
            knowledge_access_scope=KnowledgeAccessScope.for_namespace("finance"),
        )
        await _run_trunk(app)
        source_before = await store.load("trunk")
        source_transcript = await store.load_transcript("trunk")

        controller = app.fork_controller_access(SUBJECT)
        fork_events = await _drain(controller.fork(_fork_request()))
        resume_events = await _drain(
            controller.resume(
                ResumeRequest(
                    session_id="compass-child",
                    messages=[Message.text("user", "What changed in accruals?")],
                )
            )
        )
        return (
            app,
            store,
            knowledge,
            provider,
            controller_server,
            journal,
            source_before,
            source_transcript,
            fork_events,
            resume_events,
        )

    (
        app,
        store,
        knowledge,
        provider,
        controller_server,
        journal,
        source_before,
        source_transcript,
        fork_events,
        resume_events,
    ) = asyncio.run(scenario())

    assert [event.type for event in fork_events] == [EventType.SESSION_FORKED]
    forked = fork_events[0].payload
    assert forked["agent_name"] == BODY
    assert forked["fork_overlay_controller"] == CONTROLLER
    assert forked["fork_overlay_fingerprint"] == _overlay().fingerprint
    assert forked["fork_overlay_tool_names"] == [LOOKUP]

    child = asyncio.run(store.load("compass-child"))
    assert child is not None and child.parent_session_id == "trunk"
    assert child.agent_name == BODY
    assert child.status is SessionStatus.COMPLETED
    assert child.tool_capability_ceiling == ToolCapabilityCeiling(tool_names=(LOOKUP,))
    record = fork_overlay_record_from_metadata(child.metadata)
    assert record is not None
    assert record.overlay == _overlay()
    assert record.controller_subject == SUBJECT
    child_events = asyncio.run(store.load_events("compass-child"))
    decision = next(
        event for event in child_events if event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
    )
    assert decision.payload["policy_identity"] == "cayu:fork-controller-overlay-adoption:v1"
    assert decision.payload["actor"]["subject"] == SUBJECT
    assert set(decision.payload["changed_component_classes"]) <= {
        "runtime",
        "direct_tools",
        "tool_implementations",
        "tool_view_grants",
        "effect_authority",
    }

    # The controller's server received only the model's arguments.
    assert controller_server.calls == [("lookup", {"q": "accruals"})]
    assert journal.contexts == []
    assert resume_events[-1].type is EventType.SESSION_COMPLETED

    # The model saw the source history, then the overlay as a separate layer.
    child_requests = provider.requests[1:]
    assert [[tool["name"] for tool in request.tools] for request in child_requests] == [
        [LOOKUP],
        [LOOKUP],
    ]
    for request in child_requests:
        assert request.messages[0].role is MessageRole.SYSTEM
        assert "Close the books." in _text(request.messages[0])
        assert _text(request.messages[1]) == "Close September."
        layer = request.messages[-1]
        assert layer.role is MessageRole.USER
        rendered = _text(layer)
        assert rendered.startswith('<cayu_fork_overlay version="1">')
        payload = json.loads(rendered.splitlines()[1])
        assert payload == {
            "controller": CONTROLLER,
            "instructions": INSTRUCTIONS,
            "memory": [MEMORY.model_dump(mode="json", exclude_none=True)],
            "scope": "this_fork_only",
        }

    # Nothing about the overlay is written to the source, the transcript, or host memory.
    source_after = asyncio.run(store.load("trunk"))
    assert source_after is not None and source_before is not None
    assert source_after.metadata == source_before.metadata
    assert source_after.agent_name == HOST_AGENT
    assert asyncio.run(store.load_transcript("trunk")) == source_transcript
    child_transcript = asyncio.run(store.load_transcript("compass-child"))
    assert not any(
        INSTRUCTIONS in _text(message) or MEMORY.content in _text(message)
        for message in child_transcript
    )
    changes = asyncio.run(
        knowledge.read_changes(
            after_sequence=0,
            access_scope=KnowledgeAccessScope.for_namespace("finance"),
        )
    )
    assert list(changes.changes) == []
    assert app.list_agents() == (HOST_AGENT, BODY)


def test_instructions_and_memory_overlay_keeps_the_inherited_profile() -> None:
    async def scenario():
        provider = ScriptedProvider([_completed("closed"), _completed("finding")])
        store = InMemorySessionStore()
        app, _journal = _host_app(store, provider, host_tools=("post_journal",))
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        overlay = _overlay(tools=())
        await _drain(
            controller.fork(
                ForkSessionRequest(
                    source_session_id="trunk",
                    session_id="memory-only",
                    overlay=overlay,
                )
            )
        )
        await _drain(
            controller.resume(
                ResumeRequest(session_id="memory-only", messages=[Message.text("user", "go")])
            )
        )
        return store, provider, overlay

    store, provider, overlay = asyncio.run(scenario())
    child = asyncio.run(store.load("memory-only"))
    trunk = asyncio.run(store.load("trunk"))
    assert child is not None and trunk is not None
    assert child.agent_name == HOST_AGENT
    assert child.tool_capability_ceiling == trunk.tool_capability_ceiling
    assert [tool["name"] for tool in provider.requests[1].tools] == ["post_journal"]
    assert INSTRUCTIONS in _text(provider.requests[1].messages[-1])
    events = asyncio.run(store.load_events("memory-only"))
    assert [event.type for event in events[:2]] == [
        EventType.SESSION_FORKED,
        EventType.TARGETED_TOOL_GRANT_FORK_RESET,
    ]
    record = fork_overlay_record_from_metadata(child.metadata)
    assert record is not None and record.fingerprint == overlay.fingerprint


def test_only_an_authenticated_enrolled_controller_can_attach_an_overlay() -> None:
    async def scenario():
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider([_completed("closed")])
        store = InMemorySessionStore()
        app, _journal = _host_app(store, provider, toolset=toolset, extra_agents=("payroll",))
        await _run_trunk(app)
        failures: dict[str, BaseException] = {}

        async def capture(name: str, stream) -> None:
            try:
                await _drain(stream)
            except (ForkControllerDenied, ValueError) as exc:
                failures[name] = exc

        await capture("plain_fork", app.fork_session(_fork_request()))
        try:
            app.fork_controller_access("svc:unknown")
        except ForkControllerDenied as exc:
            failures["unknown_subject"] = exc
        controller = app.fork_controller_access(SUBJECT)
        await capture(
            "other_controller",
            controller.fork(_fork_request(overlay=_overlay(controller="other"))),
        )
        await capture("forged_actor", controller.fork(_fork_request(subject="svc:someone-else")))
        await capture(
            "unenrolled_agent",
            controller.fork(_fork_request(agent_name="payroll")),
        )
        return store, controller_server, failures

    store, controller_server, failures = asyncio.run(scenario())
    assert set(failures) == {
        "plain_fork",
        "unknown_subject",
        "other_controller",
        "forged_actor",
        "unenrolled_agent",
    }
    assert all(isinstance(error, ForkControllerDenied) for error in failures.values())
    assert asyncio.run(store.load("compass-child")) is None
    assert controller_server.calls == []


def test_overlay_cannot_widen_host_authority() -> None:
    async def scenario():
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider(
            [
                _completed("closed"),
                _calls("post_journal", {"amount": 100}),
                _completed("I could not post."),
            ]
        )
        store = InMemorySessionStore()
        app, journal = _host_app(store, provider, toolset=toolset)
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        failures: dict[str, BaseException] = {}

        async def capture(name: str, stream) -> None:
            try:
                await _drain(stream)
            except (ForkControllerDenied, ValueError, KeyError) as exc:
                failures[name] = exc

        await capture(
            "unserved_tool",
            controller.fork(
                _fork_request(
                    "unserved",
                    overlay=_overlay(tools=(ForkOverlayToolGrant(tool_name="post_journal"),)),
                )
            ),
        )
        await capture(
            "controller_tool_through_host_ceiling",
            controller.fork(
                _fork_request(
                    "ceiling",
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=(LOOKUP,)),
                )
            ),
        )
        await capture(
            "ordinary_run_of_body",
            app.run(
                RunRequest(
                    session_id="body-run",
                    agent_name=BODY,
                    messages=[Message.text("user", "use the controller tool")],
                )
            ),
        )
        await capture(
            "ordinary_fork_selecting_body",
            app.fork_session(
                ForkSessionRequest(
                    source_session_id="trunk",
                    session_id="body-fork",
                    agent_name=BODY,
                    execution_profile_selection="current_child",
                    profile_adoption=_fork_request().profile_adoption,
                )
            ),
        )

        # The child keeps the host's narrowed ceiling: a host tool outside it never runs.
        await _drain(controller.fork(_fork_request()))
        await capture(
            "host_fork_of_overlay_child",
            app.fork_session(ForkSessionRequest(source_session_id="compass-child")),
        )
        await _drain(
            controller.resume(
                ResumeRequest(
                    session_id="compass-child",
                    messages=[Message.text("user", "Post a journal entry for me.")],
                )
            )
        )
        return store, provider, journal, controller_server, failures

    store, provider, journal, controller_server, failures = asyncio.run(scenario())
    assert set(failures) == {
        "unserved_tool",
        "controller_tool_through_host_ceiling",
        "ordinary_run_of_body",
        "ordinary_fork_selecting_body",
        "host_fork_of_overlay_child",
    }
    assert "not served by the enrolled controller" in str(failures["unserved_tool"])
    assert "never widened" in str(failures["controller_tool_through_host_ceiling"])
    assert isinstance(failures["host_fork_of_overlay_child"], ForkControllerDenied)
    for session_id in ("unserved", "ceiling", "body-run", "body-fork"):
        assert asyncio.run(store.load(session_id)) is None
    assert journal.contexts == []
    assert controller_server.calls == []
    assert [tool["name"] for tool in provider.requests[1].tools] == [LOOKUP]
    events = asyncio.run(store.load_events("compass-child"))
    blocked = [event for event in events if event.type is EventType.TOOL_CALL_BLOCKED]
    assert [event.payload["reason"] for event in blocked] == ["not_exposed_in_request"]


def test_host_policies_govern_overlay_forks_and_tools() -> None:
    async def scenario():
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider([_completed("closed"), _calls(LOOKUP, {"q": "accruals"})])
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name=HOST_AGENT, model="fake-model"),
            tools=[PostJournal()],
            tool_policy=AlwaysRequireApprovalToolPolicy(tools=(LOOKUP,)),
        )
        app.enroll_fork_controller(
            ForkController(
                name=CONTROLLER,
                subjects=(SUBJECT,),
                agents=(HOST_AGENT,),
                mcp_toolsets=(toolset,),
                allow_model_runs=True,
            )
        )
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request()))
        events = await _drain(
            controller.resume(
                ResumeRequest(session_id="compass-child", messages=[Message.text("user", "go")])
            )
        )

        reviewed_store = InMemorySessionStore()
        reviewed_provider = ScriptedProvider([_completed("closed")])
        reviewed_toolset, _ = _controller_toolset()
        reviewed, _journal = _host_app(
            reviewed_store,
            reviewed_provider,
            toolset=reviewed_toolset,
            execution_profile_policy=RejectProfileChanges(),
        )
        await _run_trunk(reviewed)
        rejected: BaseException | None = None
        try:
            await _drain(reviewed.fork_controller_access(SUBJECT).fork(_fork_request()))
        except ExecutionProfileAdoptionRejected as exc:
            rejected = exc
        return store, controller_server, events, rejected, reviewed_store

    store, controller_server, events, rejected, reviewed_store = asyncio.run(scenario())
    # The host's approval policy still gates the controller-served tool.
    assert controller_server.calls == []
    assert events[-1].type is EventType.SESSION_INTERRUPTED
    child = asyncio.run(store.load("compass-child"))
    assert child is not None and child.status is SessionStatus.INTERRUPTED
    # A configured execution-profile policy decides overlay forks instead of the built-in rule.
    assert rejected is not None
    assert asyncio.run(reviewed_store.load("compass-child")) is None


def test_host_tool_policy_sees_the_host_agent_name() -> None:
    async def scenario():
        toolset, _server = _controller_toolset()
        provider = ScriptedProvider(
            [_completed("closed"), _calls(LOOKUP, {"q": "accruals"}), _completed("done")]
        )
        policy = RecordingToolPolicy()
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name=HOST_AGENT, model="fake-model"), tool_policy=policy)
        app.enroll_fork_controller(
            ForkController(
                name=CONTROLLER,
                subjects=(SUBJECT,),
                agents=(HOST_AGENT,),
                mcp_toolsets=(toolset,),
                allow_model_runs=True,
            )
        )
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request()))
        await _drain(
            controller.resume(
                ResumeRequest(session_id="compass-child", messages=[Message.text("user", "go")])
            )
        )
        return policy

    assert asyncio.run(scenario()).agent_names == [HOST_AGENT]


def test_controller_tools_never_receive_host_credentials() -> None:
    session = ControllerMcpSession(_definitions())
    stdio = McpToolset(
        server=McpServerSpec(name=CONTROLLER, connection_id=CONTROLLER, command=["compass-mcp"]),
        session=session,
        definitions=session.definitions,
    )
    with pytest.raises(ValueError, match="would run on the host"):
        ForkController(
            name=CONTROLLER, subjects=(SUBJECT,), agents=(HOST_AGENT,), mcp_toolsets=(stdio,)
        )
    for field_name in ("secret_headers", "secret_env"):
        toolset, _ = _controller_toolset(
            **{field_name: {"X-Token": SecretRef(name="ledger-token")}}
        )
        with pytest.raises(ValueError, match="never receive host credentials"):
            ForkController(
                name=CONTROLLER,
                subjects=(SUBJECT,),
                agents=(HOST_AGENT,),
                mcp_toolsets=(toolset,),
            )

    async def secret_overlay():
        toolset, _server = _controller_toolset()
        provider = ScriptedProvider([_completed("closed")])
        store = InMemorySessionStore()
        app, _journal = _host_app(
            store,
            provider,
            toolset=toolset,
            secret_redactor=SecretRedactor("ledger-token-value"),
        )
        await _run_trunk(app)
        with pytest.raises(ValueError, match="workload secret"):
            await _drain(
                app.fork_controller_access(SUBJECT).fork(
                    _fork_request(
                        overlay=_overlay(
                            memory=(
                                ForkOverlayMemoryItem(id="leak", content="use ledger-token-value"),
                            )
                        )
                    )
                )
            )
        return store

    store = asyncio.run(secret_overlay())
    assert asyncio.run(store.load("compass-child")) is None


def test_restart_reproduces_the_overlay_fork_without_duplicates(tmp_path: Path) -> None:
    database = tmp_path / "host.sqlite"

    async def first_process() -> tuple[str, str]:
        toolset, _server = _controller_toolset()
        app, _journal = _host_app(
            SQLiteSessionStore(database),
            ScriptedProvider([_completed("closed")]),
            toolset=toolset,
        )
        await _run_trunk(app)
        named = await _drain(app.fork_controller_access(SUBJECT).fork(_fork_request()))
        generated = await _drain(app.fork_controller_access(SUBJECT).fork(_fork_request(None)))
        return named[0].id, generated[0].session_id

    async def second_process(fork_event_id: str, generated_session_id: str):
        toolset, controller_server = _controller_toolset()
        store = SQLiteSessionStore(database)
        provider = ScriptedProvider([_calls(LOOKUP, {"q": "accruals"}), _completed("finding")])
        app, _journal = _host_app(store, provider, toolset=toolset)
        controller = app.fork_controller_access(SUBJECT)
        replayed = await _drain(controller.fork(_fork_request()))
        regenerated = await _drain(controller.fork(_fork_request(None)))
        conflict: BaseException | None = None
        try:
            await _drain(controller.fork(_fork_request(overlay=_overlay(instructions="Other."))))
        except RuntimeError as exc:
            conflict = exc
        await _drain(
            controller.resume(
                ResumeRequest(session_id="compass-child", messages=[Message.text("user", "go")])
            )
        )
        listed = await store.list_sessions()
        children = [session for session in listed.sessions if session.parent_session_id == "trunk"]
        forked = [
            event
            for event in await store.load_events("compass-child")
            if event.type is EventType.SESSION_FORKED
        ]
        assert len(forked) == 1
        return (
            replayed,
            regenerated,
            conflict,
            children,
            provider,
            controller_server,
            fork_event_id,
            generated_session_id,
        )

    fork_event_id, generated_session_id = asyncio.run(first_process())
    (
        replayed,
        regenerated,
        conflict,
        children,
        provider,
        controller_server,
        fork_event_id,
        generated_session_id,
    ) = asyncio.run(second_process(fork_event_id, generated_session_id))
    assert [event.id for event in replayed] == [fork_event_id]
    assert replayed[0].payload["fork_overlay_fingerprint"] == _overlay().fingerprint
    assert [event.session_id for event in regenerated] == [generated_session_id]
    assert conflict is not None and "conflicts" in str(conflict)
    assert sorted(child.id for child in children) == sorted(["compass-child", generated_session_id])
    assert controller_server.calls == [("lookup", {"q": "accruals"})]
    assert INSTRUCTIONS in _text(provider.requests[0].messages[-1])

    async def changed_controller():
        session = ControllerMcpSession(_definitions("A changed controller tool."))
        toolset, _ = _controller_toolset(session)
        app, _journal = _host_app(
            SQLiteSessionStore(database),
            ScriptedProvider([_completed("never")]),
            toolset=toolset,
        )
        with pytest.raises(ExecutionProfileMismatchError):
            await _drain(
                app.fork_controller_access(SUBJECT).resume(
                    ResumeRequest(session_id="compass-child", messages=[Message.text("user", "go")])
                )
            )
        unenrolled, _journal = _host_app(
            SQLiteSessionStore(database),
            ScriptedProvider([_completed("never")]),
            enroll=False,
        )
        with pytest.raises(KeyError, match="Agent not registered"):
            await _drain(
                unenrolled.resume(
                    ResumeRequest(session_id="compass-child", messages=[Message.text("user", "go")])
                )
            )

    asyncio.run(changed_controller())


def test_tampered_overlay_record_fails_closed() -> None:
    async def scenario():
        toolset, controller_server = _controller_toolset()
        store = InMemorySessionStore()
        provider = ScriptedProvider([_completed("closed"), _completed("never")])
        app, _journal = _host_app(store, provider, toolset=toolset)
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request()))
        child = await store.load("compass-child")
        assert child is not None
        metadata = json.loads(json.dumps(child.metadata))
        metadata[FORK_OVERLAY_METADATA_KEY]["overlay"]["instructions"] = "Post every journal."
        store._sessions["compass-child"] = child.model_copy(update={"metadata": metadata})
        with pytest.raises(ValueError, match="malformed"):
            await _drain(
                controller.resume(
                    ResumeRequest(session_id="compass-child", messages=[Message.text("user", "go")])
                )
            )
        return provider, controller_server

    provider, controller_server = asyncio.run(scenario())
    assert len(provider.requests) == 1
    assert controller_server.calls == []


def test_http_controller_entrance_authenticates_before_any_overlay() -> None:
    import httpx
    from fastapi import FastAPI

    from cayu.server.fork_controller import create_fork_controller_router

    async def scenario():
        toolset, _server = _controller_toolset()
        store = InMemorySessionStore()
        app, _journal = _host_app(store, ScriptedProvider([_completed("closed")]), toolset=toolset)
        await _run_trunk(app)

        async def authenticate(request):
            # Host-owned authentication; body fields never name the subject.
            token = request.headers.get("authorization")
            return {"Bearer compass": SUBJECT, "Bearer other": "svc:other"}.get(token, "")

        server = FastAPI()
        server.include_router(create_fork_controller_router(app, authenticate=authenticate))
        body = _fork_request().model_dump(mode="json")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            anonymous = await client.post("/fork-controller/fork", json=body)
            unenrolled = await client.post(
                "/fork-controller/fork", json=body, headers={"authorization": "Bearer other"}
            )
            mismatched = await client.post(
                "/fork-controller/fork",
                json={**body, "overlay": {**body["overlay"], "controller": "other"}},
                headers={"authorization": "Bearer compass"},
            )
            forked = await client.post(
                "/fork-controller/fork", json=body, headers={"authorization": "Bearer compass"}
            )
        return store, anonymous, unenrolled, mismatched, forked

    store, anonymous, unenrolled, mismatched, forked = asyncio.run(scenario())
    assert anonymous.status_code == 401
    assert unenrolled.status_code == 403
    assert mismatched.status_code == 403
    assert forked.status_code == 200
    events = [
        json.loads(line.removeprefix("data: "))
        for line in forked.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [event["type"] for event in events] == ["session.forked"]
    assert events[0]["payload"]["fork_overlay_fingerprint"] == _overlay().fingerprint
    child = asyncio.run(store.load("compass-child"))
    assert child is not None and child.agent_name == BODY


def test_controller_fork_dispatches_its_first_invocation_through_ordinary_dispatch() -> None:
    from cayu.sessions.forks import session_fork_profile_relationship
    from cayu.tasks import DispatchRequest, DispatchStatus, InMemoryTaskStore, TaskStoreDispatcher

    async def scenario():
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider(
            [_completed("closed"), _calls(LOOKUP, {"q": "accruals"}), _completed("finding")]
        )
        store = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        app, _journal = _host_app(
            store, provider, toolset=toolset, task_store=tasks, dispatcher=dispatcher
        )
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        source = await controller.snapshot_source("trunk")
        # Budget, stop condition and result shape are the first invocation's own fields.
        invocation = ResumeRequest(
            session_id="compass-child",
            messages=[Message.text("user", "Investigate accruals.")],
            max_steps=4,
        )
        fork_events = await _drain(
            controller.fork(
                _fork_request(
                    expected_source=source,
                    initial_invocation=invocation,
                    initial_dispatch_id="compass-dispatch-1",
                )
            )
        )
        submitted = await app.dispatch(
            DispatchRequest(
                session_id="compass-child",
                dispatch_id="compass-dispatch-1",
                messages=invocation.messages,
                max_steps=invocation.max_steps,
            )
        )
        completed = await dispatcher.process_next(app, worker_id="host-worker")
        return store, provider, controller_server, fork_events, submitted, completed

    store, provider, controller_server, fork_events, submitted, completed = asyncio.run(scenario())
    assert fork_events[0].payload["initial_dispatch_id"] == "compass-dispatch-1"
    assert fork_events[0].payload["fork_overlay_fingerprint"] == _overlay().fingerprint
    assert submitted.status is DispatchStatus.SUBMITTED
    assert completed is not None and completed.status is DispatchStatus.COMPLETED
    child = asyncio.run(store.load("compass-child"))
    assert child is not None and child.status is SessionStatus.COMPLETED
    relationship = session_fork_profile_relationship(child)
    assert relationship is not None and relationship.initial_invocation_profile is not None
    assert controller_server.calls == [("lookup", {"q": "accruals"})]
    assert INSTRUCTIONS in _text(provider.requests[-1].messages[-1])


async def _forbidden(stream) -> BaseException | None:
    try:
        await _drain(stream)
    except (ForkControllerDenied, ExecutionProfileAdoptionRejected, ValueError) as exc:
        return exc
    return None


def test_controller_cannot_fork_a_source_outside_its_enrollment() -> None:
    async def scenario():
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider([_completed("closed"), _completed("payroll ran")])
        store = InMemorySessionStore()
        app, _journal = _host_app(store, provider, toolset=toolset, extra_agents=("payroll",))
        await _drain(
            app.run(
                RunRequest(
                    session_id="payroll-trunk",
                    agent_name="payroll",
                    messages=[Message.text("user", "Run payroll.")],
                )
            )
        )
        controller = app.fork_controller_access(SUBJECT)
        retargeted = await _forbidden(
            controller.fork(_fork_request(source_session_id="payroll-trunk", agent_name=HOST_AGENT))
        )
        plain = await _forbidden(controller.fork(_fork_request(source_session_id="payroll-trunk")))
        return store, provider, controller_server, retargeted, plain

    store, provider, controller_server, retargeted, plain = asyncio.run(scenario())
    assert isinstance(retargeted, ForkControllerDenied)
    assert isinstance(plain, ForkControllerDenied)
    assert asyncio.run(store.load("compass-child")) is None
    assert len(provider.requests) == 1
    assert controller_server.calls == []


def test_resume_rechecks_enrollment_after_the_host_removes_an_agent(tmp_path: Path) -> None:
    database = tmp_path / "host.sqlite"

    async def first_process() -> None:
        toolset, _server = _controller_toolset()
        app, _journal = _host_app(
            SQLiteSessionStore(database), ScriptedProvider([_completed("closed")]), toolset=toolset
        )
        await _run_trunk(app)
        # Without tools the child keeps the host agent, which stays registered.
        await _drain(
            app.fork_controller_access(SUBJECT).fork(
                ForkSessionRequest(
                    source_session_id="trunk",
                    session_id="compass-child",
                    overlay=_overlay(tools=()),
                )
            )
        )

    async def second_process():
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider([_completed("never")])
        app, _journal = _host_app(
            SQLiteSessionStore(database),
            provider,
            toolset=toolset,
            extra_agents=("payroll",),
            enrolled_agents=("payroll",),
        )
        denied = await _forbidden(
            app.fork_controller_access(SUBJECT).resume(
                ResumeRequest(session_id="compass-child", messages=[Message.text("user", "go")])
            )
        )
        return provider, controller_server, denied

    asyncio.run(first_process())
    provider, controller_server, denied = asyncio.run(second_process())
    assert isinstance(denied, ForkControllerDenied)
    assert provider.requests == []
    assert controller_server.calls == []


def test_overlay_adoption_refuses_host_registration_drift(tmp_path: Path) -> None:
    database = tmp_path / "host.sqlite"

    async def first_process() -> None:
        toolset, _server = _controller_toolset()
        app, _journal = _host_app(
            SQLiteSessionStore(database), ScriptedProvider([_completed("closed")]), toolset=toolset
        )
        await _run_trunk(app)

    async def second_process(extra_tools: tuple[Tool, ...], session_id: str):
        toolset, controller_server = _controller_toolset()
        store = SQLiteSessionStore(database)
        app, _journal = _host_app(
            store, ScriptedProvider([]), toolset=toolset, extra_tools=extra_tools
        )
        failure = await _forbidden(
            app.fork_controller_access(SUBJECT).fork(_fork_request(session_id))
        )
        return failure, await store.load(session_id), controller_server

    asyncio.run(first_process())
    drifted, drifted_child, server = asyncio.run(second_process((VoidJournal(),), "drifted"))
    unchanged, unchanged_child, _ = asyncio.run(second_process((), "unchanged"))
    # A host tool added after the source was created is not the controller's grant.
    assert isinstance(drifted, ExecutionProfileAdoptionRejected)
    assert drifted_child is None
    assert server.calls == []
    assert unchanged is None
    assert unchanged_child is not None and unchanged_child.agent_name == BODY


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_overlay_children_can_be_reforked_with_same_or_overlapping_grants(
    backend: str, tmp_path: Path
) -> None:
    second_tool = "mcp__compass__search"

    async def scenario():
        session = ControllerMcpSession(_definitions(names=("lookup", "search")))
        toolset, controller_server = _controller_toolset(session)
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "host.sqlite")
        )
        app, _journal = _host_app(store, ScriptedProvider([_completed("closed")]), toolset=toolset)
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request()))
        same = await _drain(
            controller.fork(_fork_request("same", source_session_id="compass-child"))
        )
        both = _overlay(
            tools=(
                ForkOverlayToolGrant(tool_name=LOOKUP),
                ForkOverlayToolGrant(tool_name=second_tool),
            )
        )
        overlapping = await _drain(
            controller.fork(
                _fork_request("overlapping", source_session_id="compass-child", overlay=both)
            )
        )
        host_tool = await _forbidden(
            controller.fork(
                _fork_request(
                    "host-tool",
                    source_session_id="compass-child",
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("post_journal",)),
                )
            )
        )
        return store, controller_server, same, overlapping, host_tool

    store, controller_server, same, overlapping, host_tool = asyncio.run(scenario())
    assert [event.type for event in same] == [EventType.SESSION_FORKED]
    assert [event.type for event in overlapping] == [EventType.SESSION_FORKED]
    overlapping_child = asyncio.run(store.load("overlapping"))
    assert overlapping_child is not None
    assert set(overlapping_child.metadata["cayu:tool_capability_ceiling"]["tool_names"]) == {
        LOOKUP,
        second_tool,
    }
    # The source child never had the host tool, so a re-fork cannot add it.
    assert host_tool is not None
    assert asyncio.run(store.load("host-tool")) is None
    assert controller_server.calls == []


def test_overlay_forks_cannot_publish_host_knowledge() -> None:
    from cayu.tools.knowledge import RememberKnowledgePolicy, RememberKnowledgeTool

    async def scenario():
        toolset, _server = _controller_toolset()
        provider = ScriptedProvider(
            [
                # The host's own session may publish; this proves the tool works.
                _calls("remember_knowledge", {"text": "September uses accrual basis."}),
                _completed("closed"),
                _calls("remember_knowledge", {"text": MEMORY.content}, call_id="call-2"),
                _completed("noted"),
            ]
        )
        store = InMemorySessionStore()
        knowledge = InMemoryKnowledgeStore()
        # Model-authored knowledge is stored for review, so the scope reads pending entries.
        scope = KnowledgeAccessScope.for_namespace(
            "finance", allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING]
        )
        app, _journal = _host_app(
            store,
            provider,
            toolset=toolset,
            extra_tools=(
                RememberKnowledgeTool(policy=RememberKnowledgePolicy(default_namespace="finance")),
            ),
            host_tools=("remember_knowledge",),
            environment=Environment(
                EnvironmentSpec(name="finance"),
                knowledge_store=knowledge,
                knowledge_access_scope=scope,
            ),
        )
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(
            controller.fork(
                _fork_request(
                    tool_capability_ceiling=ToolCapabilityCeiling(
                        tool_names=("remember_knowledge",)
                    )
                )
            )
        )
        await _drain(
            controller.resume(
                ResumeRequest(
                    session_id="compass-child", messages=[Message.text("user", "Remember it.")]
                )
            )
        )
        changes = await knowledge.read_changes(after_sequence=0, access_scope=scope)
        return store, changes

    store, changes = asyncio.run(scenario())
    # Only the host trunk's own entry was published.
    assert [change.kind for change in changes.changes] == ["created"]
    events = asyncio.run(store.load_events("compass-child"))
    results = [
        event.payload["result"]
        for event in events
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
    ]
    assert len(results) == 1
    assert results[0]["structured"]["outcome"] == "publication_owner_closed"


def test_http_controller_routes_stamp_the_authenticated_actor_and_map_errors() -> None:
    import httpx
    from fastapi import FastAPI

    from cayu.server.fork_controller import create_fork_controller_router

    async def scenario():
        toolset, controller_server = _controller_toolset()
        store = InMemorySessionStore()
        app, _journal = _host_app(
            store,
            ScriptedProvider([_completed("closed"), _completed("payroll ran")]),
            toolset=toolset,
            extra_agents=("payroll",),
        )
        await _run_trunk(app)
        await _drain(
            app.run(
                RunRequest(
                    session_id="payroll-trunk",
                    agent_name="payroll",
                    messages=[Message.text("user", "Run payroll.")],
                )
            )
        )

        async def authenticate(request):
            return SUBJECT if request.headers.get("authorization") == "Bearer compass" else ""

        server = FastAPI()
        server.include_router(create_fork_controller_router(app, authenticate=authenticate))
        headers = {"authorization": "Bearer compass"}
        body = _fork_request().model_dump(mode="json")
        claimed = {
            **body["profile_adoption"],
            "requested_by": {
                "subject": SUBJECT,
                "source": "system",
                "tenant": "acme",
                "claims": {"role": "admin"},
            },
        }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            post = client.post
            responses = {
                "forged_subject": await post(
                    "/fork-controller/fork",
                    json={
                        **body,
                        "profile_adoption": {
                            **claimed,
                            "requested_by": {"subject": "svc:other", "source": "request"},
                        },
                    },
                    headers=headers,
                ),
                "unenrolled_source": await post(
                    "/fork-controller/fork",
                    json={**body, "source_session_id": "payroll-trunk", "agent_name": HOST_AGENT},
                    headers=headers,
                ),
                "missing_source": await post(
                    "/fork-controller/fork",
                    json={**body, "source_session_id": "no-such-session"},
                    headers=headers,
                ),
                "forked": await post(
                    "/fork-controller/fork",
                    json={**body, "profile_adoption": claimed},
                    headers=headers,
                ),
                "conflict": await post(
                    "/fork-controller/fork",
                    json={
                        **body,
                        "profile_adoption": claimed,
                        "overlay": {**body["overlay"], "instructions": "Other."},
                    },
                    headers=headers,
                ),
                "missing_snapshot": await client.get(
                    "/fork-controller/sources/no-such-session/snapshot", headers=headers
                ),
            }
        return store, controller_server, responses

    store, controller_server, responses = asyncio.run(scenario())
    statuses = {name: response.status_code for name, response in responses.items()}
    assert statuses == {
        "forged_subject": 403,
        "unenrolled_source": 403,
        "missing_source": 403,
        "forked": 200,
        "conflict": 409,
        "missing_snapshot": 403,
    }
    for name in ("unenrolled_source", "missing_source", "missing_snapshot"):
        assert responses[name].json() == {"detail": "Fork controller access denied."}
    decided = [
        event
        for event in asyncio.run(store.load_events("compass-child"))
        if event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
    ]
    assert len(decided) == 1
    actor_text = json.dumps(decided[0].payload)
    assert '"http_auth"' in actor_text
    assert "acme" not in actor_text and "admin" not in actor_text
    assert controller_server.calls == []


def test_agent_names_cannot_claim_the_fork_controller_separator() -> None:
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    with pytest.raises(ValueError, match="reserved for fork-controller bodies"):
        app.register_agent(AgentSpec(name=BODY, model="fake-model"))


def test_overlay_context_layer_is_bounded() -> None:
    from cayu.sessions.fork_overlays import FORK_OVERLAY_CONTEXT_MAX_CHARS

    memory = tuple(
        ForkOverlayMemoryItem(id=f"item-{index}", content="x" * 8_000) for index in range(4)
    )
    with pytest.raises(ValueError, match=str(FORK_OVERLAY_CONTEXT_MAX_CHARS)):
        _overlay(instructions="y", memory=memory)


def test_scoped_subjects_cannot_resume_or_recover_an_overlay_fork() -> None:
    from tests.core.test_resource_execution_access import Policy

    from cayu._resource_access_errors import ResourceAccessDenied
    from cayu.sessions.recovery import IncompleteSessionRecoveryRequest

    async def scenario():
        toolset, controller_server = _controller_toolset()
        store = InMemorySessionStore()
        provider = ScriptedProvider([_completed("closed")])
        app, _journal = _host_app(store, provider, toolset=toolset, resource_access_policy=Policy())
        alice = await app.access("alice")
        await _drain(
            alice.run(
                RunRequest(
                    session_id="trunk",
                    agent_name=HOST_AGENT,
                    messages=[Message.text("user", "Close September.")],
                    labels={"organization": "acme"},
                )
            )
        )
        await _drain(app.fork_controller_access(SUBJECT).fork(_fork_request()))
        child = await store.load("compass-child")
        assert child is not None and child.invocation.resource_access is not None
        denied: list[str] = []
        try:
            await _drain(
                alice.resume(
                    ResumeRequest(session_id="compass-child", messages=[Message.text("user", "go")])
                )
            )
        except ResourceAccessDenied:
            denied.append("resume")
        try:
            await alice.recover(IncompleteSessionRecoveryRequest(session_id="compass-child"))
        except ResourceAccessDenied:
            denied.append("recover")
        return provider, controller_server, denied

    provider, controller_server, denied = asyncio.run(scenario())
    assert denied == ["resume", "recover"]
    assert len(provider.requests) == 1
    assert controller_server.calls == []


def test_http_controller_fork_maps_a_rejected_profile_to_conflict() -> None:
    import httpx
    from fastapi import FastAPI

    from cayu.server.fork_controller import create_fork_controller_router

    async def scenario():
        toolset, controller_server = _controller_toolset()
        store = InMemorySessionStore()
        app, _journal = _host_app(
            store,
            ScriptedProvider([_completed("closed")]),
            toolset=toolset,
            execution_profile_policy=RejectProfileChanges(),
        )
        await _run_trunk(app)

        async def authenticate(request):
            return SUBJECT

        server = FastAPI()
        server.include_router(create_fork_controller_router(app, authenticate=authenticate))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            response = await client.post(
                "/fork-controller/fork", json=_fork_request().model_dump(mode="json")
            )
        return store, controller_server, response

    store, controller_server, response = asyncio.run(scenario())
    assert response.status_code == 409
    assert response.json() == {"detail": "The fork's execution profile was not adopted."}
    assert asyncio.run(store.load("compass-child")) is None
    assert controller_server.calls == []


def test_controller_fork_cannot_probe_host_session_ids() -> None:
    async def scenario():
        toolset, _server = _controller_toolset()
        store = InMemorySessionStore()
        app, _journal = _host_app(store, ScriptedProvider([_completed("closed")]), toolset=toolset)
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        results = {
            name: await _forbidden(controller.fork(request))
            for name, request in {
                "existing_host_session": _fork_request("trunk"),
                "existing_host_session_bad_source": _fork_request(
                    "trunk", source_session_id="no-such-session"
                ),
            }.items()
        }
        await _drain(controller.fork(_fork_request()))
        conflict: BaseException | None = None
        try:
            await _drain(controller.fork(_fork_request(overlay=_overlay(instructions="Other."))))
        except RuntimeError as exc:
            conflict = exc
        return results, conflict

    results, conflict = asyncio.run(scenario())
    assert all(isinstance(error, ForkControllerDenied) for error in results.values())
    # The controller's own fork still reports a real conflict.
    assert type(conflict).__name__ == "ForkRequestConflict"


@pytest.mark.parametrize("session_id", ["compass-child", None])
@pytest.mark.parametrize("with_tools", [False, True])
def test_fork_replay_requires_current_enrollment(
    tmp_path: Path, session_id: str | None, with_tools: bool
) -> None:
    import httpx
    from fastapi import FastAPI

    from cayu.server.fork_controller import create_fork_controller_router

    database = tmp_path / "host.sqlite"
    overlay = _overlay() if with_tools else _overlay(tools=())

    def request() -> ForkSessionRequest:
        if with_tools:
            # The HTTP entrance records this provenance, so SDK and HTTP replay match.
            adoption = _fork_request().profile_adoption
            assert adoption is not None
            return _fork_request(
                session_id,
                overlay=overlay,
                profile_adoption=adoption.model_copy(
                    update={
                        "requested_by": ResolutionActor(
                            subject=SUBJECT, source=ResolutionActorSource.HTTP_AUTH
                        )
                    }
                ),
            )
        return ForkSessionRequest(source_session_id="trunk", session_id=session_id, overlay=overlay)

    async def first_process() -> str:
        toolset, _server = _controller_toolset()
        app, _journal = _host_app(
            SQLiteSessionStore(database), ScriptedProvider([_completed("closed")]), toolset=toolset
        )
        await _run_trunk(app)
        events = await _drain(app.fork_controller_access(SUBJECT).fork(request()))
        return events[0].id

    async def restarted(enrolled_agents: tuple[str, ...]):
        toolset, controller_server = _controller_toolset()
        store = SQLiteSessionStore(database)
        app, _journal = _host_app(
            store,
            ScriptedProvider([]),
            toolset=toolset,
            extra_agents=("payroll",),
            enrolled_agents=enrolled_agents,
        )
        sdk: list[Event] | BaseException
        try:
            sdk = await _drain(app.fork_controller_access(SUBJECT).fork(request()))
        except ForkControllerDenied as exc:
            sdk = exc
        server = FastAPI()

        async def authenticate(_request):
            return SUBJECT

        server.include_router(create_fork_controller_router(app, authenticate=authenticate))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            http = await client.post(
                "/fork-controller/fork", json=request().model_dump(mode="json")
            )
        return sdk, http, controller_server

    fork_event_id = asyncio.run(first_process())
    sdk, http, controller_server = asyncio.run(restarted(("payroll",)))
    assert isinstance(sdk, ForkControllerDenied)
    assert http.status_code == 403
    assert "data:" not in http.text
    assert controller_server.calls == []

    # With enrollment unchanged, replay still works after the source is deleted.
    asyncio.run(SQLiteSessionStore(database).delete_session("trunk"))
    sdk, http, _server = asyncio.run(restarted((HOST_AGENT,)))
    assert not isinstance(sdk, BaseException) and [event.id for event in sdk] == [fork_event_id]
    assert http.status_code == 200 and fork_event_id in http.text


def _controller_server(app: CayuApp, authenticate=None):
    from fastapi import FastAPI

    from cayu.server.fork_controller import create_fork_controller_router

    async def default_authenticate(_request):
        return SUBJECT

    server = FastAPI()
    server.include_router(
        create_fork_controller_router(app, authenticate=authenticate or default_authenticate)
    )
    return server


async def _http(server, method: str, path: str, **options: Any):
    import httpx

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server), base_url="http://test"
    ) as client:
        return await client.request(method, path, **options)


def test_an_omitted_ceiling_narrows_to_the_host_tool_allowance() -> None:
    async def scenario():
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider(
            [_completed("closed"), _calls("post_journal", {"amount": 100}), _completed("no")]
        )
        store = InMemorySessionStore()
        app, journal = _host_app(store, provider, toolset=toolset)
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        # Neither fork sends a ceiling, so both would inherit the trunk's host tools.
        await _drain(controller.fork(_fork_request(tool_capability_ceiling=None)))
        await _drain(
            controller.fork(
                ForkSessionRequest(
                    source_session_id="trunk",
                    session_id="memory-only",
                    overlay=_overlay(tools=()),
                )
            )
        )
        await _drain(
            controller.resume(
                ResumeRequest(session_id="compass-child", messages=[Message.text("user", "Post.")])
            )
        )
        return store, provider, journal, controller_server

    store, provider, journal, controller_server = asyncio.run(scenario())
    trunk = asyncio.run(store.load("trunk"))
    assert trunk is not None and "post_journal" in trunk.tool_capability_ceiling.tool_names
    child = asyncio.run(store.load("compass-child"))
    assert child is not None
    # Controller-served tools are unaffected; host tools are narrowed to the allowance.
    assert child.tool_capability_ceiling.tool_names == (LOOKUP,)
    memory_only = asyncio.run(store.load("memory-only"))
    assert memory_only is not None and memory_only.tool_capability_ceiling.tool_names == ()
    assert [tool["name"] for tool in provider.requests[1].tools] == [LOOKUP]
    assert journal.contexts == []
    assert controller_server.calls == []


def test_an_explicit_host_tool_outside_the_allowance_is_denied() -> None:
    async def scenario():
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider([_completed("closed")])
        store = InMemorySessionStore()
        app, journal = _host_app(store, provider, toolset=toolset)
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        post_journal = ToolCapabilityCeiling(tool_names=("post_journal",))
        sdk = await _forbidden(controller.fork(_fork_request(tool_capability_ceiling=post_journal)))
        source = await controller.snapshot_source("trunk")
        first_invocation = await _forbidden(
            controller.fork(
                _fork_request(
                    tool_capability_ceiling=None,
                    expected_source=source,
                    initial_invocation=ResumeRequest(
                        session_id="compass-child",
                        messages=[Message.text("user", "Post.")],
                        tool_capability_ceiling=post_journal,
                    ),
                    initial_dispatch_id="compass-dispatch-1",
                )
            )
        )
        # A fork without the tool cannot gain it on resume either.
        await _drain(
            controller.fork(_fork_request("compass-allowed", tool_capability_ceiling=None))
        )
        resumed = await _forbidden(
            controller.resume(
                ResumeRequest(
                    session_id="compass-allowed",
                    messages=[Message.text("user", "Post.")],
                    tool_capability_ceiling=post_journal,
                )
            )
        )
        http = await _http(
            _controller_server(app),
            "POST",
            "/fork-controller/fork",
            json=_fork_request(
                session_id="compass-http",
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("post_journal",)),
            ).model_dump(mode="json"),
        )
        return store, provider, journal, controller_server, (sdk, first_invocation, resumed), http

    store, provider, journal, controller_server, denials, http = asyncio.run(scenario())
    assert all(isinstance(denial, ForkControllerDenied) for denial in denials)
    assert http.status_code == 403
    assert http.json() == {"detail": "Fork controller access denied."}
    for session_id in ("compass-child", "compass-http"):
        assert asyncio.run(store.load(session_id)) is None
    assert len(provider.requests) == 1
    assert journal.contexts == []
    assert controller_server.calls == []


def test_an_allowed_host_tool_runs_in_a_controller_fork() -> None:
    async def scenario():
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider(
            [_completed("closed"), _calls("post_journal", {"amount": 100}), _completed("posted")]
        )
        store = InMemorySessionStore()
        app, journal = _host_app(store, provider, toolset=toolset, host_tools=("post_journal",))
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(
            controller.fork(
                _fork_request(
                    tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("post_journal",))
                )
            )
        )
        await _drain(
            controller.resume(
                ResumeRequest(session_id="compass-child", messages=[Message.text("user", "Post.")])
            )
        )
        return store, provider, journal, controller_server

    store, provider, journal, controller_server = asyncio.run(scenario())
    child = asyncio.run(store.load("compass-child"))
    assert child is not None
    assert child.tool_capability_ceiling.tool_names == ("post_journal", LOOKUP)
    assert sorted(tool["name"] for tool in provider.requests[1].tools) == sorted(
        ["post_journal", LOOKUP]
    )
    assert len(journal.contexts) == 1
    assert controller_server.calls == []


def test_narrowing_the_allowance_revokes_an_existing_fork_on_resume_and_replay(
    tmp_path: Path,
) -> None:
    database = tmp_path / "host.sqlite"
    adoption = _fork_request().profile_adoption
    assert adoption is not None
    # The HTTP entrance records this provenance, so SDK and HTTP replay match.
    request = _fork_request(
        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("post_journal",)),
        profile_adoption=adoption.model_copy(
            update={
                "requested_by": ResolutionActor(
                    subject=SUBJECT, source=ResolutionActorSource.HTTP_AUTH
                )
            }
        ),
    )
    resume = ResumeRequest(session_id="compass-child", messages=[Message.text("user", "Post.")])

    async def first_process() -> str:
        toolset, _server = _controller_toolset()
        app, _journal = _host_app(
            SQLiteSessionStore(database),
            ScriptedProvider([_completed("closed")]),
            toolset=toolset,
            host_tools=("post_journal",),
        )
        await _run_trunk(app)
        events = await _drain(app.fork_controller_access(SUBJECT).fork(request))
        return events[0].id

    async def restarted(host_tools: tuple[str, ...] | None):
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider([])
        app, journal = _host_app(
            SQLiteSessionStore(database), provider, toolset=toolset, host_tools=host_tools
        )
        controller = app.fork_controller_access(SUBJECT)
        replay: list[Event] | BaseException
        try:
            replay = await _drain(controller.fork(request))
        except ForkControllerDenied as exc:
            replay = exc
        resumed = await _forbidden(controller.resume(resume))
        server = _controller_server(app)
        http_replay = await _http(
            server, "POST", "/fork-controller/fork", json=request.model_dump(mode="json")
        )
        http_resume = await _http(
            server, "POST", "/fork-controller/resume", json=resume.model_dump(mode="json")
        )
        return replay, resumed, http_replay, http_resume, provider, journal, controller_server

    fork_event_id = asyncio.run(first_process())
    replay, resumed, http_replay, http_resume, provider, journal, server = asyncio.run(
        restarted(None)
    )
    assert isinstance(replay, ForkControllerDenied)
    assert isinstance(resumed, ForkControllerDenied)
    assert http_replay.status_code == 403 and "data:" not in http_replay.text
    assert http_resume.status_code == 403 and "data:" not in http_resume.text
    assert provider.requests == []
    assert journal.contexts == []
    assert server.calls == []
    child = asyncio.run(SQLiteSessionStore(database).load("compass-child"))
    # The durable ceiling is never rewritten; the allowance is re-checked against it.
    assert child is not None
    assert child.tool_capability_ceiling.tool_names == ("post_journal", LOOKUP)

    # With the allowance restored, the same fork replays exactly.
    replay, *_rest = asyncio.run(restarted(("post_journal",)))
    assert not isinstance(replay, BaseException)
    assert [event.id for event in replay] == [fork_event_id]


def test_host_tool_allowance_is_validated_at_enrollment() -> None:
    store = InMemorySessionStore()
    for host_tools, message in (
        (("post_jornal",), "no enrolled agent registers: post_jornal"),
        # A tool of an agent outside the enrollment is not an allowance this controller can use.
        (("void_journal",), "no enrolled agent registers: void_journal"),
    ):
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedProvider([]), default=True)
        app.register_agent(AgentSpec(name=HOST_AGENT, model="fake-model"), tools=[PostJournal()])
        app.register_agent(AgentSpec(name="payroll", model="fake-model"), tools=[VoidJournal()])
        with pytest.raises(ValueError, match=message):
            app.enroll_fork_controller(
                ForkController(
                    name=CONTROLLER,
                    subjects=(SUBJECT,),
                    agents=(HOST_AGENT,),
                    host_tools=host_tools,
                )
            )
        # A rejected enrollment leaves nothing enrolled.
        with pytest.raises(ForkControllerDenied):
            app.fork_controller_access(SUBJECT)
    with pytest.raises(ValueError, match="must not repeat"):
        ForkController(
            name=CONTROLLER,
            subjects=(SUBJECT,),
            agents=(HOST_AGENT,),
            host_tools=("post_journal", "post_journal"),
        )
    with pytest.raises(TypeError, match="iterable of strings"):
        ForkController(
            name=CONTROLLER,
            subjects=(SUBJECT,),
            agents=(HOST_AGENT,),
            host_tools="post_journal",  # ty: ignore[invalid-argument-type]
        )


def test_controller_mode_is_observe_only() -> None:
    from cayu.fork_controllers import ForkControllerMode

    def enroll(**options: Any) -> ForkController:
        return ForkController(name=CONTROLLER, subjects=(SUBJECT,), agents=(HOST_AGENT,), **options)

    assert enroll().mode is ForkControllerMode.OBSERVE
    assert enroll(mode="observe").mode is ForkControllerMode.OBSERVE
    assert [mode.value for mode in ForkControllerMode] == ["observe"]
    for future in ("assist", "guard", "OBSERVE"):
        with pytest.raises(ValueError, match="no other mode exists"):
            enroll(mode=future)
    with pytest.raises(TypeError, match="ForkControllerMode"):
        enroll(mode=1)


def test_observe_mode_controller_has_no_trunk_write_path() -> None:
    from cayu.fork_controllers import ForkControllerAccess

    async def scenario():
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider(
            [_completed("closed"), _calls(LOOKUP, {"q": "accruals"}), _completed("finding")]
        )
        store = InMemorySessionStore()
        app, _journal = _host_app(store, provider, toolset=toolset)
        await _run_trunk(app)
        before = (
            await store.load("trunk"),
            await store.load_transcript("trunk"),
            await store.load_events("trunk"),
        )
        controller = app.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request()))
        await _drain(
            controller.resume(
                ResumeRequest(session_id="compass-child", messages=[Message.text("user", "go")])
            )
        )
        to_trunk = {
            "resume_trunk": await _forbidden(
                controller.resume(
                    ResumeRequest(session_id="trunk", messages=[Message.text("user", "Steer.")])
                )
            ),
            "fork_into_trunk": await _forbidden(controller.fork(_fork_request("trunk"))),
        }
        server = _controller_server(app)
        http_resume = await _http(
            server,
            "POST",
            "/fork-controller/resume",
            json=ResumeRequest(
                session_id="trunk", messages=[Message.text("user", "Steer.")]
            ).model_dump(mode="json"),
        )
        after = (
            await store.load("trunk"),
            await store.load_transcript("trunk"),
            await store.load_events("trunk"),
        )
        routes = {
            (method, route.path)
            for route in server.routes
            if route.path.startswith("/fork-controller")
            for method in getattr(route, "methods", ())
        }
        return before, after, to_trunk, http_resume, routes, controller_server

    before, after, to_trunk, http_resume, routes, controller_server = asyncio.run(scenario())
    assert after == before
    assert all(isinstance(error, ForkControllerDenied) for error in to_trunk.values())
    assert http_resume.status_code == 403
    assert controller_server.calls == [("lookup", {"q": "accruals"})]
    # The whole controller surface: no message, steering or approval entrance
    # exists; everything beyond fork and resume only reads.
    assert {name for name in vars(ForkControllerAccess) if not name.startswith("_")} == {
        "controller_name",
        "subject",
        "enrollment",
        "fork",
        "resume",
        "snapshot_source",
        "sessions",
        "session",
        "events",
        "query_events",
        "supervision_events",
    }
    assert routes == {
        ("GET", "/fork-controller/enrollment"),
        ("GET", "/fork-controller/sources/{session_id}/snapshot"),
        ("POST", "/fork-controller/fork"),
        ("POST", "/fork-controller/resume"),
        ("GET", "/fork-controller/sessions"),
        ("GET", "/fork-controller/sessions/{session_id}"),
        ("GET", "/fork-controller/sessions/{session_id}/events"),
        ("POST", "/fork-controller/events/query"),
        ("POST", "/fork-controller/supervision-events/query"),
    }


def test_enrollment_describes_only_the_callers_own_enrollment() -> None:
    from cayu.fork_controllers import ForkControllerEnrollment, ForkControllerMode

    async def scenario():
        toolset, _server = _controller_toolset(
            ControllerMcpSession(_definitions(names=("lookup", "search")))
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedProvider([]), default=True)
        app.register_agent(
            AgentSpec(name=HOST_AGENT, model="fake-model"), tools=[PostJournal(), VoidJournal()]
        )
        app.register_agent(AgentSpec(name="payroll", model="fake-model"), tools=[PostJournal()])
        app.enroll_fork_controller(
            ForkController(
                name=CONTROLLER,
                subjects=(SUBJECT, "svc:compass-standby"),
                agents=(HOST_AGENT,),
                host_tools=("post_journal",),
                mcp_toolsets=(toolset,),
                allow_model_runs=True,
            )
        )
        app.enroll_fork_controller(
            ForkController(
                name="auditor",
                subjects=("svc:auditor",),
                agents=("payroll",),
                host_tools=("post_journal",),
            )
        )
        sdk = app.fork_controller_access(SUBJECT).enrollment()
        subject = SUBJECT

        async def authenticate(_request):
            return subject

        server = _controller_server(app, authenticate)
        http = await _http(server, "GET", "/fork-controller/enrollment")
        subject = "svc:auditor"
        auditor = await _http(server, "GET", "/fork-controller/enrollment")
        subject = "svc:unknown"
        unknown = await _http(server, "GET", "/fork-controller/enrollment")
        subject = ""
        anonymous = await _http(server, "GET", "/fork-controller/enrollment")
        return sdk, http, auditor, unknown, anonymous

    sdk, http, auditor, unknown, anonymous = asyncio.run(scenario())
    assert sdk == ForkControllerEnrollment(
        controller=CONTROLLER,
        subject=SUBJECT,
        mode=ForkControllerMode.OBSERVE,
        agents=(HOST_AGENT,),
        host_tools=("post_journal",),
        served_tools=(LOOKUP, "mcp__compass__search"),
        allow_model_runs=True,
    )
    with pytest.raises(ValueError):
        sdk.__setattr__("subject", "svc:other")
    assert http.status_code == 200
    assert http.json() == sdk.model_dump(mode="json")
    for hidden in ("svc:compass-standby", "auditor", "payroll", "void_journal"):
        assert hidden not in http.text
    assert auditor.json() == {
        "controller": "auditor",
        "subject": "svc:auditor",
        "mode": "observe",
        "agents": ["payroll"],
        "host_tools": ["post_journal"],
        "served_tools": [],
        # The default enrollment creates forks but never runs them.
        "allow_model_runs": False,
    }
    assert unknown.status_code == 403
    assert anonymous.status_code == 401


def test_bearer_authenticator_maps_tokens_to_subjects_in_constant_time(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hmac
    import logging

    from cayu.server.auth import AuthConfigurationError
    from cayu.server.fork_controller import ForkControllerBearerAuth

    primary = "cmp-primary-" + "a" * 32
    rotated = "cmp-rotated-" + "b" * 32
    auditor_token = "aud-" + "c" * 40
    tokens = {primary: SUBJECT, rotated: SUBJECT, auditor_token: "svc:auditor"}
    auth = ForkControllerBearerAuth(tokens)
    env_auth = ForkControllerBearerAuth.from_environment(
        environ={"CAYU_FORK_CONTROLLER_TOKENS": json.dumps(tokens)}
    )
    token_file = tmp_path / "tokens.json"
    token_file.write_text(json.dumps(tokens), encoding="utf-8")
    file_auth = ForkControllerBearerAuth.from_file(token_file)

    compared: list[bytes] = []
    real_compare = hmac.compare_digest

    def counting_compare(left, right):
        compared.append(right)
        return real_compare(left, right)

    async def scenario():
        app, _journal = _host_app(
            InMemorySessionStore(), ScriptedProvider([]), extra_agents=("payroll",)
        )
        app.enroll_fork_controller(
            ForkController(name="auditor", subjects=("svc:auditor",), agents=("payroll",))
        )
        results: dict[str, list[Any]] = {}
        for name, authenticator in {"mapping": auth, "env": env_auth, "file": file_auth}.items():
            server = _controller_server(app, authenticator)
            responses = []
            for headers in (
                {"Authorization": f"Bearer {primary}"},
                {"Authorization": f"bearer {rotated}"},
                {"Authorization": f"Bearer {auditor_token}"},
                {"Authorization": f"Bearer {primary}x"},
                {},
                {"Authorization": "Bearer"},
                {"Authorization": f"Bearer  {primary}"},
                {"Authorization": f"Basic {primary}"},
                {"Authorization": f"Bearer {primary} extra"},
                [(b"Authorization", f"Bearer {primary}\u00e9".encode("latin-1"))],
            ):
                responses.append(
                    await _http(server, "GET", "/fork-controller/enrollment", headers=headers)
                )
            responses.append(
                await _http(
                    server,
                    "GET",
                    "/fork-controller/enrollment",
                    headers=[("Authorization", f"Bearer {primary}")] * 2,
                )
            )
            results[name] = responses
        return results

    monkeypatch.setattr(hmac, "compare_digest", counting_compare)
    with caplog.at_level(logging.DEBUG):
        results = asyncio.run(scenario())
    monkeypatch.undo()

    for responses in results.values():
        subjects = [
            response.json()["subject"] if response.status_code == 200 else response.status_code
            for response in responses
        ]
        assert subjects == [SUBJECT, SUBJECT, "svc:auditor", 401, 401, 401, 401, 401, 401, 401, 401]
        for response in responses:
            assert all(token not in response.text for token in tokens)
    # Every well-formed presentation is compared with every entry, even after a match.
    assert len(compared) == 3 * 4 * len(tokens)

    failures: list[BaseException] = []
    for build in (
        lambda: ForkControllerBearerAuth({}),
        lambda: ForkControllerBearerAuth(
            {f"{index:02d}-{primary}": SUBJECT for index in range(65)}
        ),
        lambda: ForkControllerBearerAuth({"short-" + primary[:8]: SUBJECT}),
        lambda: ForkControllerBearerAuth({primary + " space": SUBJECT}),
        lambda: ForkControllerBearerAuth({primary: " svc:compass"}),
        lambda: ForkControllerBearerAuth.from_environment(environ={}),
        lambda: ForkControllerBearerAuth.from_environment(
            environ={"CAYU_FORK_CONTROLLER_TOKENS": "{" + primary}
        ),
        lambda: ForkControllerBearerAuth.from_environment(
            environ={"CAYU_FORK_CONTROLLER_TOKENS": json.dumps([primary])}
        ),
        lambda: ForkControllerBearerAuth.from_file(tmp_path / "missing.json"),
    ):
        with pytest.raises(AuthConfigurationError) as raised:
            build()
        failures.append(raised.value)
    rendered = [repr(auth), str(auth), *(f"{error!s} {error!r}" for error in failures)]
    rendered.extend(record.getMessage() for record in caplog.records)
    output = capsys.readouterr()
    rendered.extend((output.out, output.err))
    for text in rendered:
        assert primary[:16] not in text and rotated not in text and auditor_token not in text
    assert repr(auth) == "ForkControllerBearerAuth(tokens=3)"
    # Only digests are kept, never the tokens themselves.
    assert all(token not in repr(auth._entries) for token in tokens)
    assert (
        len(ForkControllerBearerAuth({f"{i:02d}-{primary}": SUBJECT for i in range(64)})._entries)
        == 64
    )


def test_instructions_only_fork_asking_for_current_child_needs_a_policy() -> None:
    async def scenario():
        store = InMemorySessionStore()
        provider = ScriptedProvider([_completed("closed")])
        app, _journal = _host_app(store, provider)
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        with pytest.raises(ExecutionProfileAdoptionRejected):
            await _drain(
                controller.fork(
                    _fork_request(
                        overlay=_overlay(tools=()),
                        tool_capability_ceiling=None,
                    )
                )
            )
        return store, provider

    store, provider = asyncio.run(scenario())
    assert asyncio.run(store.load("compass-child")) is None
    assert len(provider.requests) == 1


def test_fork_controller_options_are_keyword_only() -> None:
    import inspect

    parameters = inspect.signature(ForkController).parameters
    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == ["name", "subjects", "agents"]
    toolset, _session = _controller_toolset()
    with pytest.raises(TypeError):
        ForkController(CONTROLLER, (SUBJECT,), (HOST_AGENT,), (toolset,))  # ty: ignore[too-many-positional-arguments]


@pytest.mark.parametrize("source", ["file_not_utf8", "file_not_json", "env_not_json"])
def test_bearer_token_configuration_errors_never_carry_the_tokens(
    tmp_path: Path, source: str
) -> None:
    from cayu.server.auth import AuthConfigurationError
    from cayu.server.fork_controller import ForkControllerBearerAuth

    token = "t" * 40
    token_file = tmp_path / "tokens.json"
    if source == "file_not_utf8":
        token_file.write_bytes(b'{"' + token.encode() + b'\xff": "svc:compass"}')
    else:
        token_file.write_text('{"' + token + '": "svc:compass",}', encoding="utf-8")
    with pytest.raises(AuthConfigurationError) as raised:
        if source == "env_not_json":
            ForkControllerBearerAuth.from_environment(
                environ={"CAYU_FORK_CONTROLLER_TOKENS": token_file.read_text(encoding="utf-8")}
            )
        else:
            ForkControllerBearerAuth.from_file(token_file)
    error = raised.value
    # Neither the error nor anything it chains may hold the tokens.
    assert error.__cause__ is None and error.__context__ is None
    rendered = [str(error), repr(error), repr(error.args), repr(vars(error))]
    assert not any(token in text for text in rendered)


def test_a_controller_cannot_fork_or_snapshot_another_controllers_fork() -> None:
    async def scenario():
        store = InMemorySessionStore()
        provider = ScriptedProvider([_completed("closed")])
        app, _journal = _host_app(store, provider)
        app.enroll_fork_controller(
            ForkController(name="auditor", subjects=("svc:auditor",), agents=(HOST_AGENT,))
        )
        await _run_trunk(app)
        compass = app.fork_controller_access(SUBJECT)
        auditor = app.fork_controller_access("svc:auditor")

        def instructions_only(session_id: str, source: str, controller: str) -> ForkSessionRequest:
            return ForkSessionRequest(
                source_session_id=source,
                session_id=session_id,
                overlay=_overlay(controller=controller, tools=()),
            )

        await _drain(compass.fork(instructions_only("compass-child", "trunk", CONTROLLER)))
        denied: dict[str, BaseException | None] = {
            "sdk_fork": await _forbidden(
                auditor.fork(instructions_only("auditor-copy", "compass-child", "auditor"))
            ),
            "host_fork": await _forbidden(
                app.fork_session(
                    ForkSessionRequest(source_session_id="compass-child", session_id="host-copy")
                )
            ),
        }
        try:
            await auditor.snapshot_source("compass-child")
            denied["sdk_snapshot"] = None
        except ForkControllerDenied as exc:
            denied["sdk_snapshot"] = exc

        async def authenticate_auditor(_request):
            return "svc:auditor"

        server = _controller_server(app, authenticate_auditor)
        http_fork = await _http(
            server,
            "POST",
            "/fork-controller/fork",
            json=instructions_only("auditor-http", "compass-child", "auditor").model_dump(
                mode="json"
            ),
        )
        http_snapshot = await _http(
            server, "GET", "/fork-controller/sources/compass-child/snapshot"
        )
        own_refork = await _drain(
            compass.fork(instructions_only("compass-again", "compass-child", CONTROLLER))
        )
        trunk_fork = await _drain(
            auditor.fork(instructions_only("auditor-trunk", "trunk", "auditor"))
        )
        return store, denied, http_fork, http_snapshot, own_refork, trunk_fork

    store, denied, http_fork, http_snapshot, own_refork, trunk_fork = asyncio.run(scenario())
    assert all(isinstance(error, ForkControllerDenied) for error in denied.values()), denied
    assert http_fork.status_code == 403 and "data:" not in http_fork.text
    assert http_snapshot.status_code == 403
    for session_id in ("auditor-copy", "host-copy", "auditor-http"):
        assert asyncio.run(store.load(session_id)) is None
    # A controller still re-forks its own forks and forks the host's sessions.
    assert [event.type for event in own_refork] == [EventType.SESSION_FORKED]
    assert [event.type for event in trunk_fork] == [EventType.SESSION_FORKED]


def test_a_stale_source_snapshot_is_a_conflict_over_http() -> None:
    async def scenario():
        toolset, controller_server = _controller_toolset()
        store = InMemorySessionStore()
        provider = ScriptedProvider([_completed("closed"), _completed("closed again")])
        app, _journal = _host_app(store, provider, toolset=toolset)
        await _run_trunk(app)
        server = _controller_server(app)
        snapshot = await _http(server, "GET", "/fork-controller/sources/trunk/snapshot")
        # The host continues the source after the controller captured it.
        await _drain(
            app.resume(
                ResumeRequest(session_id="trunk", messages=[Message.text("user", "One more.")])
            )
        )
        from cayu.sessions.forks import ForkSourceSnapshot

        request = _fork_request(
            expected_source=ForkSourceSnapshot.model_validate(snapshot.json()),
            initial_invocation=ResumeRequest(
                session_id="compass-child",
                messages=[Message.text("user", "Investigate accruals.")],
            ),
            initial_dispatch_id="compass-dispatch-1",
        ).model_dump(mode="json")
        forked = await _http(server, "POST", "/fork-controller/fork", json=request)
        return store, snapshot, forked, controller_server

    store, snapshot, forked, controller_server = asyncio.run(scenario())
    assert snapshot.status_code == 200
    assert forked.status_code == 409
    assert forked.json() == {"detail": "The session changed during the request; read it again."}
    assert asyncio.run(store.load("compass-child")) is None
    assert controller_server.calls == []


class HostedSearchProvider(ScriptedProvider):
    """A scripted provider that accepts OpenAI hosted web search."""

    def preflight_hosted_tools(self, *, model, hosted_tools, options) -> None:
        return None


@pytest.mark.parametrize("allowed", [False, True])
def test_hosted_tools_need_the_host_tool_allowance(tmp_path: Path, allowed: bool) -> None:
    from cayu.providers.hosted import OpenAIWebSearch

    database = tmp_path / "host.sqlite"
    allowance = ("openai_web_search",) if allowed else ()

    def instructions_only() -> ForkSessionRequest:
        return ForkSessionRequest(
            source_session_id="trunk", session_id="compass-child", overlay=_overlay(tools=())
        )

    async def process(host_tools: tuple[str, ...], batches):
        provider = HostedSearchProvider(batches)
        app, _journal = _host_app(
            SQLiteSessionStore(database),
            provider,
            hosted_tools=(OpenAIWebSearch(),),
            host_tools=host_tools,
        )
        return app, provider

    async def first() -> dict[str, Any]:
        app, provider = await process(allowance, [_completed("closed"), _completed("finding")])
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        outcome: dict[str, Any] = {
            "sdk_fork": await _forbidden(controller.fork(instructions_only()))
        }
        http = await _http(
            _controller_server(app),
            "POST",
            "/fork-controller/fork",
            json=instructions_only().model_dump(mode="json"),
        )
        outcome["http_status"] = http.status_code
        if allowed:
            await _drain(
                controller.resume(
                    ResumeRequest(session_id="compass-child", messages=[Message.text("user", "go")])
                )
            )
        outcome["requests"] = provider.requests
        outcome["child"] = await app.session_store.load("compass-child")
        return outcome

    outcome = asyncio.run(first())
    if not allowed:
        assert isinstance(outcome["sdk_fork"], ForkControllerDenied)
        assert outcome["http_status"] == 403
        assert outcome["child"] is None
        assert len(outcome["requests"]) == 1
        return
    assert outcome["sdk_fork"] is None and outcome["http_status"] == 200
    assert outcome["child"] is not None
    assert [tool.type for tool in outcome["requests"][-1].hosted_tools] == ["openai_web_search"]

    async def narrowed():
        # The host withdraws web search: the existing fork's replay and resume fail closed.
        app, provider = await process((), [])
        controller = app.fork_controller_access(SUBJECT)
        replay = await _forbidden(controller.fork(instructions_only()))
        resume = await _forbidden(
            controller.resume(
                ResumeRequest(session_id="compass-child", messages=[Message.text("user", "go")])
            )
        )
        return replay, resume, provider.requests

    replay, resume, requests = asyncio.run(narrowed())
    assert isinstance(replay, ForkControllerDenied)
    assert isinstance(resume, ForkControllerDenied)
    assert requests == []


def test_a_hosted_tool_no_enrolled_agent_has_fails_enrollment() -> None:
    with pytest.raises(ValueError, match="no enrolled agent registers"):
        _host_app(InMemorySessionStore(), ScriptedProvider([]), host_tools=("openai_web_search",))
