from __future__ import annotations

import asyncio

import pytest
from pydantic import SecretStr, TypeAdapter, ValidationError

from cayu._validation import copy_json_value, require_nonblank
from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    Message,
    MessageRole,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    WorkflowSpec,
)

from cayu.core.events import copy_event
from cayu.core.messages import copy_message, copy_message_part
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.environments import Environment, EnvironmentSpec
from cayu.mcp import McpServerSpec
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.runners import ExecCommand, ExecResult
from cayu.storage import KnowledgeHit, KnowledgeItem
from cayu.storage.memory import copy_knowledge_item
from cayu.vaults import ResolvedSecret, SecretRef, copy_secret_ref
from cayu.runtime import InMemoryEventSink, RunRequest, SessionStore


class EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content=args["text"], structured={"echoed": args["text"]})


def test_message_text_constructor():
    message = Message.text(MessageRole.USER, "hello")

    assert message.role == MessageRole.USER
    assert message.content == [TextPart(text="hello")]


def test_event_has_stable_contract_fields():
    event = Event(
        type=EventType.SESSION_STARTED,
        session_id="sess_1",
        agent_name="orchestrator",
        environment_name="local",
        payload={"ok": True},
    )

    assert event.type == EventType.SESSION_STARTED
    assert event.session_id == "sess_1"
    assert event.agent_name == "orchestrator"
    assert event.environment_name == "local"
    assert event.payload == {"ok": True}
    assert event.id
    assert event.timestamp


def test_event_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Event(type=EventType.SESSION_STARTED, session_id="sess_1", typo=True)  # type: ignore[call-arg]


def test_event_rejects_blank_identity_fields():
    with pytest.raises(ValidationError, match="cannot be blank"):
        Event(type=EventType.SESSION_STARTED, session_id=" ")

    with pytest.raises(ValidationError, match="cannot be blank"):
        Event(type=EventType.SESSION_STARTED, session_id="sess_1", id=" ")

    with pytest.raises(ValidationError, match="cannot be blank"):
        Event(
            type=EventType.SESSION_STARTED,
            session_id="sess_1",
            environment_name=" ",
        )


def test_event_requires_namespace_for_custom_types():
    event = Event(type="custom.pm.status.changed", session_id="sess_1")

    assert event.type == "custom.pm.status.changed"

    with pytest.raises(ValidationError, match="custom"):
        Event(type="model.starteddd", session_id="sess_1")

    malformed_custom_types = [
        "custom..bad",
        "custom. ",
        "custom.foo.",
        "custom.foo..bar",
        "custom.foo bar",
    ]
    for event_type in malformed_custom_types:
        with pytest.raises(ValidationError, match="dot-separated"):
            Event(type=event_type, session_id="sess_1")


def test_tool_result_supports_text_structured_and_artifacts():
    tool = EchoTool()
    result = asyncio.run(
        tool.run(
            ToolContext(session_id="sess_1", agent_name="agent"),
            {"text": "ok"},
        )
    )

    assert result.content == "ok"
    assert result.structured == {"echoed": "ok"}
    assert result.artifacts == []
    assert result.is_error is False


def test_tool_result_rejects_coerced_error_flags():
    with pytest.raises(ValidationError):
        ToolResult(content="failed", is_error="true")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        ToolResultPart(
            tool_call_id="call_1",
            tool_name="echo",
            is_error="true",  # type: ignore[arg-type]
        )


def test_core_value_objects_own_mutable_constructor_inputs():
    payload = {"nested": {"value": "original"}}
    event = Event(
        type=EventType.SESSION_STARTED,
        session_id="sess_1",
        environment_name="local",
        payload=payload,
    )

    agent_metadata = {"nested": {"value": "original"}}
    agent = AgentSpec(
        name="assistant",
        model="fake-model",
        metadata=agent_metadata,
    )

    workflow_metadata = {"nested": {"value": "original"}}
    workflow = WorkflowSpec(name="workflow", metadata=workflow_metadata)

    tool_result_structured = {"nested": {"value": "original"}}
    tool_result_artifacts = [{"nested": {"value": "original"}}]
    tool_result = ToolResult(
        content="ok",
        structured=tool_result_structured,
        artifacts=tool_result_artifacts,
    )

    payload["nested"]["value"] = "mutated"
    agent_metadata["nested"]["value"] = "mutated"
    workflow_metadata["nested"]["value"] = "mutated"
    tool_result_structured["nested"]["value"] = "mutated"
    tool_result_artifacts[0]["nested"]["value"] = "mutated"

    assert event.payload == {"nested": {"value": "original"}}
    assert event.environment_name == "local"
    assert agent.metadata == {"nested": {"value": "original"}}
    assert workflow.metadata == {"nested": {"value": "original"}}
    assert tool_result.structured == {"nested": {"value": "original"}}
    assert tool_result.artifacts == [{"nested": {"value": "original"}}]


def test_message_parts_own_mutable_constructor_inputs():
    arguments = {"nested": {"value": "original"}}
    call_part = ToolCallPart(
        tool_call_id="call_1",
        tool_name="echo",
        arguments=arguments,
    )
    call_message = Message.tool_call(calls=[call_part])

    structured = {"nested": {"value": "original"}}
    artifacts = [{"nested": {"value": "original"}}]
    result_part = ToolResultPart(
        tool_call_id="call_1",
        tool_name="echo",
        structured=structured,
        artifacts=artifacts,
    )
    result_message = Message.tool_result(results=[result_part])

    arguments["nested"]["value"] = "mutated"
    call_part.arguments["nested"]["value"] = "mutated again"
    structured["nested"]["value"] = "mutated"
    artifacts[0]["nested"]["value"] = "mutated"
    result_part.structured["nested"]["value"] = "mutated again"
    result_part.artifacts[0]["nested"]["value"] = "mutated again"

    assert call_message.content[0].arguments == {"nested": {"value": "original"}}
    assert result_message.content[0].structured == {"nested": {"value": "original"}}
    assert result_message.content[0].artifacts == [
        {"nested": {"value": "original"}}
    ]


def test_provider_runner_storage_and_secret_models_own_mutable_inputs():
    message = Message.text("user", "start")
    messages = [message]
    tools = [{"input_schema": {"nested": {"value": "original"}}}]
    options = {"nested": {"value": "original"}}
    request = ModelRequest(
        model="fake-model",
        messages=messages,
        tools=tools,
        options=options,
    )

    stream_arguments = {"nested": {"value": "original"}}
    stream_event = ModelStreamEvent.tool_call(
        name="echo",
        arguments=stream_arguments,
    )

    secret_metadata = {"nested": {"value": "original"}}
    secret = SecretRef(name="github_token", metadata=secret_metadata)

    knowledge_metadata = {"nested": {"value": "original"}}
    knowledge_item = KnowledgeItem(
        id="item_1",
        text="memory",
        metadata=knowledge_metadata,
    )
    knowledge_hit = KnowledgeHit(item=knowledge_item)

    exec_artifacts = [{"nested": {"value": "original"}}]
    exec_result = ExecResult(artifacts=exec_artifacts)

    messages[0].content[0].text = "mutated"
    tools[0]["input_schema"]["nested"]["value"] = "mutated"
    options["nested"]["value"] = "mutated"
    stream_arguments["nested"]["value"] = "mutated"
    secret_metadata["nested"]["value"] = "mutated"
    knowledge_metadata["nested"]["value"] = "mutated"
    knowledge_item.metadata["nested"]["value"] = "mutated again"
    exec_artifacts[0]["nested"]["value"] = "mutated"

    assert request.messages[0].content[0].text == "start"
    assert request.tools == [{"input_schema": {"nested": {"value": "original"}}}]
    assert request.options == {"nested": {"value": "original"}}
    assert stream_event.payload["arguments"] == {"nested": {"value": "original"}}
    assert secret.metadata == {"nested": {"value": "original"}}
    assert knowledge_hit.item.metadata == {"nested": {"value": "original"}}
    assert exec_result.artifacts == [{"nested": {"value": "original"}}]


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: Event(
            type=EventType.SESSION_STARTED,
            session_id="sess_1",
            payload={"bad": value},
        ),
        lambda value: ToolCallPart(
            tool_call_id="call_1",
            tool_name="echo",
            arguments={"bad": value},
        ),
        lambda value: ToolResult(
            content="ok",
            structured={"bad": value},
        ),
        lambda value: ToolResultPart(
            tool_call_id="call_1",
            tool_name="echo",
            structured={"bad": value},
        ),
        lambda value: ModelStreamEvent.tool_call(
            name="echo",
            arguments={"bad": value},
        ),
        lambda value: ModelRequest(
            model="fake-model",
            messages=[Message.text("user", "start")],
            tools=[{"bad": value}],
        ),
        lambda value: ExecResult(artifacts=[{"bad": value}]),
        lambda value: KnowledgeItem(id="item_1", text="memory", metadata={"bad": value}),
        lambda value: SecretRef(name="github_token", metadata={"bad": value}),
    ],
)
def test_json_boundary_fields_reject_non_json_values(factory):
    with pytest.raises(ValueError, match="JSON-compatible|finite JSON"):
        factory(object())

    with pytest.raises(ValueError, match="JSON-compatible|finite JSON"):
        factory(("tuple", "is", "not", "json"))

    with pytest.raises(ValueError, match="finite JSON"):
        factory(float("nan"))


def test_json_boundary_fields_validate_before_copying():
    class BadDeepcopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("deepcopy should not run")

    with pytest.raises(ValueError, match="JSON-compatible"):
        AgentSpec(
            name="assistant",
            model="fake-model",
            metadata={"bad": BadDeepcopy()},
        )


def test_json_boundary_fields_reject_circular_references():
    cyclic_dict = {}
    cyclic_dict["self"] = cyclic_dict

    cyclic_list = []
    cyclic_list.append(cyclic_list)

    with pytest.raises(ValueError, match="circular references"):
        copy_json_value(cyclic_dict, "payload")

    with pytest.raises(ValueError, match="circular references"):
        copy_json_value(cyclic_list, "payload")

    with pytest.raises(ValidationError, match="circular references"):
        Event(
            type=EventType.SESSION_STARTED,
            session_id="sess_1",
            payload=cyclic_dict,
        )

    with pytest.raises(ValidationError, match="circular references"):
        ToolSpec(name="bad", input_schema=cyclic_dict)


def test_json_boundary_fields_do_not_preserve_python_aliases():
    shared = {"value": "original"}
    payload = {"first": shared, "second": shared}

    copied = copy_json_value(payload, "payload")
    copied["first"]["value"] = "mutated"

    assert copied == {
        "first": {"value": "mutated"},
        "second": {"value": "original"},
    }
    assert copied["first"] is not copied["second"]

    event = Event(
        type=EventType.SESSION_STARTED,
        session_id="sess_1",
        payload=payload,
    )
    event.payload["first"]["value"] = "changed"

    assert event.payload["second"] == {"value": "original"}
    assert event.payload["first"] is not event.payload["second"]


def test_nonblank_helper_rejects_string_subclasses_before_strip():
    class BadString(str):
        def strip(self, chars=None):
            raise RuntimeError("strip should not run")

    with pytest.raises(ValueError, match="must be a string"):
        require_nonblank(BadString("assistant"), "name")

    agent = AgentSpec(name=BadString("assistant"), model="fake-model")
    event = Event(type=EventType.SESSION_STARTED, session_id=BadString("sess_1"))
    tool = ToolSpec(name=BadString("tool"))

    assert type(agent.name) is str
    assert type(event.session_id) is str
    assert type(tool.name) is str


def test_json_boundary_fields_reject_scalar_subclasses_before_copying():
    class BadString(str):
        def __deepcopy__(self, memo):
            raise RuntimeError("string deepcopy should not run")

    class BadInt(int):
        def __deepcopy__(self, memo):
            raise RuntimeError("integer deepcopy should not run")

    class BadFloat(float):
        def __deepcopy__(self, memo):
            raise RuntimeError("float deepcopy should not run")

    class BadKey(str):
        def __deepcopy__(self, memo):
            raise RuntimeError("key deepcopy should not run")

    for value in [BadString("value"), BadInt(1), BadFloat(1.0)]:
        with pytest.raises(ValueError, match="JSON-compatible"):
            AgentSpec(
                name="assistant",
                model="fake-model",
                metadata={"bad": value},
            )

    with pytest.raises(ValueError, match="keys must be strings"):
        AgentSpec(
            name="assistant",
            model="fake-model",
            metadata={BadKey("bad"): "value"},
        )


def test_json_boundary_fields_reject_custom_container_subclasses():
    class BadDict(dict):
        def items(self):
            raise RuntimeError("custom dict traversal should not run")

    class BadList(list):
        def __iter__(self):
            raise RuntimeError("custom list traversal should not run")

    with pytest.raises(ValueError, match="JSON-compatible"):
        AgentSpec(
            name="assistant",
            model="fake-model",
            metadata=BadDict({"bad": "value"}),
        )

    with pytest.raises(ValueError, match="JSON-compatible"):
        AgentSpec(
            name="assistant",
            model="fake-model",
            metadata={"bad": BadList(["value"])},
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: Message(role="user", content=[value]),
        lambda value: RunRequest(agent_name="assistant", messages=[value]),
        lambda value: ModelRequest(model="fake-model", messages=[value]),
        lambda value: KnowledgeHit(item=value),
        lambda value: McpServerSpec(
            name="local",
            command=["node"],
            secret_env={"TOKEN": value},
        ),
        lambda value: ExecCommand(argv=[value]),
    ],
)
def test_typed_boundary_fields_validate_before_copying(factory):
    class BadDeepcopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("deepcopy should not run")

    with pytest.raises(ValidationError):
        factory(BadDeepcopy())


def test_json_boundary_fields_accept_json_values_and_dump_to_json():
    payload = {
        "text": "ok",
        "integer": 1,
        "float": 1.25,
        "bool": True,
        "none": None,
        "list": [{"nested": "value"}],
    }
    values = [
        Event(
            type=EventType.SESSION_STARTED,
            session_id="sess_1",
            payload=payload,
        ),
        ToolCallPart(
            tool_call_id="call_1",
            tool_name="echo",
            arguments=payload,
        ),
        ToolResult(content="ok", structured=payload),
        ModelStreamEvent.tool_call(name="echo", arguments=payload),
    ]

    for value in values:
        assert TypeAdapter(type(value)).dump_json(value)


def test_model_stream_completed_rejects_invalid_explicit_payload():
    with pytest.raises(ValueError, match="dictionary|JSON-compatible"):
        ModelStreamEvent.completed([])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="dictionary|JSON-compatible"):
        ModelStreamEvent.completed(())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="dictionary|JSON-compatible"):
        ModelStreamEvent.completed(False)  # type: ignore[arg-type]


def test_model_stream_event_rejects_blank_type():
    with pytest.raises(ValueError, match="cannot be blank"):
        ModelStreamEvent(type="")

    with pytest.raises(ValueError, match="cannot be blank"):
        ModelStreamEvent(type=" ")


def test_model_stream_tool_call_rejects_invalid_helper_inputs():
    with pytest.raises(ValueError, match="cannot be blank"):
        ModelStreamEvent.tool_call(name=" ", arguments={})

    with pytest.raises(ValueError, match="cannot be blank"):
        ModelStreamEvent.tool_call(id=" ", name="echo", arguments={})

    with pytest.raises(ValueError, match="dictionary|JSON-compatible"):
        ModelStreamEvent.tool_call(name="echo", arguments=[])  # type: ignore[arg-type]


def test_model_stream_error_rejects_invalid_helper_inputs():
    with pytest.raises(ValueError, match="cannot be blank"):
        ModelStreamEvent.error(" ")

    with pytest.raises(ValueError, match="string"):
        ModelStreamEvent.error(123)  # type: ignore[arg-type]


def test_in_memory_event_sink_collects_events():
    sink = InMemoryEventSink()
    event = Event(type=EventType.SESSION_STARTED, session_id="sess_1")

    asyncio.run(sink.emit(event))

    assert sink.events == [event]


def test_in_memory_event_sink_owns_emitted_events():
    sink = InMemoryEventSink()
    event = Event(
        type=EventType.SESSION_STARTED,
        session_id="sess_1",
        payload={"nested": {"value": "original"}},
    )

    asyncio.run(sink.emit(event))
    event.payload["nested"]["value"] = "mutated"

    assert sink.events[0].payload == {"nested": {"value": "original"}}


def test_in_memory_event_sink_revalidates_constructed_events():
    class BadPayload(dict):
        def items(self):
            raise RuntimeError("sink event payload traversal should not run")

    sink = InMemoryEventSink()

    with pytest.raises(ValueError, match="`session_id` cannot be blank"):
        asyncio.run(
            sink.emit(
                Event.model_construct(
                    type=EventType.SESSION_STARTED,
                    session_id=" ",
                    payload={},
                )
            )
        )

    with pytest.raises(ValueError, match="JSON-compatible"):
        asyncio.run(
            sink.emit(
                Event.model_construct(
                    type=EventType.SESSION_STARTED,
                    session_id="sess_1",
                    payload=BadPayload({"ok": True}),
                )
            )
        )


def test_event_copy_boundaries_reject_event_subclasses_before_attribute_access():
    class BadEvent(Event):
        def __getattribute__(self, name):
            if name == "type":
                raise RuntimeError("event type access should not run")
            return super().__getattribute__(name)

    event = BadEvent.model_construct(
        type=EventType.SESSION_STARTED,
        session_id="sess_1",
        payload={},
    )

    with pytest.raises(TypeError, match="Event"):
        copy_event(event)

    with pytest.raises(TypeError, match="Event"):
        asyncio.run(InMemoryEventSink().emit(event))


def test_run_request_accepts_messages_and_metadata():
    request = RunRequest(
        agent_name="orchestrator",
        messages=[Message.text("user", "start")],
        metadata={"source": "test"},
    )

    assert request.agent_name == "orchestrator"
    assert request.messages[0].content[0].text == "start"
    assert request.metadata == {"source": "test"}


def test_agent_spec_uses_explicit_system_prompt_field():
    spec = AgentSpec(
        name="assistant",
        model="fake-model",
        system_prompt="You are careful.",
    )

    assert spec.system_prompt == "You are careful."

    with pytest.raises(ValidationError):
        AgentSpec(name="assistant", model="fake-model", prompt="too vague")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "model": "fake-model"},
        {"name": "   ", "model": "fake-model"},
        {"name": "assistant", "model": ""},
        {"name": "assistant", "model": "   "},
    ],
)
def test_agent_spec_rejects_blank_identity_fields(kwargs):
    with pytest.raises(ValidationError, match="cannot be blank"):
        AgentSpec(**kwargs)


def test_environment_spec_accepts_name_and_metadata():
    metadata = {"nested": {"value": "original"}}
    environment = Environment(EnvironmentSpec(name="local", metadata=metadata))

    metadata["nested"]["value"] = "mutated"

    assert environment.spec.name == "local"
    assert environment.spec.metadata == {"nested": {"value": "original"}}
    assert environment.workspace is None
    assert environment.runner is None
    assert environment.vault is None
    assert environment.mcp_servers == ()


def test_runtime_identity_models_reject_blank_fields():
    with pytest.raises(ValidationError, match="cannot be blank"):
        RunRequest(agent_name=" ", messages=[Message.text("user", "start")])

    with pytest.raises(ValidationError, match="cannot be blank"):
        RunRequest(
            agent_name="assistant",
            session_id=" ",
            messages=[Message.text("user", "start")],
        )

    with pytest.raises(ValidationError, match="cannot be blank"):
        RunRequest(
            agent_name="assistant",
            environment_name=" ",
            messages=[Message.text("user", "start")],
        )

    with pytest.raises(ValidationError, match="cannot be blank"):
        Event(type=EventType.SESSION_STARTED, session_id=" ")

    with pytest.raises(ValidationError, match="cannot be blank"):
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id="sess_1",
            environment_name=" ",
        )

    with pytest.raises(ValidationError, match="cannot be blank"):
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id="sess_1",
            tool_name=" ",
        )

    with pytest.raises(ValidationError, match="cannot be blank"):
        ToolContext(session_id=" ")


def test_framework_spec_models_reject_blank_names():
    with pytest.raises(ValidationError, match="cannot be blank"):
        ToolSpec(name=" ")

    with pytest.raises(ValidationError, match="cannot be blank"):
        WorkflowSpec(name=" ")

    with pytest.raises(ValidationError, match="cannot be blank"):
        EnvironmentSpec(name=" ")

    with pytest.raises(ValidationError, match="cannot be blank"):
        ModelRequest(model=" ", messages=[Message.text("user", "start")])

    with pytest.raises(ValidationError, match="cannot be blank"):
        McpServerSpec(name=" ", command=["node", "server.js"])

    with pytest.raises(ValidationError, match="cannot be blank"):
        SecretRef(name=" ")

    with pytest.raises(ValidationError, match="cannot be blank"):
        ResolvedSecret(name=" ", value=SecretStr("secret"))

    with pytest.raises(ValidationError, match="cannot be blank"):
        KnowledgeItem(id=" ", text="memory")

    with pytest.raises(ValidationError, match="cannot be blank"):
        KnowledgeItem(id="memory_1", text=" ")


def test_mcp_server_rejects_blank_transport_values():
    with pytest.raises(ValidationError, match="cannot be blank"):
        McpServerSpec(name="local", command=["node", " "])

    with pytest.raises(ValidationError, match="cannot be blank"):
        McpServerSpec(name="remote", url=" ")


def test_mcp_server_rejects_blank_config_keys():
    with pytest.raises(ValidationError, match="cannot be blank"):
        McpServerSpec(name="local", command=["node"], env={" ": "production"})

    with pytest.raises(ValidationError, match="cannot be blank"):
        McpServerSpec(
            name="local",
            command=["node"],
            secret_env={" ": SecretRef(name="token")},
        )

    with pytest.raises(ValidationError, match="cannot be blank"):
        McpServerSpec(name="remote", url="https://example.com", headers={" ": "cayu"})

    with pytest.raises(ValidationError, match="cannot be blank"):
        McpServerSpec(
            name="remote",
            url="https://example.com",
            secret_headers={" ": SecretRef(name="token")},
        )


def test_mcp_server_rejects_invalid_config_maps_with_validation_error():
    class BadDict(dict):
        def __iter__(self):
            raise RuntimeError("secret map iteration should not run")

    with pytest.raises(ValidationError, match="dictionary"):
        McpServerSpec(name="local", command=["node"], env=None)  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="dictionary"):
        McpServerSpec(
            name="local",
            command=["node"],
            secret_env=None,  # type: ignore[arg-type]
        )

    with pytest.raises(ValidationError, match="dictionary"):
        McpServerSpec(
            name="remote",
            url="https://example.com",
            headers=None,  # type: ignore[arg-type]
        )

    with pytest.raises(ValidationError, match="dictionary"):
        McpServerSpec(
            name="remote",
            url="https://example.com",
            secret_headers=None,  # type: ignore[arg-type]
        )

    with pytest.raises(ValidationError, match="dictionary"):
        McpServerSpec(
            name="local",
            command=["node"],
            secret_env=BadDict({"TOKEN": SecretRef(name="token")}),
        )


def test_knowledge_hit_rejects_non_finite_scores():
    item = KnowledgeItem(id="memory_1", text="memory")

    with pytest.raises(ValidationError, match="must be finite"):
        KnowledgeHit(item=item, score=float("nan"))

    with pytest.raises(ValidationError, match="must be finite"):
        KnowledgeHit(item=item, score=float("inf"))


def test_knowledge_hit_rejects_non_numeric_scores():
    item = KnowledgeItem(id="memory_1", text="memory")

    with pytest.raises(ValidationError, match="must be a number"):
        KnowledgeHit(item=item, score="0.3")  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="must be a number"):
        KnowledgeHit(item=item, score=True)  # type: ignore[arg-type]


def test_knowledge_hit_revalidates_constructed_items():
    class BadMetadata(dict):
        def items(self):
            raise RuntimeError("knowledge metadata traversal should not run")

    with pytest.raises(ValueError, match="`id` cannot be blank"):
        KnowledgeHit(
            item=KnowledgeItem.model_construct(
                id=" ",
                text="memory",
                source=None,
                metadata={},
            )
        )

    with pytest.raises(ValueError, match="JSON-compatible"):
        KnowledgeHit(
            item=KnowledgeItem.model_construct(
                id="memory_1",
                text="memory",
                source=None,
                metadata=BadMetadata({"bad": "value"}),
            )
        )


def test_knowledge_item_copy_boundaries_reject_subclasses_before_attribute_access():
    class BadItem(KnowledgeItem):
        def __getattribute__(self, name):
            if name == "id":
                raise RuntimeError("knowledge item id access should not run")
            return super().__getattribute__(name)

    item = BadItem.model_construct(
        id="memory_1",
        text="memory",
        source=None,
        metadata={},
    )

    with pytest.raises(TypeError, match="KnowledgeItem"):
        copy_knowledge_item(item)

    with pytest.raises(TypeError, match="KnowledgeItem"):
        KnowledgeHit(item=item)


def test_message_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Message(role="user", content=[], unexpected=True)  # type: ignore[call-arg]


def test_message_revalidates_constructed_parts():
    with pytest.raises(ValueError, match="`text` cannot be blank"):
        Message(
            role="user",
            content=[TextPart.model_construct(text=" ")],
        )

    with pytest.raises(ValueError, match="`tool_call_id` cannot be blank"):
        Message(
            role="assistant",
            content=[
                ToolCallPart.model_construct(
                    tool_call_id=" ",
                    tool_name="echo",
                    arguments={},
                )
            ],
        )

    with pytest.raises(ValueError, match="`tool_name` cannot be blank"):
        Message(
            role="tool",
            content=[
                ToolResultPart.model_construct(
                    tool_call_id="call_1",
                    tool_name=" ",
                    content="",
                    structured=None,
                    artifacts=[],
                    is_error=False,
                )
            ],
        )

    with pytest.raises(ValidationError):
        Message(
            role="tool",
            content=[
                ToolResultPart.model_construct(
                    tool_call_id="call_1",
                    tool_name="echo",
                    content="",
                    structured=None,
                    artifacts=[],
                    is_error="false",
                )
            ],
        )


def test_request_models_revalidate_constructed_messages():
    message = Message.model_construct(
        role="user",
        content=[TextPart.model_construct(text=" ")],
    )

    with pytest.raises(ValueError, match="`text` cannot be blank"):
        RunRequest(agent_name="assistant", messages=[message])

    with pytest.raises(ValueError, match="`text` cannot be blank"):
        ModelRequest(model="fake-model", messages=[message])


def test_message_copy_boundaries_reject_subclasses_before_attribute_access():
    class BadMessage(Message):
        def __getattribute__(self, name):
            if name == "content":
                raise RuntimeError("message content access should not run")
            return super().__getattribute__(name)

    class BadTextPart(TextPart):
        def __getattribute__(self, name):
            if name == "text":
                raise RuntimeError("text access should not run")
            return super().__getattribute__(name)

    message = BadMessage.model_construct(
        role="user",
        content=[TextPart(text="hi")],
    )
    part = BadTextPart.model_construct(text="hi")

    with pytest.raises(TypeError, match="Message"):
        copy_message(message)

    with pytest.raises(TypeError, match="message parts"):
        copy_message_part(part)

    with pytest.raises(TypeError, match="message parts"):
        Message(role="user", content=[part])


def test_text_part_rejects_blank_text():
    with pytest.raises(ValueError, match="`text` cannot be blank"):
        TextPart(text="")

    with pytest.raises(ValueError, match="`text` cannot be blank"):
        Message.text("user", "   ")


def test_message_supports_structured_tool_call_parts():
    message = Message.tool_call(
        tool_call_id="call_1",
        tool_name="echo",
        arguments={"text": "done"},
    )

    part = message.content[0]
    assert message.role == "assistant"
    assert part.type == "tool_call"
    assert part.tool_call_id == "call_1"
    assert part.tool_name == "echo"
    assert part.arguments == {"text": "done"}


def test_message_tool_call_rejects_invalid_explicit_arguments():
    with pytest.raises(ValueError, match="dictionary|JSON-compatible"):
        Message.tool_call(
            tool_call_id="call_1",
            tool_name="echo",
            arguments=[],  # type: ignore[arg-type]
        )


def test_tool_call_part_rejects_blank_ids_and_names():
    with pytest.raises(ValueError, match="`tool_call_id` cannot be blank"):
        ToolCallPart(tool_call_id="", tool_name="echo")

    with pytest.raises(ValueError, match="`tool_name` cannot be blank"):
        ToolCallPart(tool_call_id="call_1", tool_name=" ")

    with pytest.raises(ValueError, match="`tool_call_id` cannot be blank"):
        Message.tool_call(tool_call_id="", tool_name="echo")


def test_message_tool_call_rejects_non_string_ids_and_names_cleanly():
    with pytest.raises(ValueError, match="`tool_call_id` must be a string"):
        Message.tool_call(
            tool_call_id=1,  # type: ignore[arg-type]
            tool_name="echo",
        )

    with pytest.raises(ValueError, match="`tool_name` must be a string"):
        Message.tool_call(
            tool_call_id="call_1",
            tool_name=1,  # type: ignore[arg-type]
        )


def test_message_supports_grouped_structured_tool_call_parts():
    message = Message.tool_call(
        calls=[
            ToolCallPart(
                tool_call_id="call_1",
                tool_name="echo",
                arguments={"text": "one"},
            ),
            ToolCallPart(
                tool_call_id="call_2",
                tool_name="echo",
                arguments={"text": "two"},
            ),
        ],
    )

    assert message.role == "assistant"
    assert [part.tool_call_id for part in message.content] == ["call_1", "call_2"]


def test_message_supports_structured_tool_result_parts():
    message = Message.tool_result(
        tool_call_id="call_1",
        tool_name="echo",
        content="done",
        structured={"ok": True},
        is_error=False,
    )

    part = message.content[0]
    assert message.role == "tool"
    assert part.type == "tool_result"
    assert part.tool_call_id == "call_1"
    assert part.tool_name == "echo"
    assert part.content == "done"
    assert part.structured == {"ok": True}
    assert part.artifacts == []
    assert part.is_error is False


def test_message_tool_result_rejects_invalid_explicit_artifacts():
    with pytest.raises(ValueError, match="JSON-compatible"):
        Message.tool_result(
            tool_call_id="call_1",
            tool_name="echo",
            artifacts=(),  # type: ignore[arg-type]
        )


def test_tool_result_part_rejects_blank_ids_and_names():
    with pytest.raises(ValueError, match="`tool_call_id` cannot be blank"):
        ToolResultPart(tool_call_id="", tool_name="echo")

    with pytest.raises(ValueError, match="`tool_name` cannot be blank"):
        ToolResultPart(tool_call_id="call_1", tool_name=" ")

    with pytest.raises(ValueError, match="`tool_name` cannot be blank"):
        Message.tool_result(tool_call_id="call_1", tool_name="")


def test_message_tool_result_rejects_non_string_ids_and_names_cleanly():
    with pytest.raises(ValueError, match="`tool_call_id` must be a string"):
        Message.tool_result(
            tool_call_id=1,  # type: ignore[arg-type]
            tool_name="echo",
        )

    with pytest.raises(ValueError, match="`tool_name` must be a string"):
        Message.tool_result(
            tool_call_id="call_1",
            tool_name=1,  # type: ignore[arg-type]
        )


def test_message_supports_grouped_structured_tool_result_parts():
    message = Message.tool_result(
        results=[
            ToolResultPart(
                tool_call_id="call_1",
                tool_name="echo",
                content="one",
            ),
            ToolResultPart(
                tool_call_id="call_2",
                tool_name="echo",
                content="two",
            ),
        ],
    )

    assert message.role == "tool"
    assert [part.tool_call_id for part in message.content] == ["call_1", "call_2"]


def test_message_grouped_tool_result_rejects_invalid_falsey_scalar_fields():
    result = ToolResultPart(tool_call_id="call_1", tool_name="echo")

    for content in (0, False):
        with pytest.raises(ValueError, match="`content` must be a string"):
            Message.tool_result(
                results=[result],
                content=content,  # type: ignore[arg-type]
            )

    for is_error in (0, "", None):
        with pytest.raises(ValueError, match="`is_error` must be a bool"):
            Message.tool_result(
                results=[result],
                is_error=is_error,  # type: ignore[arg-type]
            )


def test_message_rejects_invalid_role_content_combinations():
    with pytest.raises(ValueError, match="content cannot be empty"):
        Message(role="assistant", content=[])

    with pytest.raises(ValueError, match="user messages only support"):
        Message(
            role="user",
            content=[
                ToolCallPart(
                    tool_call_id="call_1",
                    tool_name="echo",
                )
            ],
        )

    with pytest.raises(ValueError, match="assistant messages only support"):
        Message(
            role="assistant",
            content=[
                ToolResultPart(
                    tool_call_id="call_1",
                    tool_name="echo",
                )
            ],
        )

    with pytest.raises(ValueError, match="tool messages only support"):
        Message(role="tool", content=[TextPart(text="not a result")])


def test_tool_message_constructors_reject_ambiguous_grouped_inputs():
    with pytest.raises(ValueError, match="`calls` cannot be empty"):
        Message.tool_call(calls=[])

    with pytest.raises(ValueError, match="`calls` cannot be combined"):
        Message.tool_call(
            tool_call_id="call_1",
            tool_name="echo",
            calls=[
                ToolCallPart(
                    tool_call_id="call_2",
                    tool_name="echo",
                )
            ],
        )

    with pytest.raises(ValueError, match="`results` cannot be empty"):
        Message.tool_result(results=[])

    with pytest.raises(ValueError, match="`results` cannot be combined"):
        Message.tool_result(
            tool_call_id="call_1",
            tool_name="echo",
            results=[
                ToolResultPart(
                    tool_call_id="call_2",
                    tool_name="echo",
                    content="done",
                )
            ],
        )


def test_run_request_session_id_is_explicit_unique_session_id():
    request = RunRequest(
        agent_name="orchestrator",
        messages=[Message.text("user", "resume")],
        session_id="sess_existing",
    )

    assert request.session_id == "sess_existing"
    assert request.max_steps == 16


def test_run_request_rejects_coerced_max_steps():
    with pytest.raises(ValidationError):
        RunRequest(
            agent_name="orchestrator",
            messages=[Message.text("user", "resume")],
            max_steps="2",  # type: ignore[arg-type]
        )


def test_session_store_is_contract_only():
    with pytest.raises(TypeError):
        SessionStore()


def test_tool_requires_explicit_spec():
    class BrokenTool(Tool):
        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content="never")

    with pytest.raises(TypeError, match="must define `spec"):
        BrokenTool()


def test_tool_spec_is_copied_from_class_level_definition():
    first = EchoTool()
    second = EchoTool()

    assert first.spec == second.spec
    assert first.spec is not second.spec

    with pytest.raises(Exception):
        first.spec.name = "changed"  # type: ignore[misc]


def test_tool_schema_property_returns_copy():
    tool = EchoTool()
    schema = tool.schema

    schema["properties"]["text"]["type"] = "integer"

    assert tool.schema["properties"]["text"]["type"] == "string"
    assert tool.schema["required"] == ["text"]


def test_tool_spec_input_schema_returns_isolated_copy():
    tool = EchoTool()
    schema = tool.spec.input_schema

    schema["type"] = "array"
    schema["properties"]["text"]["type"] = "integer"
    schema["required"].append("other")
    schema |= {"extra": True}
    schema["properties"] |= {"other": {"type": "number"}}

    assert tool.spec.input_schema == {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    assert tool.spec.model_dump()["input_schema"]["required"] == ["text"]
    assert tool.spec.model_dump_json()


def test_tool_spec_copy_preserves_schema_contract():
    tool = EchoTool()
    copied = tool.spec.model_copy(deep=True)

    assert copied == tool.spec
    assert copied.input_schema == tool.spec.input_schema
    assert copied.input_schema is not tool.spec.input_schema


def test_tool_spec_copy_update_handles_input_schema():
    spec = ToolSpec(name="example", input_schema={"old": True})

    copied = spec.model_copy(update={"input_schema": {"new": ["value"]}})

    assert copied.input_schema == {"new": ["value"]}
    assert spec.input_schema == {"old": True}


def test_tool_spec_validation_errors_are_pydantic_errors():
    with pytest.raises(ValidationError):
        ToolSpec(name="bad", input_schema=[])  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        ToolSpec.model_validate({"name": "bad", "input_schema": []})

    with pytest.raises(ValidationError):
        ToolSpec(name="bad", unexpected=True)  # type: ignore[call-arg]


def test_tool_spec_rejects_non_json_input_schema_values():
    with pytest.raises(ValidationError, match="JSON-compatible"):
        ToolSpec(name="bad", input_schema={"bad": object()})

    with pytest.raises(ValidationError, match="JSON-compatible"):
        ToolSpec(name="bad", input_schema={"bad": ("tuple",)})

    with pytest.raises(ValidationError, match="finite JSON"):
        ToolSpec(name="bad", input_schema={"bad": float("nan")})


def test_tool_spec_internal_schema_storage_has_no_mutable_dict():
    spec = ToolSpec(name="example", input_schema={"nested": {"value": 1}})

    assert isinstance(spec._input_schema._items, tuple)

    with pytest.raises(TypeError):
        spec._input_schema._items[0] = ("changed", True)  # type: ignore[index]

    with pytest.raises(Exception):
        spec._input_schema._items += (("changed", True),)

    assert spec.input_schema == {"nested": {"value": 1}}


def test_tool_spec_json_schema_documents_input_schema():
    schema = ToolSpec.model_json_schema()

    assert schema["properties"]["input_schema"]["type"] == "object"
    assert "input_schema" not in schema.get("required", [])


def test_exec_command_separates_process_and_shell_execution():
    process = ExecCommand.process("python", "-m", "pytest")
    shell = ExecCommand.bash("echo ok")

    assert process.kind == "process"
    assert process.argv == ["python", "-m", "pytest"]
    assert shell.kind == "shell"
    assert shell.shell == "echo ok"

    with pytest.raises(ValidationError, match="non-empty argv"):
        ExecCommand(kind="process")

    with pytest.raises(ValidationError, match="argv entries"):
        ExecCommand.process("")

    with pytest.raises(ValidationError, match="argv entries"):
        ExecCommand.process("python", " ")

    with pytest.raises(ValidationError, match="cannot define shell"):
        ExecCommand(kind="process", argv=["echo"], shell="echo ok")

    with pytest.raises(ValidationError, match="non-empty script"):
        ExecCommand(kind="shell")

    with pytest.raises(ValidationError, match="non-empty script"):
        ExecCommand.bash("   ")

    with pytest.raises(ValidationError, match="cannot define argv"):
        ExecCommand(kind="shell", argv=["echo"], shell="echo ok")


def test_exec_command_rejects_constructed_string_subclasses_before_strip():
    class BadString(str):
        def strip(self, chars=None):
            raise RuntimeError("strip should not run")

    with pytest.raises(ValueError, match="argv entries"):
        ExecCommand.model_construct(
            kind="process",
            argv=[BadString("python")],
            shell=None,
        ).validate_shape()

    with pytest.raises(ValueError, match="non-empty script"):
        ExecCommand.model_construct(
            kind="shell",
            argv=None,
            shell=BadString("echo ok"),
        ).validate_shape()


def test_exec_result_rejects_coerced_status_fields():
    with pytest.raises(ValidationError):
        ExecResult(exit_code=True)

    with pytest.raises(ValidationError):
        ExecResult(timed_out="yes")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        ExecResult(cancelled=1)  # type: ignore[arg-type]


def test_secret_ref_does_not_store_raw_value():
    ref = SecretRef(name="github_token", handle="vault://github_token")

    assert ref.model_dump() == {
        "name": "github_token",
        "handle": "vault://github_token",
        "metadata": {},
    }

    with pytest.raises(ValueError):
        SecretRef(name="github_token", value="secret")  # type: ignore[call-arg]


def test_resolved_secret_masks_value_in_dumps():
    secret = ResolvedSecret(name="github_token", value=SecretStr("real-secret"))

    dumped = secret.model_dump()

    assert str(dumped["value"]) == "**********"
    assert secret.value.get_secret_value() == "real-secret"


def test_mcp_server_requires_one_transport():
    McpServerSpec(name="notion", command=["npx", "@notion/mcp"])
    McpServerSpec(name="linear", url="https://mcp.linear.example/sse")

    with pytest.raises(ValueError, match="exactly one"):
        McpServerSpec(name="bad")

    with pytest.raises(ValueError, match="exactly one"):
        McpServerSpec(name="bad", command=["npx"], url="https://example.com")


def test_mcp_server_splits_plain_config_from_secret_refs():
    spec = McpServerSpec(
        name="notion",
        command=["npx", "@notion/mcp"],
        env={"NODE_ENV": "production"},
        secret_env={"NOTION_TOKEN": SecretRef(name="notion_token", handle="vault://notion")},
        headers={"X-Client-Name": "cayu"},
        secret_headers={"Authorization": SecretRef(name="linear_token")},
    )

    dumped = spec.model_dump()

    assert dumped["env"]["NODE_ENV"] == "production"
    assert dumped["secret_env"]["NOTION_TOKEN"]["name"] == "notion_token"
    assert "value" not in dumped["secret_env"]["NOTION_TOKEN"]
    assert dumped["headers"]["X-Client-Name"] == "cayu"
    assert dumped["secret_headers"]["Authorization"]["name"] == "linear_token"


def test_mcp_server_revalidates_constructed_secret_refs():
    class BadMetadata(dict):
        def items(self):
            raise RuntimeError("secret metadata traversal should not run")

    with pytest.raises(ValidationError, match="`name` cannot be blank"):
        McpServerSpec(
            name="notion",
            command=["npx", "@notion/mcp"],
            secret_env={
                "NOTION_TOKEN": SecretRef.model_construct(
                    name=" ",
                    handle=None,
                    metadata={},
                )
            },
        )

    with pytest.raises(ValidationError, match="JSON-compatible"):
        McpServerSpec(
            name="linear",
            url="https://mcp.linear.example/sse",
            secret_headers={
                "Authorization": SecretRef.model_construct(
                    name="linear_token",
                    handle=None,
                    metadata=BadMetadata({"bad": "value"}),
                )
            },
        )


def test_secret_ref_copy_boundaries_reject_subclasses_before_attribute_access():
    class BadRef(SecretRef):
        def __getattribute__(self, name):
            if name == "name":
                raise RuntimeError("secret name access should not run")
            return super().__getattribute__(name)

    ref = BadRef.model_construct(
        name="token",
        handle=None,
        metadata={},
    )

    with pytest.raises(TypeError, match="SecretRef"):
        copy_secret_ref(ref)

    with pytest.raises(TypeError, match="SecretRef"):
        McpServerSpec(
            name="notion",
            command=["npx", "@notion/mcp"],
            secret_env={"NOTION_TOKEN": ref},
        )


def test_mcp_server_rejects_mixed_plain_and_secret_config():
    with pytest.raises(ValidationError):
        McpServerSpec(
            name="notion",
            command=["npx", "@notion/mcp"],
            env={"NOTION_TOKEN": SecretRef(name="notion_token", handle="vault://notion")},  # type: ignore[dict-item]
        )

    with pytest.raises(ValidationError):
        McpServerSpec(
            name="linear",
            url="https://mcp.linear.example/sse",
            headers={"Authorization": SecretRef(name="linear_token")},  # type: ignore[dict-item]
        )

    with pytest.raises(ValidationError):
        McpServerSpec(
            name="github",
            command=["npx", "@github/mcp"],
            secret_env={"GITHUB_API_KEY": "raw-key"},  # type: ignore[dict-item]
        )

    with pytest.raises(ValidationError):
        McpServerSpec(
            name="local",
            command=["node", "server.js"],
            secret_headers={"Authorization": "Bearer raw-token"},  # type: ignore[dict-item]
        )


def test_mcp_server_allows_non_sensitive_raw_config():
    spec = McpServerSpec(
        name="local",
        command=["node", "server.js"],
        env={"NODE_ENV": "production", "KEYBOARD_LAYOUT": "us"},
        headers={"X-Client-Name": "cayu"},
    )

    assert spec.env["NODE_ENV"] == "production"
    assert spec.env["KEYBOARD_LAYOUT"] == "us"
