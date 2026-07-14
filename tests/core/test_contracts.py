from __future__ import annotations

import asyncio
import errno
import pickle
import sys
import time
from types import SimpleNamespace

import pytest
from pydantic import SecretStr, TypeAdapter, ValidationError

import cayu.artifacts.local as artifact_local_module
import cayu.runners._subprocess as runner_subprocess_module
from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.artifacts import (
    ArtifactListResult,
    ArtifactMetadata,
    ArtifactReadResult,
    ArtifactScope,
    ArtifactStoreUnavailableError,
    FileAttachment,
    FileAttachmentKind,
    InvalidArtifactIdError,
    LocalArtifactStore,
    ResolvedFileAttachment,
    copy_artifact_read_result,
    file_attachment,
    file_attachment_from_payload,
)
from cayu.core import (
    EVENT_ID_MAX_CHARS,
    AgentSpec,
    Event,
    EventType,
    FilePart,
    Message,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    WorkflowSpec,
)
from cayu.core.events import copy_event
from cayu.core.messages import copy_message, copy_message_part
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.environments import Environment, EnvironmentSpec
from cayu.mcp import McpServerSpec
from cayu.providers import (
    ModelContextOverflowError,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.runners import ExecCommand, ExecResult, LocalRunner
from cayu.runtime import (
    DispatchHandle,
    DispatchRequest,
    DispatchStatus,
    InMemoryEventSink,
    LoopPolicy,
    ResumeRequest,
    RetryPolicy,
    RetryReason,
    RunRequest,
    Session,
    SessionStatus,
    SessionStore,
    StaticToolPolicy,
    StructuredOutputSpec,
    StructuredOutputStrategy,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
    retry_decision,
)
from cayu.storage import KnowledgeEntry, KnowledgeHit
from cayu.storage.memory import copy_knowledge_entry
from cayu.vaults import (
    REDACTED_SECRET,
    ResolvedSecret,
    SecretEnv,
    SecretRef,
    StaticVault,
    copy_secret_ref,
)
from cayu.workspaces import LocalWorkspace, WorkspaceListResult, WorkspaceReadResult


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
    assert message.content == (TextPart(text="hello"),)


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


def test_event_rejects_ids_beyond_the_public_replay_limit():
    Event(
        type=EventType.SESSION_STARTED,
        session_id="sess_1",
        id="e" * EVENT_ID_MAX_CHARS,
    )

    with pytest.raises(ValidationError, match="at most 512 characters"):
        Event(
            type=EventType.SESSION_STARTED,
            session_id="sess_1",
            id="e" * (EVENT_ID_MAX_CHARS + 1),
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


def test_dispatch_request_validates_boundary_data():
    structured_output = StructuredOutputSpec(
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
    )
    request = DispatchRequest(
        session_id="sess_dispatch",
        messages=[Message.text("user", "continue")],
        dispatch_id="dispatch_1",
        metadata={"source": "test"},
        structured_output=structured_output,
    )
    structured_output.json_schema["required"].append("mutated")

    assert request.session_id == "sess_dispatch"
    assert request.dispatch_id == "dispatch_1"
    assert request.metadata == {"source": "test"}
    assert request.structured_output is not None
    assert request.structured_output.json_schema["required"] == ["answer"]

    with pytest.raises(ValidationError, match="cannot be blank"):
        DispatchRequest(
            session_id=" ",
            messages=[Message.text("user", "continue")],
        )

    with pytest.raises(ValidationError, match="cannot be empty"):
        DispatchRequest(
            session_id="sess_dispatch",
            messages=[],
        )

    with pytest.raises(ValidationError, match="JSON-compatible"):
        DispatchRequest(
            session_id="sess_dispatch",
            messages=[Message.text("user", "continue")],
            metadata={"bad": object()},
        )


def test_runtime_only_loop_policies_do_not_break_request_json_schema():
    class NoopLoopPolicy(LoopPolicy):
        pass

    schema_classes = [
        RunRequest,
        ResumeRequest,
        DispatchRequest,
        ToolApprovalRequest,
        ToolApprovalRecoveryRequest,
    ]
    for schema_class in schema_classes:
        schema = schema_class.model_json_schema()
        assert "loop_policies" not in schema.get("properties", {})

    policy = NoopLoopPolicy()
    request = RunRequest(
        agent_name="assistant",
        messages=[Message.text("user", "hello")],
        loop_policies=(policy,),
    )

    assert request.loop_policies == (policy,)
    assert "loop_policies" not in request.model_dump()

    with pytest.raises(TypeError, match="LoopPolicy"):
        RunRequest(
            agent_name="assistant",
            messages=[Message.text("user", "hello")],
            loop_policies=(object(),),
        )


def test_dispatch_handle_validates_boundary_data():
    handle = DispatchHandle(
        dispatch_id="dispatch_1",
        session_id="sess_dispatch",
        backend="inline",
        status=DispatchStatus.COMPLETED,
        metadata={"events": 3},
    )

    assert handle.dispatch_id == "dispatch_1"
    assert handle.status == DispatchStatus.COMPLETED
    assert handle.metadata == {"events": 3}

    with pytest.raises(ValidationError, match="cannot be blank"):
        DispatchHandle(
            dispatch_id=" ",
            session_id="sess_dispatch",
            backend="inline",
        )

    with pytest.raises(ValidationError, match="JSON-compatible"):
        DispatchHandle(
            dispatch_id="dispatch_1",
            session_id="sess_dispatch",
            backend="inline",
            metadata={"bad": object()},
        )


def test_static_tool_policy_allows_by_default_and_denies_explicit_names():
    session = Session(
        id="sess_policy",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        status=SessionStatus.RUNNING,
    )
    request = ToolPolicyRequest(
        session=session,
        agent=AgentSpec(name="assistant", model="fake-model"),
        tool_name="write_file",
        tool_call_id="call_1",
        arguments={"path": "notes/result.txt"},
    )

    default_policy = StaticToolPolicy()
    default_result = asyncio.run(default_policy.authorize(request))
    assert default_result.decision == ToolPolicyDecision.ALLOW

    denied_policy = StaticToolPolicy(deny=["write_file"])
    denied_result = asyncio.run(denied_policy.authorize(request))
    assert denied_result.decision == ToolPolicyDecision.DENY
    assert denied_result.reason == "Tool denied by policy: write_file"


def test_session_defaults_causal_budget_id_to_session_id():
    session = Session(
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
    )

    assert session.id
    assert session.causal_budget_id == session.id


def test_static_tool_policy_allowlist_blocks_unlisted_tools_and_deny_wins():
    session = Session(
        id="sess_policy",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        status=SessionStatus.RUNNING,
    )
    request = ToolPolicyRequest(
        session=session,
        agent=AgentSpec(name="assistant", model="fake-model"),
        tool_name="exec_command",
        tool_call_id="call_1",
    )

    allowlist_policy = StaticToolPolicy(allow=["read_file"])
    allowlist_result = asyncio.run(allowlist_policy.authorize(request))
    assert allowlist_result.decision == ToolPolicyDecision.DENY
    assert allowlist_result.reason == "Tool not allowed by policy: exec_command"

    deny_wins_policy = StaticToolPolicy(allow=["exec_command"], deny=["exec_command"])
    deny_wins_result = asyncio.run(deny_wins_policy.authorize(request))
    assert deny_wins_result.decision == ToolPolicyDecision.DENY
    assert deny_wins_result.reason == "Tool denied by policy: exec_command"


def test_tool_policy_contract_rejects_invalid_boundary_data():
    session = Session(
        id="sess_policy",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        status=SessionStatus.RUNNING,
    )
    agent = AgentSpec(name="assistant", model="fake-model")

    with pytest.raises(ValidationError, match="cannot be blank"):
        ToolPolicyRequest(
            session=session,
            agent=agent,
            tool_name=" ",
            tool_call_id="call_1",
        )

    with pytest.raises(ValidationError, match="JSON-compatible"):
        ToolPolicyRequest(
            session=session,
            agent=agent,
            tool_name="read_file",
            tool_call_id="call_1",
            arguments={"bad": object()},
        )

    with pytest.raises(ValidationError, match="cannot be blank"):
        ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason=" ")

    with pytest.raises(ValidationError, match="JSON-compatible"):
        ToolPolicyResult(
            decision=ToolPolicyDecision.DENY,
            metadata={"bad": object()},
        )

    with pytest.raises(ValueError, match="cannot be blank"):
        StaticToolPolicy(allow=["read_file", " "])

    with pytest.raises(TypeError, match="iterable"):
        StaticToolPolicy(deny="read_file")  # type: ignore[arg-type]


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
    assert result_message.content[0].artifacts == [{"nested": {"value": "original"}}]


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
    knowledge_entry = KnowledgeEntry(
        id="item_1",
        text="memory",
        metadata=knowledge_metadata,
    )
    knowledge_hit = KnowledgeHit(entry=knowledge_entry)

    exec_artifacts = [{"nested": {"value": "original"}}]
    exec_result = ExecResult(artifacts=exec_artifacts)

    # Messages are frozen: mutating the shared instance is rejected outright,
    # so the request's transcript cannot be corrupted through the caller list.
    with pytest.raises(ValidationError):
        messages[0].content[0].text = "mutated"  # type: ignore[misc]
    tools[0]["input_schema"]["nested"]["value"] = "mutated"
    options["nested"]["value"] = "mutated"
    stream_arguments["nested"]["value"] = "mutated"
    secret_metadata["nested"]["value"] = "mutated"
    knowledge_metadata["nested"]["value"] = "mutated"
    knowledge_entry.metadata["nested"]["value"] = "mutated again"
    exec_artifacts[0]["nested"]["value"] = "mutated"

    assert request.messages[0].content[0].text == "start"
    assert request.tools == [{"input_schema": {"nested": {"value": "original"}}}]
    assert request.options == {"nested": {"value": "original"}}
    assert stream_event.payload["arguments"] == {"nested": {"value": "original"}}
    assert secret.metadata == {"nested": {"value": "original"}}
    assert knowledge_hit.entry.metadata == {"nested": {"value": "original"}}
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
        lambda value: KnowledgeEntry(id="item_1", text="memory", metadata={"bad": value}),
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


def test_clean_nonblank_helper_rejects_edge_whitespace_without_stripping():
    with pytest.raises(ValueError, match="must not start or end with whitespace"):
        require_clean_nonblank(" assistant", "name")

    with pytest.raises(ValueError, match="must not start or end with whitespace"):
        require_clean_nonblank("assistant ", "name")

    assert require_clean_nonblank("assistant builder", "name") == "assistant builder"


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
        lambda value: KnowledgeHit(entry=value),
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


def test_model_stream_event_rejects_unknown_type():
    with pytest.raises(ValueError, match="not a valid"):
        ModelStreamEvent(type="custom_provider_event")


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

    with pytest.raises(ValueError, match="Exception"):
        ModelStreamEvent.error("boom", cause="not an exception")  # type: ignore[arg-type]


def test_model_provider_error_carries_typed_classification_fields():
    error = ModelProviderError(
        "HTTP 429: rate limited",
        provider="anthropic",
        status_code=429,
        error_type="rate_limit_error",
        error_code="rate_limited",
        request_id="req_1",
        retryable=True,
        retry_after_s=2,
    )

    assert error.provider == "anthropic"
    assert error.retryable is True
    assert error.retry_after_s == 2.0
    assert type(error.retry_after_s) is float
    assert error.error_payload_fields() == {
        "provider": "anthropic",
        "status_code": 429,
        "provider_error_type": "rate_limit_error",
        "provider_error_code": "rate_limited",
        "request_id": "req_1",
        "retryable": True,
        "retry_after_s": 2.0,
    }

    unclassified = ModelProviderError("boom", provider="anthropic")
    assert unclassified.retryable is None
    assert unclassified.error_payload_fields() == {"provider": "anthropic"}


def test_model_provider_error_rejects_invalid_classification_fields():
    with pytest.raises(ValueError, match="retryable must be a boolean"):
        ModelProviderError("boom", provider="p", retryable="yes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="retry_after_s must be a non-negative number"):
        ModelProviderError("boom", provider="p", retry_after_s=-1.0)

    with pytest.raises(ValueError, match="retry_after_s must be a non-negative number"):
        ModelProviderError("boom", provider="p", retry_after_s=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="status_code must be a valid HTTP status code"):
        ModelProviderError("boom", provider="p", status_code=42)


def test_model_context_overflow_error_is_typed_and_never_retryable():
    error = ModelContextOverflowError(
        "context too large",
        provider="anthropic",
        status_code=400,
        error_type="invalid_request_error",
        request_id="req_overflow",
    )

    assert isinstance(error, ModelProviderError)
    assert error.retryable is False
    assert error.retry_after_s is None
    assert error.error_payload_fields() == {
        "provider": "anthropic",
        "status_code": 400,
        "provider_error_type": "invalid_request_error",
        "request_id": "req_overflow",
        "retryable": False,
    }


def test_model_stream_error_event_preserves_typed_provider_fields():
    cause = ModelProviderError(
        "HTTP 429: rate limited",
        provider="anthropic",
        status_code=429,
        error_type="rate_limit_error",
        request_id="req_1",
        retryable=True,
        retry_after_s=1.5,
    )

    event = ModelStreamEvent.error("HTTP 429: rate limited", cause=cause)

    assert event.payload == {
        "error": "HTTP 429: rate limited",
        "error_type": "ModelProviderError",
        "provider": "anthropic",
        "status_code": 429,
        "provider_error_type": "rate_limit_error",
        "request_id": "req_1",
        "retryable": True,
        "retry_after_s": 1.5,
    }


def test_model_stream_error_event_marks_context_overflow_cause():
    cause = ModelContextOverflowError(
        "context too large",
        provider="openai",
        status_code=400,
        error_code="context_length_exceeded",
    )

    event = ModelStreamEvent.error("context too large", cause=cause)

    assert event.payload == {
        "error": "context too large",
        "error_type": "ModelContextOverflowError",
        "provider": "openai",
        "status_code": 400,
        "provider_error_code": "context_length_exceeded",
        "retryable": False,
        "context_overflow": True,
    }
    # Non-overflow provider errors never carry the overflow marker.
    plain = ModelStreamEvent.error("boom", cause=ModelProviderError("boom", provider="openai"))
    assert "context_overflow" not in plain.payload


def test_model_stream_error_event_keeps_plain_payload_without_provider_cause():
    assert ModelStreamEvent.error("boom").payload == {"error": "boom"}
    assert ModelStreamEvent.error("boom", cause=RuntimeError("boom")).payload == {
        "error": "boom",
        "error_type": "RuntimeError",
    }


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


def test_retry_policy_validates_retry_controls():
    assert 529 in RetryPolicy().retry_on_status_codes

    policy = RetryPolicy(
        max_attempts=3,
        initial_delay_s=0.0,
        max_delay_s=2.0,
        backoff_multiplier=1.5,
        retry_on_status_codes=(429, 503),
    )

    assert policy.max_attempts == 3
    assert policy.retry_on_status_codes == (429, 503)

    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=0)

    with pytest.raises(ValidationError):
        RetryPolicy(retry_on_status_codes=(99,))

    with pytest.raises(ValidationError):
        RetryPolicy(retry_on_status_codes=(600,))


def test_retry_policy_classifies_common_status_code_formats():
    policy = RetryPolicy(max_attempts=2, retry_on_status_codes=(429, 500, 503, 529))

    retryable_errors = [
        "OpenAI API request failed with HTTP 500: unavailable",
        "Anthropic API request failed with HTTP 529: overloaded_error",
        "HTTP/2 503 service unavailable",
        "HTTP/1.1 500 internal server error",
        "provider failed with HTTP status 503",
        "provider failed with HTTP status code 503",
        "provider failed with status 429",
        "provider failed with status_code=429",
        "provider failed with status-code: 500",
        "response status code: 503",
    ]

    for error in retryable_errors:
        decision = retry_decision(policy=policy, attempt=1, error=error)
        assert decision.retry is True
        assert decision.reason == RetryReason.HTTP_STATUS
        assert decision.status_code in {429, 500, 503, 529}


def test_retry_policy_does_not_retry_permanent_quota_errors():
    permanent_errors = [
        "OpenAI API request failed with HTTP 429: insufficient_quota",
        "OpenAI API request failed with HTTP 429: You exceeded your current quota",
        "OpenAI API request failed with HTTP 429: run out of credits",
        "OpenAI API request failed with HTTP 429: hit your maximum monthly spend",
        "OpenAI API request failed with HTTP 429: check your plan and billing details",
        "OpenAI API request failed: insufficient_quota",
    ]

    for error in permanent_errors:
        decision = retry_decision(
            policy=RetryPolicy(max_attempts=2),
            attempt=1,
            error=error,
        )
        assert decision.retry is False
        assert decision.reason is None
        assert decision.status_code in {None, 429}


def test_retry_policy_permanent_patterns_do_not_mask_retryable_server_errors():
    decision = retry_decision(
        policy=RetryPolicy(max_attempts=2),
        attempt=1,
        error="Provider failed with HTTP 503: billing service temporarily unavailable",
    )

    assert decision.retry is True
    assert decision.reason == RetryReason.HTTP_STATUS
    assert decision.status_code == 503


def test_retry_policy_still_retries_rate_limit_429_errors():
    decision = retry_decision(
        policy=RetryPolicy(max_attempts=2),
        attempt=1,
        error="OpenAI API request failed with HTTP 429: Rate limit reached for requests",
    )

    assert decision.retry is True
    assert decision.reason == RetryReason.HTTP_STATUS
    assert decision.status_code == 429


def test_retry_policy_does_not_treat_unlabeled_numbers_as_status_codes():
    decision = retry_decision(
        policy=RetryPolicy(max_attempts=2),
        attempt=1,
        error="model emitted 500 tokens before failing validation",
    )

    assert decision.retry is False
    assert decision.status_code is None


def test_retry_policy_prefers_typed_status_code_over_message_text():
    # The message carries no HTTP token, but the typed status code is retryable.
    decision = retry_decision(
        policy=RetryPolicy(max_attempts=2, retry_on_status_codes=(429, 529)),
        attempt=1,
        error="Anthropic API request failed: overloaded_error",
        status_code=529,
    )

    assert decision.retry is True
    assert decision.reason == RetryReason.HTTP_STATUS
    assert decision.status_code == 529


def test_retry_policy_typed_retryable_false_is_terminal():
    # Message text looks like a transient timeout, but the provider classified
    # the failure as non-retryable; the typed verdict wins.
    decision = retry_decision(
        policy=RetryPolicy(max_attempts=3),
        attempt=1,
        error="request timed out",
        retryable=False,
    )

    assert decision.retry is False
    assert decision.reason is None


def test_retry_policy_typed_retryable_true_is_honored_without_pattern_match():
    decision = retry_decision(
        policy=RetryPolicy(max_attempts=2),
        attempt=1,
        error="provider failed for an unrecognized reason",
        status_code=418,
        retryable=True,
    )

    assert decision.retry is True
    assert decision.reason == RetryReason.CONNECTION
    assert decision.status_code == 418


def test_retry_policy_honors_retry_after_directive_capped_by_max_delay():
    decision = retry_decision(
        policy=RetryPolicy(max_attempts=2, max_delay_s=10.0),
        attempt=1,
        error="HTTP 429: rate limited",
        status_code=429,
        retry_after_s=45.0,
    )

    assert decision.retry is True
    # Retry-After (45s) is honored but bounded by max_delay_s (10s).
    assert decision.delay_seconds == 10.0


def test_retry_policy_uses_retry_after_below_max_delay_verbatim():
    decision = retry_decision(
        policy=RetryPolicy(max_attempts=2, max_delay_s=30.0, initial_delay_s=0.5),
        attempt=1,
        error="HTTP 503: service unavailable",
        status_code=503,
        retry_after_s=7.0,
    )

    assert decision.retry is True
    assert decision.delay_seconds == 7.0


def test_retry_policy_rejects_negative_retry_after():
    with pytest.raises(ValueError):
        retry_decision(
            policy=RetryPolicy(max_attempts=2),
            attempt=1,
            error="HTTP 503",
            retry_after_s=-1.0,
        )


def test_retry_policy_jitter_does_not_exceed_max_delay():
    policy = RetryPolicy(
        max_attempts=2,
        initial_delay_s=10.0,
        max_delay_s=10.0,
        jitter_s=5.0,
    )

    for _ in range(50):
        decision = retry_decision(
            policy=policy,
            attempt=1,
            error="HTTP 503",
        )
        assert decision.retry is True
        assert decision.delay_seconds <= policy.max_delay_s


def test_agent_spec_uses_explicit_system_prompt_field():
    spec = AgentSpec(
        name="assistant",
        model="fake-model",
        system_prompt="You are careful.",
    )

    assert spec.system_prompt == "You are careful."

    with pytest.raises(ValidationError):
        AgentSpec(name="assistant", model="fake-model", prompt="too vague")  # type: ignore[call-arg]


def test_agent_spec_declares_machine_checkable_workflow_tool_names() -> None:
    names = ["read_source", "edit_file"]

    spec = AgentSpec(
        name="reviewer",
        model="fake-model",
        workflow_tool_names=names,
    )
    names.append("stale_name")

    assert spec.workflow_tool_names == ("read_source", "edit_file")

    with pytest.raises(ValidationError, match="workflow_tool_names.*unique"):
        AgentSpec(
            name="reviewer",
            model="fake-model",
            workflow_tool_names=("read_source", "read_source"),
        )


def test_agent_spec_copies_provider_options():
    options = {"openai": {"prompt_cache_key": "tenant-a-agent"}}
    spec = AgentSpec(
        name="assistant",
        model="fake-model",
        provider_options=options,
    )

    options["openai"]["prompt_cache_key"] = "mutated"

    assert spec.provider_options == {"openai": {"prompt_cache_key": "tenant-a-agent"}}


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": " assistant", "model": "fake-model"},
        {"name": "assistant ", "model": "fake-model"},
        {"name": "assistant", "model": " fake-model"},
        {"name": "assistant", "model": "fake-model "},
    ],
)
def test_agent_spec_rejects_edge_whitespace_identity_fields(kwargs):
    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        AgentSpec(**kwargs)


def test_agent_spec_allows_internal_spaces_in_identity_fields():
    spec = AgentSpec(name="builder agent", model="fake model")

    assert spec.name == "builder agent"
    assert spec.model == "fake model"


def test_environment_spec_accepts_name_and_metadata():
    metadata = {"nested": {"value": "original"}}
    environment = Environment(EnvironmentSpec(name="local", metadata=metadata))

    metadata["nested"]["value"] = "mutated"

    assert environment.spec.name == "local"
    assert environment.spec.metadata == {"nested": {"value": "original"}}
    assert environment.workspace is None
    assert environment.artifact_store is None
    assert environment.runner is None
    assert environment.vault is None
    assert environment.mcp_servers == ()


def test_structured_output_spec_validates_json_schema_and_options():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    spec = StructuredOutputSpec(name="answer", json_schema=schema, max_retries=2)
    default_spec = StructuredOutputSpec(json_schema=schema)

    assert spec.name == "answer"
    assert spec.json_schema == schema
    assert spec.max_retries == 2
    assert default_spec.max_retries == 2
    assert spec.strategy == StructuredOutputStrategy.TOOL

    native_spec = StructuredOutputSpec(json_schema=schema, strategy="native")
    assert native_spec.strategy == StructuredOutputStrategy.NATIVE

    schema["required"].append("mutated")
    assert spec.json_schema["required"] == ["answer"]
    assert native_spec.json_schema["required"] == ["answer"]

    with pytest.raises(ValidationError, match="Invalid structured output JSON Schema"):
        StructuredOutputSpec(json_schema={"type": "not-a-json-schema-type"})

    with pytest.raises(ValidationError, match="JSON Schema must be an object"):
        StructuredOutputSpec(json_schema=["not", "an", "object"])  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="JSON-compatible"):
        StructuredOutputSpec(json_schema={"bad": object()})  # type: ignore[dict-item]

    with pytest.raises(ValidationError, match="cannot be blank"):
        StructuredOutputSpec(name=" ", json_schema=schema)

    with pytest.raises(ValidationError):
        StructuredOutputSpec(json_schema=schema, max_retries=9)

    with pytest.raises(ValidationError):
        StructuredOutputSpec(json_schema=schema, strategy="unknown")


def test_run_and_resume_requests_copy_structured_output_spec():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    spec = StructuredOutputSpec(json_schema=schema)
    request = RunRequest(
        agent_name="assistant",
        messages=[Message.text("user", "start")],
        structured_output=spec,
    )
    resume = ResumeRequest(
        session_id="sess_structured",
        messages=[Message.text("user", "continue")],
        structured_output=spec,
    )

    spec.json_schema["required"].append("mutated")

    assert request.structured_output is not spec
    assert resume.structured_output is not spec
    assert request.structured_output is not None
    assert resume.structured_output is not None
    assert request.structured_output.json_schema["required"] == ["answer"]
    assert resume.structured_output.json_schema["required"] == ["answer"]
    assert request.structured_output.strategy == spec.strategy
    assert resume.structured_output.strategy == spec.strategy


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
            task_id=" ",
            messages=[Message.text("user", "start")],
        )

    with pytest.raises(ValidationError, match="cannot be blank"):
        RunRequest(
            agent_name="assistant",
            task_id="task_1",
            task_worker_id=" ",
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


def test_run_request_requires_task_id_for_task_worker_id():
    with pytest.raises(ValidationError, match="task_worker_id requires task_id"):
        RunRequest(
            agent_name="assistant",
            task_worker_id="worker_1",
            messages=[Message.text("user", "start")],
        )


def test_runtime_identity_models_reject_edge_whitespace_fields():
    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        RunRequest(agent_name=" assistant", messages=[Message.text("user", "start")])

    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        RunRequest(
            agent_name="assistant",
            session_id="sess_1 ",
            messages=[Message.text("user", "start")],
        )

    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        RunRequest(
            agent_name="assistant",
            task_id="task_1",
            task_worker_id="worker_1 ",
            messages=[Message.text("user", "start")],
        )

    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        Session(
            id=" sess_1",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )

    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        Event(type=EventType.TOOL_CALL_STARTED, session_id="sess_1", tool_name=" echo")

    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        ToolContext(session_id="sess_1", workspace_id=" workspace")


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
        KnowledgeEntry(id=" ", text="memory")

    with pytest.raises(ValidationError, match="cannot be blank"):
        KnowledgeEntry(id="memory_1", text=" ")


def test_framework_spec_models_reject_edge_whitespace_identity_fields():
    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        ToolSpec(name=" echo")

    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        WorkflowSpec(name="workflow ")

    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        EnvironmentSpec(name=" local")

    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        ModelRequest(model=" fake-model", messages=[Message.text("user", "start")])

    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        McpServerSpec(name=" local", command=["node", "server.js"])

    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        SecretRef(name=" OPENAI_API_KEY")

    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        KnowledgeEntry(id=" memory_1", text="memory")

    entry = KnowledgeEntry(id="memory_1", text=" memory text ", source_uri="project-docs")
    assert entry.text == " memory text "
    assert entry.source_uri == "project-docs"


def test_mcp_server_rejects_blank_transport_values():
    with pytest.raises(ValidationError, match="cannot be blank"):
        McpServerSpec(name="local", command=["node", " "])

    with pytest.raises(ValidationError, match="cannot be blank"):
        McpServerSpec(name="remote", url=" ")


def test_mcp_server_rejects_blank_config_keys():
    with pytest.raises(ValidationError, match="cannot be blank"):
        McpServerSpec(name="local", command=["node"], env={" ": "production"})

    with pytest.raises(ValidationError, match="must not start or end with whitespace"):
        McpServerSpec(name="local", command=["node"], headers={" Authorization": "token"})

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
    entry = KnowledgeEntry(id="memory_1", text="memory")

    with pytest.raises(ValidationError, match="must be finite"):
        KnowledgeHit(entry=entry, score=float("nan"))

    with pytest.raises(ValidationError, match="must be finite"):
        KnowledgeHit(entry=entry, score=float("inf"))


def test_knowledge_hit_rejects_non_numeric_scores():
    entry = KnowledgeEntry(id="memory_1", text="memory")

    with pytest.raises(ValidationError, match="must be a number"):
        KnowledgeHit(entry=entry, score="0.3")  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="must be a number"):
        KnowledgeHit(entry=entry, score=True)  # type: ignore[arg-type]


def test_knowledge_hit_revalidates_constructed_items():
    class BadMetadata(dict):
        def items(self):
            raise RuntimeError("knowledge metadata traversal should not run")

    with pytest.raises(ValueError, match="`id` cannot be blank"):
        KnowledgeHit(
            entry=KnowledgeEntry.model_construct(
                id=" ",
                text="memory",
                metadata={},
            )
        )

    with pytest.raises(ValueError, match="JSON-compatible"):
        KnowledgeHit(
            entry=KnowledgeEntry.model_construct(
                id="memory_1",
                text="memory",
                metadata=BadMetadata({"bad": "value"}),
            )
        )


def test_knowledge_entry_copy_boundaries_reject_subclasses_before_attribute_access():
    class BadEntry(KnowledgeEntry):
        def __getattribute__(self, name):
            if name == "id":
                raise RuntimeError("knowledge entry id access should not run")
            return super().__getattribute__(name)

    entry = BadEntry.model_construct(
        id="memory_1",
        text="memory",
        metadata={},
    )

    with pytest.raises(TypeError, match="KnowledgeEntry"):
        copy_knowledge_entry(entry)

    with pytest.raises(TypeError, match="KnowledgeEntry"):
        KnowledgeHit(entry=entry)


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


def test_message_supports_provider_state_parts_on_assistant_messages():
    message = Message(
        role="assistant",
        content=[
            ProviderStatePart(
                provider="openai",
                state={"type": "reasoning", "id": "rs_1", "summary": []},
            )
        ],
    )

    part = message.content[0]
    assert isinstance(part, ProviderStatePart)
    assert part.type == "provider_state"
    assert part.provider == "openai"
    assert part.state == {"type": "reasoning", "id": "rs_1", "summary": []}


def test_provider_state_part_is_copied_and_json_validated():
    state = {"nested": {"value": "original"}}
    part = ProviderStatePart(provider="openai", state=state)
    copied = copy_message_part(part)
    state["nested"]["value"] = "mutated"

    assert isinstance(copied, ProviderStatePart)
    assert copied.state == {"nested": {"value": "original"}}

    with pytest.raises(ValueError, match="JSON-compatible"):
        ProviderStatePart(provider="openai", state={"bad": object()})

    with pytest.raises(ValueError, match="cannot be blank"):
        ProviderStatePart(provider=" ", state={})


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

    with pytest.raises(ValueError, match="system messages only support"):
        Message(
            role="system",
            content=[ProviderStatePart(provider="openai", state={})],
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


def test_message_supports_user_file_parts():
    attachment = file_attachment(
        artifact_id="art_1",
        kind=FileAttachmentKind.IMAGE,
        filename="invoice.png",
        content_type="image/png",
        size_bytes=5,
    )

    message = Message(
        role="user",
        content=[TextPart(text="Read the invoice."), FilePart(attachment=attachment)],
    )

    part = message.content[1]
    assert isinstance(part, FilePart)
    assert part.attachment == attachment

    # JSON round-trip (the storage path) rebuilds the same part type.
    rebuilt = Message(**message.model_dump(mode="json"))
    assert isinstance(rebuilt.content[1], FilePart)
    assert rebuilt.content[1].attachment == attachment

    with pytest.raises(ValueError, match="system messages only support"):
        Message(role="system", content=[FilePart(attachment=attachment)])

    with pytest.raises(ValueError, match="assistant messages only support"):
        Message(role="assistant", content=[FilePart(attachment=attachment)])


def test_file_part_validates_and_detaches_attachment_payload():
    with pytest.raises(ValidationError):
        FilePart(attachment={"artifact_id": "art_1"})

    with pytest.raises(ValidationError):
        FilePart(attachment="not-an-object")  # type: ignore[arg-type]

    payload = file_attachment(
        artifact_id="art_1",
        kind=FileAttachmentKind.DOCUMENT,
        filename="report.pdf",
        content_type="application/pdf",
        size_bytes=9,
        metadata={"source": "upload"},
    )
    part = FilePart(attachment=payload)
    payload["metadata"]["source"] = "mutated"
    assert part.attachment["metadata"] == {"source": "upload"}

    copied = copy_message_part(part)
    assert isinstance(copied, FilePart)
    assert copied.attachment == part.attachment


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


def test_resume_request_requires_existing_session_id_and_new_messages():
    request = ResumeRequest(
        session_id="sess_existing",
        messages=[Message.text("user", "continue")],
        model="upgraded-model",
        metadata={"source": "test"},
    )

    assert request.session_id == "sess_existing"
    assert request.messages[0].content[0].text == "continue"
    assert request.model == "upgraded-model"
    assert request.metadata == {"source": "test"}
    assert request.max_steps == 16

    with pytest.raises(ValidationError, match="cannot be blank"):
        ResumeRequest(
            session_id=" ",
            messages=[Message.text("user", "continue")],
        )

    with pytest.raises(ValidationError, match="cannot be empty"):
        ResumeRequest(session_id="sess_existing", messages=[])

    with pytest.raises(ValidationError, match="cannot be blank"):
        ResumeRequest(
            session_id="sess_existing",
            messages=[Message.text("user", "continue")],
            model=" ",
        )


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


def test_tool_spec_is_shared_from_class_level_definition():
    first = EchoTool()
    second = EchoTool()

    # ToolSpec is frozen and deeply immutable, so instances safely share the
    # class-level spec instead of deep-copying it per instantiation.
    assert first.spec == second.spec
    assert first.spec is second.spec

    with pytest.raises(Exception):
        first.spec.name = "changed"  # type: ignore[misc]

    # Materialized schemas remain isolated per access.
    first.schema["properties"]["text"]["type"] = "number"
    assert second.schema["properties"]["text"]["type"] == "string"


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


def test_tool_spec_effect_defaults_and_validates():
    default = ToolSpec(name="example")
    assert default.effect is ToolEffect.EXTERNAL

    idempotent = ToolSpec(name="refund", effect="idempotent")
    assert idempotent.effect is ToolEffect.IDEMPOTENT

    copied = idempotent.model_copy(update={"effect": ToolEffect.NONE})
    assert copied.effect is ToolEffect.NONE

    with pytest.raises(ValidationError):
        ToolSpec(name="bad", effect="retryable")  # type: ignore[arg-type]


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

    with pytest.raises(ValidationError):
        ExecResult(stdout_truncated="yes")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        ExecResult(stderr_truncated=1)  # type: ignore[arg-type]


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


def test_local_workspace_reads_writes_and_lists_files(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")

    asyncio.run(workspace.write_bytes("notes/todo.txt", b"ship it"))

    assert workspace.id == "local"
    read_result = asyncio.run(workspace.read_bytes("notes/todo.txt"))
    list_result = asyncio.run(workspace.list("**/*.txt"))

    assert read_result.content == b"ship it"
    assert read_result.total_bytes == 7
    assert read_result.truncated is False
    assert list_result.paths == ("notes/todo.txt",)
    assert list_result.total_count == 1
    assert list_result.truncated is False


def test_local_artifact_store_puts_reads_lists_and_deletes_artifacts(tmp_path):
    with pytest.raises(ValueError, match="surrogate code points"):
        LocalArtifactStore(tmp_path / "invalid-artifacts", store_id="artifacts_\ud800")

    store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")

    session_artifact = asyncio.run(
        store.put_bytes(
            b"invoice text",
            filename="invoice.txt",
            content_type="text/plain",
            session_id="sess_1",
            agent_name="assistant",
            environment_name="local-dev",
            metadata={"source": "upload"},
        )
    )
    environment_artifact = asyncio.run(
        store.put_bytes(
            b"shared text",
            filename="shared.txt",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="local-dev",
        )
    )

    read_result = asyncio.run(store.read_bytes(session_artifact.id))
    session_list = asyncio.run(store.list(scope=ArtifactScope.SESSION, session_id="sess_1"))
    environment_list = asyncio.run(
        store.list(
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="local-dev",
        )
    )

    assert store.id == "artifacts"
    assert read_result.metadata == session_artifact
    assert read_result.content == b"invoice text"
    assert read_result.total_bytes == 12
    assert read_result.truncated is False
    assert session_list.artifacts == (session_artifact,)
    assert session_list.total_count == 1
    assert session_list.truncated is False
    assert environment_list.artifacts == (environment_artifact,)

    asyncio.run(store.delete(session_artifact.id))
    with pytest.raises(FileNotFoundError):
        asyncio.run(store.read_bytes(session_artifact.id))

    store.id = "artifacts_\ud800"
    with pytest.raises(ValueError, match="surrogate code points"):
        Environment(
            EnvironmentSpec(name="invalid-store"),
            artifact_store=store,
        )


def test_local_artifact_store_enforces_scope_owners_and_limits(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ValueError, match="session_id"):
        asyncio.run(store.put_bytes(b"content", filename="a.txt"))

    with pytest.raises(ValueError, match="environment_name"):
        asyncio.run(
            store.put_bytes(
                b"content",
                filename="a.txt",
                scope=ArtifactScope.ENVIRONMENT,
            )
        )

    artifact = asyncio.run(
        store.put_bytes(
            b"abcdef",
            filename="a.txt",
            session_id="sess_1",
        )
    )
    read_result = asyncio.run(store.read_bytes(artifact.id, max_bytes=3))

    assert read_result.content == b"abc"
    assert read_result.total_bytes == 6
    assert read_result.truncated is True

    with pytest.raises(TypeError, match="bytes"):
        asyncio.run(
            store.put_bytes(
                "content",  # type: ignore[arg-type]
                filename="a.txt",
                session_id="sess_1",
            )
        )

    with pytest.raises(ValueError, match="limit"):
        asyncio.run(store.list(limit=0))

    for invalid_id in (
        "",
        " ",
        " art_1",
        "not_artifact",
        "art_1",
        f"art_{'a' * 300}",
        "art_\x00x",
        "art_\nx",
        f"art_{'A' * 32}",
    ):
        with pytest.raises(InvalidArtifactIdError):
            asyncio.run(store.read_bytes(invalid_id))
        with pytest.raises(InvalidArtifactIdError):
            asyncio.run(store.delete(invalid_id))

    non_artifact_dir = tmp_path / "artifacts" / "not_artifact"
    non_artifact_dir.mkdir()
    with pytest.raises(ValueError, match="local artifact id"):
        asyncio.run(store.delete("not_artifact"))
    assert non_artifact_dir.exists()

    malformed_artifact_dir = tmp_path / "artifacts" / "art_not_addressable"
    malformed_artifact_dir.mkdir()
    malformed_metadata = ArtifactMetadata(
        id=malformed_artifact_dir.name,
        filename="malformed.txt",
        content_type="text/plain",
        size_bytes=0,
        session_id="sess_1",
    )
    (malformed_artifact_dir / "metadata.json").write_text(
        malformed_metadata.model_dump_json(),
        encoding="utf-8",
    )
    (malformed_artifact_dir / "content").write_bytes(b"")

    listed = asyncio.run(store.list())

    assert malformed_artifact_dir.name not in {artifact.id for artifact in listed.artifacts}


def test_local_artifact_store_rejects_symlinked_artifact_files(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    artifact = asyncio.run(
        store.put_bytes(
            b"artifact-content",
            filename="artifact.txt",
            session_id="sess_1",
        )
    )
    outside_content = tmp_path / "outside.txt"
    outside_content.write_bytes(b"host-secret-data")
    content_path = store.root / artifact.id / "content"
    content_path.unlink()
    try:
        content_path.symlink_to(outside_content)
    except OSError:
        pytest.skip("Symbolic links are unavailable on this platform.")

    with pytest.raises(ValueError, match="not a regular file"):
        asyncio.run(store.read_bytes(artifact.id))

    second = asyncio.run(
        store.put_bytes(
            b"second-content",
            filename="second.txt",
            session_id="sess_1",
        )
    )
    metadata_path = store.root / second.id / "metadata.json"
    outside_metadata = tmp_path / "outside-metadata.json"
    outside_metadata.write_bytes(metadata_path.read_bytes())
    metadata_path.unlink()
    metadata_path.symlink_to(outside_metadata)

    with pytest.raises(ValueError, match="not a regular file"):
        asyncio.run(store.read_bytes(second.id))


def test_local_artifact_store_without_directory_fd_support(monkeypatch, tmp_path):
    monkeypatch.setattr(artifact_local_module, "_SUPPORTS_DIRECTORY_FD", False)
    store = LocalArtifactStore(tmp_path / "artifacts")
    artifact = asyncio.run(
        store.put_bytes(
            b"fallback-content",
            filename="fallback.txt",
            session_id="sess_1",
        )
    )

    read = asyncio.run(store.read_bytes(artifact.id, max_bytes=8))
    listed = asyncio.run(store.list())

    assert read.content == b"fallback"
    assert read.total_bytes == len(b"fallback-content")
    assert read.truncated is True
    assert listed.artifacts == (artifact,)

    outside_content = tmp_path / "outside.txt"
    outside_content.write_bytes(b"host-secret-data")
    content_path = store.root / artifact.id / "content"
    content_path.unlink()
    try:
        content_path.symlink_to(outside_content)
    except OSError:
        content_path.write_bytes(b"fallback-content")
    else:
        with pytest.raises(ValueError, match="not a regular file"):
            asyncio.run(store.read_bytes(artifact.id))

    asyncio.run(store.delete(artifact.id))
    assert not (store.root / artifact.id).exists()


def test_local_artifact_store_classifies_permission_failures_as_unavailable(
    monkeypatch,
    tmp_path,
):
    store = LocalArtifactStore(tmp_path / "artifacts")
    artifact = asyncio.run(
        store.put_bytes(
            b"artifact-content",
            filename="artifact.txt",
            session_id="sess_1",
        )
    )
    content_path = store.root / artifact.id / "content"
    real_open = artifact_local_module.os.open

    def deny_content_open(path, flags, mode=0o777, *, dir_fd=None):
        if path == "content" or path == content_path:
            raise PermissionError(errno.EACCES, "Permission denied", str(content_path))
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifact_local_module.os, "open", deny_content_open)

    with pytest.raises(ArtifactStoreUnavailableError) as exc_info:
        asyncio.run(store.read_bytes(artifact.id))

    assert isinstance(exc_info.value.__cause__, PermissionError)


def test_local_artifact_store_does_not_hide_list_backend_failures(monkeypatch, tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    asyncio.run(
        store.put_bytes(
            b"artifact-content",
            filename="artifact.txt",
            session_id="sess_1",
        )
    )

    def deny_metadata_read(_path, *, parent_fd=None):
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(artifact_local_module, "_load_metadata", deny_metadata_read)

    with pytest.raises(ArtifactStoreUnavailableError) as exc_info:
        asyncio.run(store.list())

    assert isinstance(exc_info.value.__cause__, PermissionError)


def test_local_artifact_store_rejects_replaced_root(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    artifact = asyncio.run(
        store.put_bytes(
            b"artifact-content",
            filename="artifact.txt",
            session_id="sess_1",
        )
    )
    original_root = tmp_path / "original-artifacts"
    store.root.rename(original_root)
    store.root.mkdir()

    operations = (
        lambda: asyncio.run(store.list()),
        lambda: asyncio.run(store.read_bytes(artifact.id)),
        lambda: asyncio.run(
            store.put_bytes(
                b"redirected-content",
                filename="redirected.txt",
                session_id="sess_1",
            )
        ),
        lambda: asyncio.run(store.delete(artifact.id)),
    )
    for operation in operations:
        with pytest.raises(ArtifactStoreUnavailableError, match="root changed"):
            operation()

    assert list(store.root.iterdir()) == []
    assert (original_root / artifact.id / "content").read_bytes() == b"artifact-content"

    store.root.rmdir()
    outside_root = tmp_path / "outside-artifacts"
    outside_root.mkdir()
    try:
        store.root.symlink_to(outside_root, target_is_directory=True)
    except OSError:
        pass
    else:
        with pytest.raises(ArtifactStoreUnavailableError, match="root changed"):
            asyncio.run(store.list())
        assert list(outside_root.iterdir()) == []


def test_local_artifact_store_anchors_inflight_write_to_open_root(monkeypatch, tmp_path):
    if not artifact_local_module._SUPPORTS_DIRECTORY_FD:
        pytest.skip("Directory-relative filesystem operations are unavailable.")

    store = LocalArtifactStore(tmp_path / "artifacts")
    original_root = tmp_path / "original-artifacts"
    replacement_root = tmp_path / "replacement-artifacts"
    replacement_root.mkdir()
    real_mkdir = artifact_local_module.os.mkdir
    swapped = False

    def swap_root_before_artifact_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if dir_fd is not None and not swapped:
            store.root.rename(original_root)
            replacement_root.rename(store.root)
            swapped = True
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifact_local_module.os, "mkdir", swap_root_before_artifact_mkdir)

    with pytest.raises(ArtifactStoreUnavailableError, match="root changed"):
        asyncio.run(
            store.put_bytes(
                b"anchored-content",
                filename="anchored.txt",
                session_id="sess_1",
            )
        )

    assert swapped is True
    assert list(store.root.iterdir()) == []
    created = list(original_root.glob("art_*/content"))
    assert len(created) == 1
    assert created[0].read_bytes() == b"anchored-content"


def test_local_artifact_store_detects_windows_reparse_points():
    assert artifact_local_module._is_windows_reparse_point(
        SimpleNamespace(st_file_attributes=0x400, st_reparse_tag=0)  # type: ignore[arg-type]
    )
    assert artifact_local_module._is_windows_reparse_point(
        SimpleNamespace(st_file_attributes=0, st_reparse_tag=0xA0000003)  # type: ignore[arg-type]
    )
    assert not artifact_local_module._is_windows_reparse_point(
        SimpleNamespace(st_file_attributes=0, st_reparse_tag=0)  # type: ignore[arg-type]
    )


def test_artifact_result_types_validate_boundary_values():
    artifact = ArtifactMetadata(
        id="art_1",
        filename="a.txt",
        content_type="text/plain",
        size_bytes=3,
        session_id="sess_1",
        metadata={"nested": {"items": [1]}},
    )
    read_result = ArtifactReadResult(
        metadata=artifact,
        content=b"abc",
        total_bytes=3,
    )
    list_result = ArtifactListResult(artifacts=[artifact], total_count=1)

    assert read_result.metadata == artifact
    assert list_result.artifacts == (artifact,)

    with pytest.raises(ValidationError, match="frozen"):
        artifact.content_type = "text/html"  # type: ignore[misc]
    with pytest.raises(TypeError, match="cannot be mutated"):
        artifact.metadata["new"] = True
    with pytest.raises(TypeError, match="cannot be mutated"):
        artifact.metadata["nested"]["items"].append(2)
    with pytest.raises(TypeError):
        dict.__setitem__(artifact.metadata, "base_mutation", True)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        list.append(artifact.metadata["nested"]["items"], 2)
    with pytest.raises(TypeError, match="cannot be mutated"):
        artifact.metadata._data = {}  # type: ignore[attr-defined]
    assert artifact.model_dump(mode="json")["metadata"] == {"nested": {"items": [1]}}

    default_metadata = ArtifactMetadata(
        id="art_default",
        filename="default.txt",
        size_bytes=0,
        session_id="sess_1",
    )
    with pytest.raises(TypeError):
        default_metadata.metadata["new"] = True  # type: ignore[index]
    restored_artifact = pickle.loads(pickle.dumps(artifact))
    assert restored_artifact == artifact
    assert restored_artifact.model_dump(mode="json") == artifact.model_dump(mode="json")

    copied_artifact = artifact.model_copy(
        update={"metadata": {"nested": {"items": [2]}}},
    )
    assert copied_artifact.metadata == {"nested": {"items": [2]}}
    with pytest.raises(TypeError, match="cannot be mutated"):
        copied_artifact.metadata["nested"]["items"].append(3)
    with pytest.raises(ValidationError, match="control characters"):
        artifact.model_copy(update={"content_type": "text/plain\r\nX-Bad: value"})
    with pytest.raises(ValidationError, match="Extra inputs"):
        artifact.model_copy(update={"unknown_field": True})

    sparse_artifact = ArtifactMetadata(
        id="art_sparse",
        filename="sparse.txt",
        size_bytes=0,
        session_id="sess_1",
    )
    sparse_copy = sparse_artifact.model_copy(update={"content_type": "text/plain"})
    assert sparse_copy.model_fields_set == sparse_artifact.model_fields_set | {"content_type"}
    assert sparse_copy.model_dump(mode="json", exclude_unset=True) == {
        "id": "art_sparse",
        "filename": "sparse.txt",
        "content_type": "text/plain",
        "size_bytes": 0,
        "session_id": "sess_1",
    }

    with pytest.raises(ValueError, match="session_id"):
        ArtifactMetadata(
            id="art_1",
            filename="a.txt",
            size_bytes=0,
        )

    with pytest.raises(ValueError, match="environment_name"):
        ArtifactMetadata(
            id="art_1",
            filename="a.txt",
            size_bytes=0,
            scope=ArtifactScope.ENVIRONMENT,
        )

    with pytest.raises(ValueError, match="surrogate code points"):
        ArtifactMetadata(
            id="art_1",
            filename="bad\ud800.txt",
            size_bytes=0,
            session_id="sess_1",
        )

    for field_name in ("id", "session_id", "agent_name", "environment_name"):
        values = {
            "id": "art_1",
            "filename": "a.txt",
            "size_bytes": 0,
            "session_id": "sess_1",
        }
        values[field_name] = f"{field_name}_\ud800"
        with pytest.raises(ValueError, match="surrogate code points"):
            ArtifactMetadata(**values)

    for metadata in ({"bad_\ud800": "value"}, {"nested": ["bad_\ud800"]}):
        with pytest.raises(ValueError, match="surrogate code points"):
            ArtifactMetadata(
                id="art_1",
                filename="a.txt",
                size_bytes=0,
                session_id="sess_1",
                metadata=metadata,
            )

    for content_type in ("text/pläin", "text/\ud800"):
        with pytest.raises(ValueError, match="printable ASCII"):
            ArtifactMetadata(
                id="art_1",
                filename="a.txt",
                content_type=content_type,
                size_bytes=0,
                session_id="sess_1",
            )

    with pytest.raises(ValueError, match="at most 1024"):
        ArtifactMetadata(
            id="art_1",
            filename="a.txt",
            content_type="x" * 1025,
            size_bytes=0,
            session_id="sess_1",
        )

    invalid_constructed_metadata = ArtifactMetadata.model_construct(
        id="art_1",
        filename="a.txt",
        content_type="text/\ud800",
        size_bytes=0,
        scope=ArtifactScope.SESSION,
        session_id="sess_1",
        agent_name=None,
        environment_name=None,
        created_at=artifact.created_at,
        metadata={},
    )
    with pytest.raises(ValueError, match="printable ASCII"):
        ArtifactReadResult(
            metadata=invalid_constructed_metadata,
            content=b"",
            total_bytes=0,
        )
    with pytest.raises(ValueError, match="printable ASCII"):
        ArtifactListResult(
            artifacts=(invalid_constructed_metadata,),
            total_count=1,
        )

    with pytest.raises(TypeError, match="content"):
        ArtifactReadResult(metadata=artifact, content="text", total_bytes=4)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="total_bytes"):
        ArtifactReadResult(metadata=artifact, content=b"", total_bytes=-1)

    with pytest.raises(TypeError, match="truncated"):
        ArtifactReadResult(metadata=artifact, content=b"", total_bytes=0, truncated=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="smaller than content"):
        ArtifactReadResult(metadata=artifact, content=b"abc", total_bytes=1)

    with pytest.raises(ValueError, match="truncated must match"):
        ArtifactReadResult(metadata=artifact, content=b"abc", total_bytes=3, truncated=True)

    with pytest.raises(ValueError, match="truncated must match"):
        ArtifactReadResult(metadata=artifact, content=b"abc", total_bytes=4, truncated=False)

    mismatched_size_metadata = artifact.model_copy(update={"size_bytes": 4})
    with pytest.raises(ValueError, match="size_bytes must equal total_bytes"):
        ArtifactReadResult(
            metadata=mismatched_size_metadata,
            content=b"abc",
            total_bytes=3,
        )

    with pytest.raises(ValueError, match="different artifact id"):
        copy_artifact_read_result(read_result, expected_artifact_id="art_other")
    with pytest.raises(ValueError, match="requested byte limit"):
        copy_artifact_read_result(read_result, max_content_bytes=2)

    with pytest.raises(TypeError, match="artifacts"):
        ArtifactListResult(artifacts="a.txt", total_count=1)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="artifact entries"):
        ArtifactListResult(artifacts=[artifact, "bad"], total_count=2)  # type: ignore[list-item]

    with pytest.raises(ValueError, match="total_count"):
        ArtifactListResult(artifacts=[], total_count=-1)

    with pytest.raises(TypeError, match="truncated"):
        ArtifactListResult(artifacts=[], total_count=0, truncated=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="required"):
        ArtifactListResult(artifacts=[artifact], total_count=None, truncated=False)

    with pytest.raises(ValueError, match="must equal artifacts"):
        ArtifactListResult(artifacts=[artifact], total_count=2, truncated=False)

    with pytest.raises(ValueError, match="smaller than artifacts"):
        ArtifactListResult(artifacts=[artifact, artifact], total_count=1, truncated=True)


def test_file_attachment_contracts_are_json_safe_references():
    payload = file_attachment(
        artifact_id="art_1",
        kind=FileAttachmentKind.IMAGE,
        filename="invoice.png",
        content_type="image/png",
        size_bytes=123,
        metadata={"source": "upload"},
    )
    attachment = file_attachment_from_payload(payload)
    resolved = ResolvedFileAttachment(
        artifact_id="art_1",
        kind=FileAttachmentKind.IMAGE,
        filename="invoice.png",
        content_type="image/png",
        data_base64="aGVsbG8=",
    )

    assert payload == {
        "type": "cayu.file_attachment.v1",
        "artifact_id": "art_1",
        "kind": "image",
        "filename": "invoice.png",
        "content_type": "image/png",
        "size_bytes": 123,
        "metadata": {"source": "upload"},
    }
    assert attachment is not None
    assert attachment.artifact_id == "art_1"
    assert resolved.kind == FileAttachmentKind.IMAGE
    assert file_attachment_from_payload({"type": "other"}) is None

    with pytest.raises(ValidationError, match="cannot be blank"):
        FileAttachment(
            artifact_id=" ",
            kind=FileAttachmentKind.IMAGE,
            filename="invoice.png",
            content_type="image/png",
            size_bytes=123,
        )

    with pytest.raises(ValidationError, match="greater than zero"):
        FileAttachment(
            artifact_id="art_1",
            kind=FileAttachmentKind.IMAGE,
            filename="invoice.png",
            content_type="image/png",
            size_bytes=0,
        )

    with pytest.raises(ValidationError, match="Image file attachments require"):
        FileAttachment(
            artifact_id="art_1",
            kind=FileAttachmentKind.IMAGE,
            filename="invoice.pdf",
            content_type="application/pdf",
            size_bytes=123,
        )

    with pytest.raises(ValidationError, match="Document file attachments require"):
        FileAttachment(
            artifact_id="art_1",
            kind=FileAttachmentKind.DOCUMENT,
            filename="invoice.png",
            content_type="image/png",
            size_bytes=123,
        )

    with pytest.raises(ValidationError, match="Image file attachments require"):
        ResolvedFileAttachment(
            artifact_id="art_1",
            kind=FileAttachmentKind.IMAGE,
            filename="invoice.pdf",
            content_type="application/pdf",
            data_base64="aGVsbG8=",
        )

    with pytest.raises(ValidationError, match="JSON-compatible"):
        file_attachment(
            artifact_id="art_1",
            kind=FileAttachmentKind.IMAGE,
            filename="invoice.png",
            content_type="image/png",
            size_bytes=123,
            metadata={"bad": object()},
        )


def test_workspace_result_types_validate_boundary_values():
    list_result = WorkspaceListResult(paths=["a.txt", "b.txt"], total_count=2)
    truncated_read = WorkspaceReadResult(
        content=b"abc",
        total_bytes=6,
        truncated=True,
    )
    truncated_list = WorkspaceListResult(
        paths=["a.txt"],
        total_count=2,
        truncated=True,
    )
    unknown_total_truncated_list = WorkspaceListResult(
        paths=["a.txt"],
        total_count=None,
        truncated=True,
    )

    assert list_result.paths == ("a.txt", "b.txt")
    assert truncated_read.content == b"abc"
    assert truncated_read.total_bytes == 6
    assert truncated_read.truncated is True
    assert truncated_list.paths == ("a.txt",)
    assert truncated_list.total_count == 2
    assert truncated_list.truncated is True
    assert unknown_total_truncated_list.paths == ("a.txt",)
    assert unknown_total_truncated_list.total_count is None
    assert unknown_total_truncated_list.truncated is True

    with pytest.raises(TypeError, match="content"):
        WorkspaceReadResult(content="text", total_bytes=4)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="total_bytes"):
        WorkspaceReadResult(content=b"", total_bytes=-1)

    with pytest.raises(TypeError, match="truncated"):
        WorkspaceReadResult(content=b"", total_bytes=0, truncated=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="smaller than content"):
        WorkspaceReadResult(content=b"abc", total_bytes=1)

    with pytest.raises(ValueError, match="truncated must match"):
        WorkspaceReadResult(content=b"abc", total_bytes=3, truncated=True)

    with pytest.raises(ValueError, match="truncated must match"):
        WorkspaceReadResult(content=b"abc", total_bytes=4, truncated=False)

    with pytest.raises(TypeError, match="paths"):
        WorkspaceListResult(paths="a.txt", total_count=1)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="paths entries"):
        WorkspaceListResult(paths=["a.txt", 1], total_count=2)  # type: ignore[list-item]

    with pytest.raises(ValueError, match="total_count"):
        WorkspaceListResult(paths=[], total_count=-1)

    with pytest.raises(TypeError, match="truncated"):
        WorkspaceListResult(paths=[], total_count=0, truncated=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="required"):
        WorkspaceListResult(paths=["a.txt"], total_count=None, truncated=False)

    with pytest.raises(ValueError, match="must equal paths"):
        WorkspaceListResult(paths=["a.txt"], total_count=2, truncated=False)

    with pytest.raises(ValueError, match="smaller than paths"):
        WorkspaceListResult(paths=["a.txt", "b.txt"], total_count=1, truncated=True)


def test_local_workspace_enforces_read_and_list_limits(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")

    asyncio.run(workspace.write_bytes("a.txt", b"abcdef"))
    asyncio.run(workspace.write_bytes("b.txt", b""))

    read_result = asyncio.run(workspace.read_bytes("a.txt", max_bytes=3))
    list_result = asyncio.run(workspace.list("*.txt", limit=1))

    assert read_result.content == b"abc"
    assert read_result.total_bytes == 6
    assert read_result.truncated is True
    assert len(list_result.paths) == 1
    assert list_result.paths[0] in {"a.txt", "b.txt"}
    assert list_result.total_count is None
    assert list_result.truncated is True


def test_local_workspace_bounded_read_uses_extra_byte_for_truncation(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")

    asyncio.run(workspace.write_bytes("a.txt", b"abcdef"))

    result = asyncio.run(workspace.read_bytes("a.txt", max_bytes=3))
    exact_result = asyncio.run(workspace.read_bytes("a.txt", max_bytes=6))

    assert result.content == b"abc"
    assert result.total_bytes == 6
    assert result.truncated is True
    assert exact_result.content == b"abcdef"
    assert exact_result.total_bytes == 6
    assert exact_result.truncated is False


def test_local_workspace_unbounded_read_reports_returned_content_size(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")

    asyncio.run(workspace.write_bytes("a.txt", b"abcdef"))

    result = asyncio.run(workspace.read_bytes("a.txt"))

    assert result.content == b"abcdef"
    assert result.total_bytes == len(result.content)
    assert result.truncated is False


def test_local_workspace_rejects_paths_outside_root(tmp_path):
    workspace = LocalWorkspace(tmp_path)

    with pytest.raises(ValueError, match="relative"):
        asyncio.run(workspace.read_bytes(str(tmp_path / "file.txt")))

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.read_bytes("../outside.txt"))

    with pytest.raises(ValueError, match="pattern"):
        asyncio.run(workspace.list("../*"))


def test_local_workspace_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}_outside.txt"
    outside.write_bytes(b"secret")
    (tmp_path / "link.txt").symlink_to(outside)
    workspace = LocalWorkspace(tmp_path)

    try:
        with pytest.raises(ValueError, match="escapes"):
            asyncio.run(workspace.read_bytes("link.txt"))
    finally:
        outside.unlink(missing_ok=True)


def test_local_workspace_rejects_delete_symlink_leaf_inside_root(tmp_path):
    target = tmp_path / "target.txt"
    target.write_bytes(b"keep")
    (tmp_path / "link.txt").symlink_to(target)
    workspace = LocalWorkspace(tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.delete("link.txt"))

    assert target.read_bytes() == b"keep"
    assert (tmp_path / "link.txt").is_symlink()


def test_local_workspace_rejects_write_symlink_leaf_inside_root(tmp_path):
    target = tmp_path / "target.txt"
    target.write_bytes(b"keep")
    (tmp_path / "link.txt").symlink_to(target)
    workspace = LocalWorkspace(tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.write_bytes("link.txt", b"overwrite"))

    assert target.read_bytes() == b"keep"
    assert (tmp_path / "link.txt").is_symlink()


def test_local_workspace_rejects_delete_through_symlink_parent_inside_root(tmp_path):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target = target_dir / "a.txt"
    target.write_bytes(b"keep")
    (tmp_path / "link").symlink_to(target_dir)
    workspace = LocalWorkspace(tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.delete("link/a.txt"))

    assert target.read_bytes() == b"keep"
    assert (tmp_path / "link").is_symlink()


def test_local_workspace_rejects_write_through_symlink_parent_inside_root(tmp_path):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target = target_dir / "a.txt"
    target.write_bytes(b"keep")
    (tmp_path / "link").symlink_to(target_dir)
    workspace = LocalWorkspace(tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.write_bytes("link/a.txt", b"overwrite"))

    assert target.read_bytes() == b"keep"
    assert (tmp_path / "link").is_symlink()


def test_local_workspace_list_skips_symlink_paths_inside_root(tmp_path):
    target = tmp_path / "target.txt"
    target.write_bytes(b"keep")
    (tmp_path / "link.txt").symlink_to(target)
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    nested = target_dir / "a.txt"
    nested.write_bytes(b"nested")
    (tmp_path / "link_dir").symlink_to(target_dir)
    workspace = LocalWorkspace(tmp_path)

    result = asyncio.run(workspace.list("**/*.txt"))

    assert result.paths == ("target.txt", "target/a.txt")
    assert result.total_count == 2
    assert result.truncated is False


def test_local_workspace_rejects_invalid_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalWorkspace(tmp_path / "missing")

    file_path = tmp_path / "file.txt"
    file_path.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        LocalWorkspace(file_path)

    workspace = LocalWorkspace(tmp_path)
    with pytest.raises(TypeError, match="bytes"):
        asyncio.run(workspace.write_bytes("file.txt", "text"))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="max_bytes"):
        asyncio.run(workspace.read_bytes("file.txt", max_bytes="1"))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="limit"):
        asyncio.run(workspace.list("*.txt", limit=0))


def test_local_runner_executes_process_and_shell_commands(tmp_path):
    runner = LocalRunner(tmp_path)

    process_result = asyncio.run(
        runner.exec(
            ExecCommand.process(
                sys.executable,
                "-c",
                "import os,sys; print(os.getcwd()); print(sys.stdin.read())",
            ),
            stdin="hello",
        )
    )
    shell_result = asyncio.run(runner.exec(ExecCommand.bash("printf shell-ok")))

    assert process_result.exit_code == 0
    assert process_result.timed_out is False
    assert str(tmp_path) in process_result.stdout
    assert "hello" in process_result.stdout
    assert shell_result.stdout == "shell-ok"


def test_local_runner_restricts_cwd_to_root(tmp_path):
    runner = LocalRunner(tmp_path)

    root_result = asyncio.run(runner.exec(ExecCommand.process("pwd"), cwd=str(tmp_path)))
    assert root_result.stdout.strip() == str(tmp_path)

    work = tmp_path / "work"
    work.mkdir()
    child_result = asyncio.run(runner.exec(ExecCommand.process("pwd"), cwd=str(work)))
    assert child_result.stdout.strip() == str(work)

    with pytest.raises(ValueError, match="outside the runner root"):
        asyncio.run(runner.exec(ExecCommand.process("pwd"), cwd=str(tmp_path.parent)))

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "outside-link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the runner root"):
        asyncio.run(runner.exec(ExecCommand.process("pwd"), cwd=str(link)))

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(runner.exec(ExecCommand.process("pwd"), cwd="../"))

    with pytest.raises(FileNotFoundError):
        asyncio.run(runner.exec(ExecCommand.process("pwd"), cwd="missing"))

    file_path = tmp_path / "file.txt"
    file_path.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        asyncio.run(runner.exec(ExecCommand.process("pwd"), cwd="file.txt"))


def test_local_runner_supports_cwd_env_and_stdin(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    runner = LocalRunner(tmp_path)

    result = asyncio.run(
        runner.exec(
            ExecCommand.process(
                sys.executable,
                "-c",
                (
                    "import os,sys; "
                    "print(os.path.basename(os.getcwd())); "
                    "print(os.environ['LOCAL_RUNNER_TEST']); "
                    "print(sys.stdin.read())"
                ),
            ),
            cwd="work",
            env={"LOCAL_RUNNER_TEST": "ok"},
            stdin="input",
        )
    )

    assert result.exit_code == 0
    assert result.stdout.splitlines() == ["work", "ok", "input"]


def test_local_runner_captures_failure_and_timeout(tmp_path):
    runner = LocalRunner(tmp_path)

    missing = asyncio.run(runner.exec(ExecCommand.process("cayu-command-that-does-not-exist")))
    failed = asyncio.run(
        runner.exec(
            ExecCommand.process(
                sys.executable,
                "-c",
                "import sys; print('bad', file=sys.stderr); sys.exit(7)",
            )
        )
    )
    timed_out = asyncio.run(
        runner.exec(
            ExecCommand.process(
                sys.executable,
                "-c",
                "import time; time.sleep(5)",
            ),
            timeout_s=1,
        )
    )
    shell_timed_out = asyncio.run(
        runner.exec(
            ExecCommand.bash("sleep 5"),
            timeout_s=1,
        )
    )

    assert missing.exit_code == 127
    assert missing.stderr == "Command not found: cayu-command-that-does-not-exist"
    assert failed.exit_code == 7
    assert failed.stderr.strip() == "bad"
    assert timed_out.timed_out is True
    assert timed_out.exit_code != 0
    assert shell_timed_out.timed_out is True
    assert shell_timed_out.exit_code != 0


def test_local_runner_enforces_output_limit_while_process_runs(tmp_path):
    runner = LocalRunner(tmp_path)

    result = asyncio.run(
        runner.exec(
            ExecCommand.process(
                sys.executable,
                "-c",
                "import sys; sys.stdout.write('abcdef'); sys.stderr.write('uvwxyz')",
            ),
            output_limit_bytes=3,
        )
    )

    assert result.stdout == "abc"
    assert result.stderr == "uvw"
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True


def test_local_runner_kills_process_on_cancellation(tmp_path):
    started = tmp_path / "started.txt"
    marker = tmp_path / "marker.txt"
    runner = LocalRunner(tmp_path)

    async def run_and_cancel() -> None:
        task = asyncio.create_task(
            runner.exec(
                ExecCommand.process(
                    sys.executable,
                    "-c",
                    (
                        "import pathlib,time; "
                        "pathlib.Path('started.txt').write_text('started'); "
                        "time.sleep(2); "
                        "pathlib.Path('marker.txt').write_text('alive')"
                    ),
                )
            )
        )
        for _ in range(20):
            if started.exists():
                break
            await asyncio.sleep(0.05)
        assert started.exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_and_cancel())
    time.sleep(2.2)
    assert not marker.exists()


def test_local_runner_cleans_up_when_cancelled_during_timeout(tmp_path, monkeypatch):
    marker = tmp_path / "marker.txt"
    runner = LocalRunner(tmp_path)
    original_waiter = runner_subprocess_module._await_process_exit

    async def run_and_cancel() -> None:
        cleanup_started = asyncio.Event()

        async def delayed_waiter(wait_task):
            cleanup_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await original_waiter(wait_task)
                raise

        monkeypatch.setattr(runner_subprocess_module, "_await_process_exit", delayed_waiter)
        task = asyncio.create_task(
            runner.exec(
                ExecCommand.process(
                    sys.executable,
                    "-c",
                    (
                        "import pathlib,time; "
                        "time.sleep(2); "
                        "pathlib.Path('marker.txt').write_text('alive')"
                    ),
                ),
                timeout_s=1,
            )
        )
        await cleanup_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_and_cancel())
    time.sleep(1.2)
    assert not marker.exists()


def test_local_runner_is_fail_closed_on_host_env_inheritance(tmp_path, monkeypatch):
    monkeypatch.setenv("CAYU_LOCAL_RUNNER_SECRET", "secret")

    probe = ExecCommand.process(
        sys.executable,
        "-c",
        (
            "import os; "
            "print(os.environ.get('CAYU_LOCAL_RUNNER_SECRET', '')); "
            "print(bool(os.environ.get('PATH')))"
        ),
    )
    default = asyncio.run(LocalRunner(tmp_path).exec(probe))
    inherited = asyncio.run(LocalRunner(tmp_path, inherit_env=True).exec(probe))
    isolated = asyncio.run(LocalRunner(tmp_path, inherit_env=False).exec(probe))

    # Default is fail-closed: host secrets are absent, but the minimal safe
    # base env (PATH etc.) is forwarded so commands still resolve binaries.
    assert default.stdout.splitlines() == ["", "True"]
    assert inherited.stdout.splitlines() == ["secret", "True"]
    assert isolated.stdout.splitlines() == ["", "True"]


def test_local_runner_injects_and_redacts_declared_secret_env(tmp_path):
    runner = LocalRunner(
        tmp_path,
        secret_env=[SecretEnv(name="API_TOKEN", ref=SecretRef(name="api_token"))],
        secret_resolver=StaticVault({"api_token": "sk-local-secret-1"}),
    )

    result = asyncio.run(
        runner.exec(
            ExecCommand.process(
                sys.executable,
                "-c",
                (
                    "import os,sys; "
                    "print('token:', os.environ['API_TOKEN']); "
                    "print(os.environ['API_TOKEN'], file=sys.stderr)"
                ),
            )
        )
    )

    assert result.exit_code == 0
    # The secret reached the child process env but is scrubbed from output.
    assert result.stdout.strip() == f"token: {REDACTED_SECRET}"
    assert result.stderr.strip() == REDACTED_SECRET


def test_local_runner_rejects_env_key_colliding_with_secret_env(tmp_path):
    runner = LocalRunner(
        tmp_path,
        secret_env={"API_TOKEN": SecretRef(name="api_token")},
        secret_resolver=StaticVault({"api_token": "sk-local-secret-1"}),
    )

    with pytest.raises(ValueError, match="collides with declared secret_env"):
        asyncio.run(
            runner.exec(ExecCommand.process("env"), env={"API_TOKEN": "override"}),
        )

    with pytest.raises(ValueError, match="secret_resolver"):
        LocalRunner(tmp_path, secret_env={"API_TOKEN": SecretRef(name="api_token")})


def test_local_runner_rejects_invalid_inputs(tmp_path):
    runner = LocalRunner(tmp_path)

    with pytest.raises(FileNotFoundError):
        LocalRunner(tmp_path / "missing")

    with pytest.raises(TypeError, match="inherit_env"):
        LocalRunner(tmp_path, inherit_env="true")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="ExecCommand"):
        asyncio.run(runner.exec("echo bad"))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="dictionary"):
        asyncio.run(runner.exec(ExecCommand.process("env"), env=[]))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="keys"):
        asyncio.run(runner.exec(ExecCommand.process("env"), env={" ": "bad"}))

    with pytest.raises(ValueError, match="values"):
        asyncio.run(runner.exec(ExecCommand.process("env"), env={"X": 1}))  # type: ignore[dict-item]

    with pytest.raises(ValueError, match="greater than zero"):
        asyncio.run(runner.exec(ExecCommand.process("pwd"), timeout_s=0))

    with pytest.raises(TypeError, match="stdin"):
        asyncio.run(runner.exec(ExecCommand.process("cat"), stdin=b"bad"))  # type: ignore[arg-type]
