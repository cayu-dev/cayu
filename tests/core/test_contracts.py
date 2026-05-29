from __future__ import annotations

import pytest
import asyncio

from cayu.core import Event, EventType, Message, MessageRole, TextPart
from pydantic import SecretStr, ValidationError

from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.mcp import McpServerSpec
from cayu.runners import ExecCommand
from cayu.vaults import ResolvedSecret, SecretRef
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
        payload={"ok": True},
    )

    assert event.type == EventType.SESSION_STARTED
    assert event.session_id == "sess_1"
    assert event.agent_name == "orchestrator"
    assert event.payload == {"ok": True}
    assert event.id
    assert event.timestamp


def test_event_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Event(type=EventType.SESSION_STARTED, session_id="sess_1", typo=True)  # type: ignore[call-arg]


def test_event_requires_namespace_for_custom_types():
    event = Event(type="custom.pm.status.changed", session_id="sess_1")

    assert event.type == "custom.pm.status.changed"

    with pytest.raises(ValidationError, match="custom"):
        Event(type="model.starteddd", session_id="sess_1")


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


def test_in_memory_event_sink_collects_events():
    sink = InMemoryEventSink()
    event = Event(type=EventType.SESSION_STARTED, session_id="sess_1")

    asyncio.run(sink.emit(event))

    assert sink.events == [event]


def test_run_request_accepts_messages_and_metadata():
    request = RunRequest(
        agent_name="orchestrator",
        messages=[Message.text("user", "start")],
        metadata={"source": "test"},
    )

    assert request.agent_name == "orchestrator"
    assert request.messages[0].content[0].text == "start"
    assert request.metadata == {"source": "test"}


def test_message_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Message(role="user", content=[], unexpected=True)  # type: ignore[call-arg]


def test_run_request_session_id_is_explicit_resume_or_idempotency_hint():
    request = RunRequest(
        agent_name="orchestrator",
        messages=[Message.text("user", "resume")],
        session_id="sess_existing",
    )

    assert request.session_id == "sess_existing"


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
