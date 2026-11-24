"""A tool runs another agent on a file through ToolContext.agents."""

from __future__ import annotations

import asyncio
import json

import pytest
from tests.runtime.runtime_attachment_support import _app_with_artifact_store, _valid_png_bytes

from cayu import AgentSpec, Message, RunRequest, SessionStatus
from cayu.artifacts.attachments import RESOLVED_FILE_ATTACHMENTS_OPTION
from cayu.artifacts.local import LocalArtifactStore
from cayu.context.structured_output import StructuredOutputSpec
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.evals import ScriptedModelProvider, scripted_structured_output
from cayu.messages import FilePart
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.sessions.outcomes import run_to_completion
from cayu.tools import AgentFile, Tool, ToolEffect, ToolResult, ToolSpec
from cayu.tools.agent_runs import AGENT_RUN_SESSION_PREFIX

_SPEC = StructuredOutputSpec(
    name="reading",
    json_schema={
        "type": "object",
        "properties": {"total": {"type": "string"}},
        "required": ["total"],
    },
)


class _ReadInvoice(Tool):
    spec = ToolSpec(name="read_invoice", input_schema={"type": "object"}, effect=ToolEffect.NONE)
    runs_agents = ("reader",)

    def __init__(self, image: bytes, *, runs: int = 1, agent: str = "reader") -> None:
        super().__init__()
        self._image = image
        self._runs = runs
        self._agent = agent
        self.outcomes = []
        self.child_session_ids = []

    async def run(self, ctx, args):
        del args
        assert ctx.agents is not None
        for _ in range(self._runs):
            outcome = await ctx.agents.run(
                self._agent,
                "Transcribe this invoice.",
                files=[AgentFile(self._image, filename="invoice.png", kind="image")],
                structured_output=_SPEC,
            )
            self.outcomes.append(outcome)
            self.child_session_ids.append(outcome.session_id)
        assert outcome.structured_output is not None
        return ToolResult(content=json.dumps(outcome.structured_output.output))


class _NoAgents(Tool):
    spec = ToolSpec(name="plain", input_schema={"type": "object"}, effect=ToolEffect.NONE)

    async def run(self, ctx, args):
        del args
        return ToolResult(content="none" if ctx.agents is None else "bound")


def _scenario(tmp_path, tool: Tool, tool_name: str, *, reader_environment: bool = False):
    reader_requests: list[ModelRequest] = []

    def reader(request: ModelRequest):
        reader_requests.append(request)
        return list(scripted_structured_output({"total": "12.50"}))

    async def scenario():
        app, _ = _app_with_artifact_store(tmp_path)
        app.register_provider(
            ScriptedModelProvider(
                events=[
                    [
                        ModelStreamEvent.tool_call(name=tool_name, arguments={}, id="c1"),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
                ],
                name="clerk",
            ),
            default=True,
        )
        app.register_provider(ScriptedModelProvider(response_factory=reader, name="reader"))
        app.register_agent(AgentSpec(name="clerk", model="model"), tools=[tool])
        reader_registration = {}
        if reader_environment:
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="vision"),
                    artifact_store=LocalArtifactStore(tmp_path / "vision", store_id="vision"),
                )
            )
            reader_registration["environment_name"] = "vision"
        app.register_agent(
            AgentSpec(name="reader", model="model", provider_name="reader"),
            **reader_registration,
        )
        outcome = await run_to_completion(
            app,
            RunRequest(
                agent_name="clerk", session_id="parent", messages=[Message.text("user", "go")]
            ),
        )
        children = [
            await app.session_store.load(sid) for sid in getattr(tool, "child_session_ids", [])
        ]
        return outcome, children

    outcome, children = asyncio.run(scenario())
    return outcome, children, reader_requests


def test_a_tool_runs_another_agent_on_a_file(tmp_path) -> None:
    tool = _ReadInvoice(_valid_png_bytes())

    outcome, children, reader_requests = _scenario(tmp_path, tool, "read_invoice")

    assert outcome.status is SessionStatus.COMPLETED, outcome.error
    (call,) = outcome.tool_calls
    assert call.disposition == "completed"
    assert call.result is not None
    assert json.loads(call.result.content) == {"total": "12.50"}
    (request,) = reader_requests
    user_parts = [part for message in request.messages for part in message.content]
    assert any(type(part) is FilePart for part in user_parts)
    assert request.options[RESOLVED_FILE_ATTACHMENTS_OPTION]
    (child,) = children
    assert child is not None
    assert child.id.startswith(AGENT_RUN_SESSION_PREFIX)
    assert child.agent_name == "reader"
    assert child.metadata["tool_agent_run"]["parent_session_id"] == "parent"
    assert child.causal_budget_id == "parent"


def test_files_go_to_the_environment_the_agent_runs_in(tmp_path) -> None:
    tool = _ReadInvoice(_valid_png_bytes())

    outcome, children, reader_requests = _scenario(
        tmp_path, tool, "read_invoice", reader_environment=True
    )

    assert outcome.status is SessionStatus.COMPLETED, outcome.error
    (request,) = reader_requests
    assert request.options[RESOLVED_FILE_ATTACHMENTS_OPTION]
    (child,) = children
    assert child is not None
    assert child.environment_name == "vision"


def test_running_again_in_the_same_call_returns_the_finished_run(tmp_path) -> None:
    tool = _ReadInvoice(_valid_png_bytes(), runs=2)

    outcome, _, reader_requests = _scenario(tmp_path, tool, "read_invoice")

    assert outcome.status is SessionStatus.COMPLETED, outcome.error
    assert len(reader_requests) == 1
    first, second = tool.outcomes
    assert first.session_id == second.session_id
    assert second.structured_output is not None
    assert second.structured_output.output == {"total": "12.50"}


def test_a_tool_may_run_only_the_agents_it_names(tmp_path) -> None:
    tool = _ReadInvoice(_valid_png_bytes(), agent="clerk")

    outcome, _, reader_requests = _scenario(tmp_path, tool, "read_invoice")

    (call,) = outcome.tool_calls
    assert call.disposition == "failed"
    assert reader_requests == []


def test_a_tool_that_names_no_agents_gets_no_handle(tmp_path) -> None:
    outcome, _, _ = _scenario(tmp_path, _NoAgents(), "plain")

    (call,) = outcome.tool_calls
    assert call.result is not None
    assert call.result.content == "none"


def test_runs_agents_must_be_a_tuple_of_names() -> None:
    from cayu.tools.agent_runs import declared_agent_runs

    class _Bad(_NoAgents):
        runs_agents = "reader"  # ty: ignore[invalid-assignment]

    with pytest.raises(TypeError):
        declared_agent_runs(_Bad())
    assert declared_agent_runs(_ReadInvoice(b"")) == frozenset({"reader"})
