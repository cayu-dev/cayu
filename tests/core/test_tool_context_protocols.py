"""Typed ToolContext handles, isinstance seams, and the non-async run() convention."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest
from pydantic import ValidationError

from cayu import (
    Agent,
    AgentSpec,
    ArtifactStoreHandle,
    CredentialProxyHandle,
    Event,
    EventType,
    KnowledgeStoreHandle,
    Message,
    RunnerHandle,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
    VaultHandle,
    Workflow,
    WorkflowSpec,
    WorkspaceHandle,
)
from cayu.artifacts import LocalArtifactStore
from cayu.environments import (
    Environment,
    EnvironmentSpec,
    copy_environment,
)
from cayu.environments.factory import EnvironmentFactoryResult
from cayu.proxies import AllowlistProxy, PassthroughProxy
from cayu.runners import LocalRunner
from cayu.vaults import LocalEnvVault, StaticVault
from cayu.workspaces import LocalWorkspace


def test_concrete_implementations_satisfy_tool_context_protocols(tmp_path) -> None:
    assert isinstance(LocalWorkspace(tmp_path, workspace_id="ws"), WorkspaceHandle)
    assert isinstance(
        LocalArtifactStore(tmp_path / "artifacts", store_id="store"),
        ArtifactStoreHandle,
    )
    assert isinstance(LocalRunner(tmp_path), RunnerHandle)
    assert isinstance(StaticVault({}), VaultHandle)
    assert isinstance(LocalEnvVault({}), VaultHandle)
    assert isinstance(PassthroughProxy(StaticVault({})), CredentialProxyHandle)
    assert isinstance(
        AllowlistProxy(StaticVault({}), allowed_destinations=("api.example.com",)),
        CredentialProxyHandle,
    )


def test_tool_context_accepts_structural_handles(tmp_path) -> None:
    class DuckWorkspace:
        async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> Any:
            return b""

        async def write_bytes(self, path: str, content: bytes) -> None:
            return None

        async def delete(self, path: str) -> None:
            return None

        async def list(self, pattern: str = "**/*", *, limit: int | None = None) -> Any:
            return []

    ctx = ToolContext(
        session_id="sess_1",
        workspace=DuckWorkspace(),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts", store_id="store"),
        runner=LocalRunner(tmp_path),
        vault=StaticVault({}),
        proxy=PassthroughProxy(StaticVault({})),
    )
    assert isinstance(ctx.workspace, WorkspaceHandle)


def test_tool_context_rejects_non_conforming_handles() -> None:
    class Empty:
        pass

    for field, protocol_name in (
        ("workspace", "WorkspaceHandle"),
        ("artifact_store", "ArtifactStoreHandle"),
        ("runner", "RunnerHandle"),
        ("vault", "VaultHandle"),
        ("proxy", "CredentialProxyHandle"),
        ("knowledge_store", "KnowledgeStoreHandle"),
    ):
        with pytest.raises(ValidationError, match=protocol_name):
            ToolContext(session_id="sess_1", **{field: Empty()})


def test_knowledge_store_handle_is_the_read_surface() -> None:
    class ReadOnlyStore:
        async def search(self, *args: Any, **kwargs: Any) -> Any:
            return []

        async def list_entries(self, *args: Any, **kwargs: Any) -> Any:
            return []

        async def read_chunks(self, *args: Any, **kwargs: Any) -> Any:
            return []

    assert isinstance(ReadOnlyStore(), KnowledgeStoreHandle)
    ctx = ToolContext(session_id="sess_1", knowledge_store=ReadOnlyStore())
    assert ctx.knowledge_store is not None


def test_tool_accepts_tool_spec_subclass() -> None:
    class ExtendedToolSpec(ToolSpec):
        pass

    class SubclassSpecTool(Tool):
        spec = ExtendedToolSpec(name="subclass_tool")

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content="ok")

    tool = SubclassSpecTool()
    assert tool.name == "subclass_tool"

    explicit = SubclassSpecTool(ExtendedToolSpec(name="explicit_tool"))
    assert explicit.name == "explicit_tool"


def test_copy_environment_preserves_environment_subclass() -> None:
    class TracedEnvironment(Environment):
        pass

    environment = TracedEnvironment(EnvironmentSpec(name="traced"))
    copied = copy_environment(environment)
    assert type(copied) is TracedEnvironment
    assert copied is not environment
    assert copied.spec.name == "traced"

    result = EnvironmentFactoryResult(environment=environment)
    assert type(result.environment) is TracedEnvironment


def test_copy_environment_still_rejects_non_environments() -> None:
    class EnvironmentLike:
        spec = EnvironmentSpec(name="fake")

    with pytest.raises(TypeError, match="Environment"):
        copy_environment(EnvironmentLike())  # type: ignore[arg-type]


def test_agent_and_workflow_run_use_single_calling_convention() -> None:
    class StreamingAgent(Agent):
        spec = AgentSpec(name="streamer", model="fake-model")
        tools: list[Tool] = []

        async def run(self, messages: list[Message]):
            yield Event(type=EventType.SESSION_STARTED, session_id="sess_1")

    class StreamingWorkflow(Workflow):
        spec = WorkflowSpec(name="flow")

        async def run(self, session_id: str):
            yield Event(type=EventType.SESSION_STARTED, session_id=session_id)

    # The abstract methods are plain callables returning an AsyncIterator, so
    # implementations are async generators consumed without an extra await.
    assert not inspect.iscoroutinefunction(Agent.run)
    assert not inspect.iscoroutinefunction(Workflow.run)

    async def consume() -> list[Event]:
        events: list[Event] = []
        async for event in StreamingAgent().run([Message.text("user", "hi")]):
            events.append(event)
        async for event in StreamingWorkflow().run("sess_1"):
            events.append(event)
        return events

    events = asyncio.run(consume())
    assert [event.session_id for event in events] == ["sess_1", "sess_1"]
