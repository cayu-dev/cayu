from __future__ import annotations

import asyncio

import pytest
from tests.core.test_builtin_tools import TINY_PNG_BYTES, AttachmentTool
from tests.core.test_foreground_subagent_recovery import _app, _identity, _Provider

from cayu import (
    ArtifactScope,
    Environment,
    EnvironmentSpec,
    InMemorySessionStore,
    LocalArtifactStore,
    Message,
    ResumeRequest,
    RunRequest,
    SessionQuery,
    SQLiteSessionStore,
    SubagentTool,
)
from cayu.core import ToolResultPart
from cayu.providers import ModelStreamEvent


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_foreground_recovery_does_not_promote_child_attachments(backend, tmp_path, monkeypatch):
    async def scenario():
        database = tmp_path / "sessions.sqlite"
        sessions = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(database)
        artifacts = LocalArtifactStore(tmp_path / "artifacts", store_id="child-attachments")
        artifact = await artifacts.put_bytes(
            TINY_PNG_BYTES,
            filename="invoice.png",
            content_type="image/png",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="local",
        )
        provider = _Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "child", "task": "answer"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(id="attach", name="attach_file", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("image reviewed"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("parent done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )

        def build_app():
            tool = AttachmentTool(artifact.id, artifact.size_bytes)
            tool.spec = tool.spec.model_copy(
                update={"execution_profile_identity": _identity("child-attachment")}
            )
            app = _app(sessions, provider, child_tools=[tool])
            app.register_environment(
                Environment(
                    EnvironmentSpec(
                        name="local", execution_profile_identity=_identity("attachment-environment")
                    ),
                    artifact_store=LocalArtifactStore(
                        tmp_path / "artifacts", store_id="child-attachments"
                    ),
                ),
                default=True,
            )
            return app

        original = SubagentTool.run
        selected = []

        async def lose_return(tool, context, arguments):
            selected.append(await original(tool, context, arguments))
            raise ConnectionError("parent lost child result with attachment history")

        try:
            with monkeypatch.context() as patch:
                patch.setattr(SubagentTool, "run", lose_return)
                initial = [
                    event
                    async for event in build_app().run(
                        RunRequest(
                            session_id="parent",
                            agent_name="parent",
                            messages=[Message.text("user", "go")],
                        )
                    )
                ]
            assert initial[-1].type == "session.interrupted"
            assert len(selected) == 1 and not selected[0].is_error, selected
            assert not selected[0].artifacts
            child = (
                await sessions.list_sessions(SessionQuery(parent_session_id="parent"))
            ).sessions[0]
            child_transcript = await sessions.load_transcript(child.id)
            child_results = [
                part
                for message in child_transcript
                for part in message.content
                if isinstance(part, ToolResultPart)
            ]
            assert len(child_results) == 1
            assert child_results[0].artifacts[0]["artifact_id"] == artifact.id
            assert "cayu_file_attachments" in provider.requests[2].options
            if isinstance(sessions, SQLiteSessionStore):
                await sessions.close()
                sessions = SQLiteSessionStore(database)
            resumed = [
                event
                async for event in build_app().resume(
                    ResumeRequest(session_id="parent", messages=[Message.text("user", "continue")])
                )
            ]
            assert resumed[-1].type == "session.completed"
            assert len(provider.requests) == 4
            assert await sessions.load_transcript(child.id) == child_transcript
            terminals = [
                event
                for event in await sessions.load_events("parent")
                if event.type in {"tool.call.completed", "tool.call.failed"}
            ]
            assert len(terminals) == 1
            assert terminals[0].payload["result"] == selected[0].model_dump(mode="json")
            parent_results = [
                part
                for message in await sessions.load_transcript("parent")
                for part in message.content
                if isinstance(part, ToolResultPart)
            ]
            assert len(parent_results) == 1 and not parent_results[0].artifacts
            assert not provider.requests[-1].options.get("cayu_file_attachments")
        finally:
            if isinstance(sessions, SQLiteSessionStore):
                await sessions.close()

    asyncio.run(scenario())
