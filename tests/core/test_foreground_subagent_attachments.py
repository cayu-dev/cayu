from __future__ import annotations

import asyncio
from base64 import b64decode
from hashlib import sha256

import pytest
from tests.core.test_builtin_tools import TINY_PNG_BYTES, AttachmentTool
from tests.core.test_foreground_subagent_recovery import _app, _identity, _Provider

from cayu.approvals.user_input import UserInputResponse
from cayu.artifacts.base import ArtifactScope
from cayu.artifacts.local import LocalArtifactStore
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.messages import Message, ToolResultPart
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.runtime.execution_profiles import ExecutionProfileMismatchError
from cayu.sessions.base import InMemorySessionStore, ResumeRequest, RunRequest, SessionQuery
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.subagents import SubagentTool
from cayu.tools.user_input import UserInputTool


def _assert_child_attachment(request: ModelRequest, artifact_id: str) -> None:
    resolved = request.options["cayu_file_attachments"]
    assert set(resolved) == {artifact_id}
    attachment = resolved[artifact_id]
    assert attachment["artifact_id"] == artifact_id
    assert attachment["kind"] == "image"
    assert attachment["filename"] == "invoice.png"
    assert attachment["content_type"] == "image/png"
    assert attachment["metadata"] == {}
    assert b64decode(attachment["data_base64"], validate=True) == TINY_PNG_BYTES
    assert attachment["content_sha256"] == sha256(TINY_PNG_BYTES).hexdigest()


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
            _assert_child_attachment(provider.requests[2], artifact.id)
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


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_nested_input_reconstruction_preserves_environment_and_attachment_scope(tmp_path, backend):
    async def scenario():
        database = tmp_path / "paused-attachments.sqlite"
        sessions = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(database)
        artifacts = LocalArtifactStore(tmp_path / "artifacts", store_id="paused-child-artifacts")
        artifact = await artifacts.put_bytes(
            TINY_PNG_BYTES,
            filename="invoice.png",
            content_type="image/png",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="local",
        )
        opening = _Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "child", "task": "inspect"}
                    ),
                    ModelStreamEvent.completed(),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="ask", name="ask_user", arguments={"question": "Continue?"}
                    ),
                    ModelStreamEvent.completed(),
                ],
            ]
        )

        def build(provider, *, changed=False):
            tool = AttachmentTool(artifact.id, artifact.size_bytes)
            tool.spec = tool.spec.model_copy(
                update={"execution_profile_identity": _identity("paused-attachment")}
            )
            app = _app(sessions, provider, child_tools=[tool, UserInputTool()])
            app.register_environment(
                Environment(
                    EnvironmentSpec(
                        name="local",
                        execution_profile_identity=_identity(
                            "changed-env" if changed else "paused-env"
                        ),
                    ),
                    artifact_store=LocalArtifactStore(
                        tmp_path / "artifacts", store_id="paused-child-artifacts"
                    ),
                ),
                default=True,
            )
            return app

        try:
            app = build(opening)
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="parent",
                        agent_name="parent",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            child = (
                await sessions.list_sessions(SessionQuery(parent_session_id="parent"))
            ).sessions[0]
            pending = next(
                event
                for event in await sessions.load_events(child.id)
                if event.type == "session.awaiting_user_input"
            )
            response = UserInputResponse(
                session_id=child.id, input_id=pending.payload["input_id"], answer="yes"
            )
            assert await app.drain_background_interruptions(timeout_s=10)
            if backend == "sqlite":
                await sessions.close()
                sessions = SQLiteSessionStore(database)
            checkpoint = await sessions.load_checkpoint(child.id)
            transcript = await sessions.load_transcript(child.id)
            rejected_provider = _Provider([])
            with pytest.raises(ExecutionProfileMismatchError):
                _ = [
                    event
                    async for event in build(rejected_provider, changed=True).resolve_user_input(
                        response
                    )
                ]
            assert rejected_provider.requests == []
            assert await sessions.load_checkpoint(child.id) == checkpoint
            assert await sessions.load_transcript(child.id) == transcript
            provider = _Provider(
                [
                    [
                        ModelStreamEvent.tool_call(id="attach", name="attach_file", arguments={}),
                        ModelStreamEvent.completed(),
                    ],
                    [
                        ModelStreamEvent.text_delta("image reviewed 雪"),
                        ModelStreamEvent.completed(),
                    ],
                    [ModelStreamEvent.text_delta("parent complete"), ModelStreamEvent.completed()],
                ]
            )
            app = build(provider)
            _ = [event async for event in app.resolve_user_input(response)]
            assert await app.drain_background_interruptions(timeout_s=10)
            child_results = [
                part
                for message in await sessions.load_transcript(child.id)
                for part in message.content
                if isinstance(part, ToolResultPart)
            ]
            attached = [part for part in child_results if part.artifacts]
            assert len(attached) == 1 and attached[0].artifacts[0]["artifact_id"] == artifact.id
            _assert_child_attachment(provider.requests[1], artifact.id)
            parent_results = [
                part
                for message in await sessions.load_transcript("parent")
                for part in message.content
                if isinstance(part, ToolResultPart)
            ]
            assert len(parent_results) == 1 and not parent_results[0].artifacts
            assert not provider.requests[-1].options.get("cayu_file_attachments")
            assert len(provider.requests) == 3
        finally:
            if backend == "sqlite":
                await sessions.close()

    asyncio.run(scenario())
