from __future__ import annotations

import asyncio
import io
from copy import deepcopy

import pytest
from PIL import Image
from pypdf import PdfWriter
from tests.core._execution_profile_fixtures import versioned_test_provider_identity
from tests.core.test_builtin_tools import FakeProvider

from cayu import ArtifactScope, Environment, EnvironmentSpec, LocalArtifactStore, file_attachment
from cayu.core import AgentSpec, EventType, Message
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.messages import FilePart
from cayu.core.tools import ToolContext
from cayu.providers import ModelStreamEvent
from cayu.runtime import CayuApp, ResumeRequest, RunRequest
from cayu.runtime._model_step_executor import _file_attachment_refs
from cayu.runtime.context import DefaultContextPolicy
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.files import ReadFileTool


class ReconstructableProvider(FakeProvider):
    @property
    def execution_profile_identity(self):
        return versioned_test_provider_identity(self)


def _attachment():
    return file_attachment(
        artifact_id="art_" + "1" * 32,
        kind="image",
        filename="fixture.jpg",
        content_type="image/jpeg",
        size_bytes=83606,
        metadata={"source_artifact_id": "art_" + "2" * 32},
    )


def _tool_message(attachment, call="read"):
    return Message.tool_result(
        tool_call_id=call, tool_name="read_file", content="attached", artifacts=[attachment]
    )


@pytest.mark.parametrize("prompt", [False, True])
@pytest.mark.parametrize("source", [None, "art_" + "2" * 32, "art_" + "3" * 32])
def test_compatible_occurrences_retain_provenance(prompt, source):
    first = _attachment()
    second = deepcopy(first)
    if source is None:
        second["metadata"].pop("source_artifact_id")
    else:
        second["metadata"]["source_artifact_id"] = source
    messages = [
        Message(role="user", content=[FilePart(attachment=first)])
        if prompt
        else _tool_message(first),
        _tool_message(second, "read_again"),
    ]
    # Exercise the same JSON representation used for retained Message records.
    restored = [Message.model_validate_json(message.model_dump_json()) for message in messages]
    refs, prompt_ids, tool_ids = _file_attachment_refs(restored)
    assert [ref.model_dump(mode="json") for ref in refs] == [first, second]
    assert prompt_ids == ({first["artifact_id"]} if prompt else set())
    assert tool_ids == {first["artifact_id"]}


@pytest.mark.parametrize("prompt", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        {"filename": "other.jpg"},
        {"size_bytes": 42},
        {"content_type": "image/png"},
        {"kind": "document", "content_type": "application/pdf"},
        {"metadata": {"pages": "2"}},
        {"metadata": {"content_sha256": "a" * 64}},
        {"metadata": {"browser_visual_screenshot_sha256": "a" * 64}},
        {"metadata": {"resolution": "high"}},
    ],
)
def test_incompatible_occurrences_fail_closed(prompt, change):
    first = _attachment()
    second = {**deepcopy(first), **change}
    with pytest.raises(RuntimeError, match=f"Conflicting file attachment.*{first['artifact_id']}"):
        _file_attachment_refs(
            [
                Message(role="user", content=[FilePart(attachment=first)])
                if prompt
                else _tool_message(first),
                _tool_message(second, "read_again"),
            ]
        )


@pytest.mark.parametrize("kind", ["image", "document"])
def test_native_derived_reuse_assembles_and_survives_sqlite_restore(tmp_path, kind):
    buffer = io.BytesIO()
    if kind == "image":
        # An uncompressed PNG forces the real Pillow resize/encoding path.
        Image.new("RGB", (256, 256), "purple").save(buffer, format="PNG", compress_level=0)
        content_type, filename = "image/png", "fixture.png"
        options = {"max_attachment_bytes": 4096}
    else:
        writer = PdfWriter()
        for _ in range(3):
            writer.add_blank_page(width=100, height=100)
        writer.write(buffer)
        content_type, filename = "application/pdf", "fixture.pdf"
        options = {"pages": "1"}

    async def scenario():
        root = tmp_path / "artifacts"
        store = LocalArtifactStore(root, store_id="artifacts")
        sources = [
            await store.put_bytes(
                buffer.getvalue(),
                filename=filename,
                content_type=content_type,
                session_id="reuse",
                agent_name="assistant",
                environment_name="local",
            )
            for _ in range(2)
        ]
        assert sources[0].id != sources[1].id
        provider = ReconstructableProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id=f"read_{index}",
                        name="read_file",
                        arguments={"artifact_id": source.id, **options},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
                for index, source in enumerate(sources)
            ]
            + [[ModelStreamEvent.completed({"finish_reason": "stop"})]]
        )
        db_path = tmp_path / "sessions.sqlite"
        sessions = SQLiteSessionStore(db_path)

        def make_app(session_store, artifacts, model_provider):
            app = CayuApp(
                session_store=session_store,
                enable_logging=False,
            )
            app.register_provider(model_provider, default=True)
            app.register_environment(
                Environment(
                    EnvironmentSpec(
                        name="local",
                        execution_profile_identity=ExecutionProfileBehaviorIdentity(
                            name="tests:attachment-provenance",
                            behavior_version="1",
                            implementation_version="test-v1",
                        ),
                    ),
                    artifact_store=artifacts,
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[ReadFileTool()],
                context_policy=DefaultContextPolicy(max_attachment_results=2),
            )
            return app

        def assert_request(request):
            refs, _, _ = _file_attachment_refs(request.messages)
            assert len(refs) == 2
            assert refs[0].artifact_id == refs[1].artifact_id
            assert refs[0].artifact_id not in {source.id for source in sources}
            assert [ref.metadata["source_artifact_id"] for ref in refs] == [s.id for s in sources]
            resolved = request.options["cayu_file_attachments"]
            assert list(resolved) == [refs[0].artifact_id]
            assert resolved[refs[0].artifact_id]["metadata"] == refs[0].metadata
            assert resolved[refs[0].artifact_id]["data_base64"]
            if kind == "document":
                assert all(ref.metadata["pages"] == "1" for ref in refs)
            return refs

        try:
            app = make_app(sessions, store, provider)
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="reuse",
                        agent_name="assistant",
                        messages=[Message.text("user", "read both")],
                    )
                )
            ]
            assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
            assert len(provider.requests) == 3
            refs = assert_request(provider.requests[-1])
            stored_refs, _, _ = _file_attachment_refs(await sessions.load_transcript("reuse"))
            assert stored_refs == refs
            listing = await store.list(scope=ArtifactScope.SESSION, session_id="reuse")
            assert listing.total_count == 3
        finally:
            await sessions.close()

        restored_sessions = SQLiteSessionStore(db_path)
        try:
            restored_store = LocalArtifactStore(root, store_id="artifacts")
            restored_provider = ReconstructableProvider(
                [ModelStreamEvent.completed({"finish_reason": "stop"})]
            )
            restored_app = make_app(restored_sessions, restored_store, restored_provider)
            events = [
                event
                async for event in restored_app.resume(
                    ResumeRequest(
                        session_id="reuse",
                        messages=[Message.text("user", "compare the retained reads")],
                    )
                )
            ]
            assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
            assert_request(restored_provider.requests[0])
        finally:
            await restored_sessions.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("scope", [ArtifactScope.SESSION, ArtifactScope.ENVIRONMENT])
def test_native_reader_rejects_unauthorized_source(tmp_path, scope):
    async def scenario():
        store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
        buffer = io.BytesIO()
        Image.new("RGB", (256, 256)).save(buffer, format="PNG", compress_level=0)
        source = await store.put_bytes(
            buffer.getvalue(),
            filename="fixture.png",
            content_type="image/png",
            scope=scope,
            session_id="other",
            environment_name="other",
            agent_name="other",
        )
        result = await ReadFileTool().run(
            ToolContext(
                session_id="reuse",
                environment_name="local",
                agent_name="assistant",
                artifact_store=store,
            ),
            {"artifact_id": source.id, "max_attachment_bytes": 4096},
        )
        assert result.is_error
        assert not result.artifacts
        assert "not available" in result.content

    asyncio.run(scenario())


@pytest.mark.parametrize("scope", [ArtifactScope.SESSION, ArtifactScope.ENVIRONMENT])
@pytest.mark.parametrize("prompt", [False, True])
def test_compatible_provenance_cannot_authorize_unavailable_derived_artifact(
    tmp_path, scope, prompt
):
    from tests._session_provenance import fixture_session_invocation

    from cayu.runtime import Session
    from cayu.runtime._model_step_executor import (
        _FileAttachmentUnavailable,
        _resolved_file_attachments,
    )

    async def scenario():
        store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
        derived = await store.put_bytes(
            b"derived",
            filename="fixture.jpg",
            content_type="image/jpeg",
            scope=scope,
            session_id="other",
            agent_name="other",
            environment_name="other",
        )
        first = {**_attachment(), "artifact_id": derived.id, "size_bytes": derived.size_bytes}
        second = deepcopy(first)
        second["metadata"]["source_artifact_id"] = "art_" + "3" * 32
        messages = [
            Message(role="user", content=[FilePart(attachment=first)])
            if prompt
            else _tool_message(first),
            _tool_message(second, "read_again"),
        ]
        app = CayuApp(enable_logging=False)
        app.register_environment(
            Environment(
                EnvironmentSpec(
                    name="local",
                    execution_profile_identity=ExecutionProfileBehaviorIdentity(
                        name="tests:attachment-provenance",
                        behavior_version="1",
                        implementation_version="test-v1",
                    ),
                ),
                artifact_store=store,
            ),
            default=True,
        )
        session = Session(
            id="reuse",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            causal_budget_id="reuse",
            invocation=fixture_session_invocation("reuse"),
        )
        with pytest.raises(_FileAttachmentUnavailable, match="not available"):
            await _resolved_file_attachments(
                messages=messages,
                session=session,
                registered_environment=app._environments["local"],
                max_file_attachment_bytes=100,
                max_total_file_attachment_bytes=200,
                max_file_attachments_per_request=2,
            )

    asyncio.run(scenario())
