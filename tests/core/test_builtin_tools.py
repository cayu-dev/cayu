from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator

import pytest

import cayu.tools.files as files_module
from cayu import (
    DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
    DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST,
    DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES,
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    ArtifactMetadata,
    ArtifactReadResult,
    ArtifactScope,
    ArtifactStore,
    Environment,
    EnvironmentSpec,
    file_attachment,
)
from cayu.artifacts import LocalArtifactStore
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.runners import ExecCommand, ExecResult, LocalRunner, Runner
from cayu.runtime import CayuApp, RunRequest
from cayu.tools import ExecCommandTool
from cayu.tools.commands import (
    DEFAULT_OUTPUT_LIMIT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_OUTPUT_LIMIT_BYTES,
    MAX_TIMEOUT_SECONDS,
)
from cayu.tools.files import (
    DEFAULT_ATTACHMENT_LIMIT_BYTES,
    DEFAULT_LIST_LIMIT,
    DEFAULT_MAX_ATTACHMENT_LIMIT_BYTES,
    DEFAULT_READ_LIMIT_BYTES,
    DEFAULT_WRITE_LIMIT_BYTES,
    MAX_LIST_LIMIT,
    MAX_READ_LIMIT_BYTES,
    MAX_WRITE_LIMIT_BYTES,
    ArtifactReadRequest,
    ListArtifactsTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from cayu.workspaces import LocalWorkspace

TINY_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00"
    b"\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(
        self,
        events: list[ModelStreamEvent] | list[list[ModelStreamEvent]],
    ) -> None:
        if events and isinstance(events[0], list):
            self.event_batches = events  # type: ignore[assignment]
        else:
            self.event_batches = [events]  # type: ignore[list-item]
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self.event_batches[len(self.requests) - 1]:
            yield event


class ContextRecordingTool(Tool):
    spec = ToolSpec(
        name="record_context",
        description="Record runtime tool context.",
        input_schema={"type": "object"},
    )

    def __init__(self) -> None:
        super().__init__()
        self.context: ToolContext | None = None

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.context = ctx
        return ToolResult(content="recorded")


class AttachmentTool(Tool):
    spec = ToolSpec(
        name="attach_file",
        description="Return a file attachment reference.",
        input_schema={"type": "object"},
    )

    def __init__(self, artifact_id: str, size_bytes: int) -> None:
        super().__init__()
        self.artifact_id = artifact_id
        self.size_bytes = size_bytes

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content="Attached image for inspection.",
            artifacts=[
                file_attachment(
                    artifact_id=self.artifact_id,
                    kind="image",
                    filename="invoice.png",
                    content_type="image/png",
                    size_bytes=self.size_bytes,
                )
            ],
        )


class ConflictingAttachmentsTool(Tool):
    spec = ToolSpec(
        name="conflicting_attachments",
        description="Return conflicting file attachment references.",
        input_schema={"type": "object"},
    )

    def __init__(self, artifact_id: str, size_bytes: int) -> None:
        super().__init__()
        self.artifact_id = artifact_id
        self.size_bytes = size_bytes

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content="Attached conflicting files for inspection.",
            artifacts=[
                file_attachment(
                    artifact_id=self.artifact_id,
                    kind="image",
                    filename="invoice.png",
                    content_type="image/png",
                    size_bytes=self.size_bytes,
                ),
                file_attachment(
                    artifact_id=self.artifact_id,
                    kind="document",
                    filename="invoice.pdf",
                    content_type="application/pdf",
                    size_bytes=self.size_bytes,
                ),
            ],
        )


class MultipleAttachmentsTool(Tool):
    spec = ToolSpec(
        name="multiple_attachments",
        description="Return multiple file attachment references.",
        input_schema={"type": "object"},
    )

    def __init__(self, attachments: list[tuple[str, int]]) -> None:
        super().__init__()
        self.attachments = attachments

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content="Attached multiple files for inspection.",
            artifacts=[
                file_attachment(
                    artifact_id=artifact_id,
                    kind="image",
                    filename=f"{artifact_id}.png",
                    content_type="image/png",
                    size_bytes=size_bytes,
                )
                for artifact_id, size_bytes in self.attachments
            ],
        )


class DuplicateAttachmentReferencesTool(Tool):
    spec = ToolSpec(
        name="duplicate_attachments",
        description="Return repeated references to the same file attachment.",
        input_schema={"type": "object"},
    )

    def __init__(self, artifact_id: str, size_bytes: int, *, count: int) -> None:
        super().__init__()
        self.artifact_id = artifact_id
        self.size_bytes = size_bytes
        self.count = count

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content="Attached repeated files for inspection.",
            artifacts=[
                file_attachment(
                    artifact_id=self.artifact_id,
                    kind="image",
                    filename="invoice.png",
                    content_type="image/png",
                    size_bytes=self.size_bytes,
                )
                for _ in range(self.count)
            ],
        )


class SequencedAttachmentTool(Tool):
    spec = ToolSpec(
        name="sequenced_attachment",
        description="Return attachment references in sequence.",
        input_schema={"type": "object"},
    )

    def __init__(self, attachments: list[tuple[str, int]]) -> None:
        super().__init__()
        self.attachments = attachments
        self.index = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        artifact_id, size_bytes = self.attachments[self.index]
        self.index += 1
        return ToolResult(
            content=f"Attached file {artifact_id}.",
            artifacts=[
                file_attachment(
                    artifact_id=artifact_id,
                    kind="image",
                    filename=f"{artifact_id}.png",
                    content_type="image/png",
                    size_bytes=size_bytes,
                )
            ],
        )


class SyntheticArtifactStore(ArtifactStore):
    id = "synthetic"

    def __init__(self, *, artifact_id: str, size_bytes: int) -> None:
        self.artifact_id = artifact_id
        self.size_bytes = size_bytes
        self.read_limits: list[int | None] = []

    async def put_bytes(self, content: bytes, **kwargs):
        raise NotImplementedError

    async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
        if artifact_id != self.artifact_id:
            raise FileNotFoundError(artifact_id)
        self.read_limits.append(max_bytes)
        return ArtifactReadResult(
            metadata=ArtifactMetadata(
                id=artifact_id,
                filename="large.png",
                content_type="image/png",
                size_bytes=self.size_bytes,
                session_id="sess_attachments",
            ),
            content=b"x",
            total_bytes=1,
            truncated=False,
        )

    async def list(self, **kwargs):
        raise NotImplementedError

    async def delete(self, artifact_id: str) -> None:
        raise NotImplementedError


class CustomPdfReader:
    def can_read(self, artifact) -> bool:
        return artifact.content_type == "application/pdf"

    async def read(self, request: ArtifactReadRequest) -> ToolResult:
        return ToolResult(
            content=f"custom pdf reader: {request.artifact.filename}",
            structured={
                **request.structured,
                "reader": "custom",
            },
        )


class RecordingRunner(Runner):
    def __init__(self, result: ExecResult | None = None) -> None:
        self.result = result or ExecResult(stdout="ok\n")
        self.timeout_s: int | None = None

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = None,
    ) -> ExecResult:
        self.timeout_s = timeout_s
        return self.result


def test_tool_context_carries_services_without_serializing_them(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    runner = LocalRunner(tmp_path)
    ctx = ToolContext(
        session_id="sess_1",
        agent_name="assistant",
        environment_name="local-dev",
        workspace_id="local",
        artifact_store_id="artifacts",
        workspace=workspace,
        artifact_store=artifact_store,
        runner=runner,
        mcp_servers=[object()],
    )

    dumped = ctx.model_dump()

    assert ctx.workspace is workspace
    assert ctx.artifact_store is artifact_store
    assert ctx.runner is runner
    assert ctx.mcp_servers
    assert dumped == {
        "session_id": "sess_1",
        "agent_name": "assistant",
        "environment_name": "local-dev",
        "workspace_id": "local",
        "artifact_store_id": "artifacts",
        "metadata": {},
    }


def test_builtin_tool_limits_are_model_context_sized():
    assert DEFAULT_READ_LIMIT_BYTES == 256 * 1024
    assert MAX_READ_LIMIT_BYTES == 4 * 1024 * 1024
    assert DEFAULT_MAX_FILE_ATTACHMENT_BYTES == 8 * 1024 * 1024
    assert DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES == 32 * 1024 * 1024
    assert DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST == 20
    assert DEFAULT_ATTACHMENT_LIMIT_BYTES == DEFAULT_MAX_FILE_ATTACHMENT_BYTES
    assert DEFAULT_MAX_ATTACHMENT_LIMIT_BYTES == DEFAULT_MAX_FILE_ATTACHMENT_BYTES
    assert DEFAULT_WRITE_LIMIT_BYTES == 256 * 1024
    assert MAX_WRITE_LIMIT_BYTES == 4 * 1024 * 1024
    assert DEFAULT_LIST_LIMIT == 500
    assert MAX_LIST_LIMIT == 10_000
    assert DEFAULT_OUTPUT_LIMIT_BYTES == 50_000
    assert MAX_OUTPUT_LIMIT_BYTES == 200_000
    assert DEFAULT_TIMEOUT_SECONDS == 60
    assert MAX_TIMEOUT_SECONDS == 600

    assert ReadFileTool().schema["properties"]["max_bytes"]["default"] == 256 * 1024
    assert ReadFileTool().schema["properties"]["max_bytes"]["maximum"] == 4 * 1024 * 1024
    assert ReadFileTool().schema["properties"]["max_attachment_bytes"]["default"] == (
        DEFAULT_MAX_FILE_ATTACHMENT_BYTES
    )
    assert ReadFileTool().schema["properties"]["max_attachment_bytes"]["maximum"] == (
        DEFAULT_MAX_FILE_ATTACHMENT_BYTES
    )
    custom_read_file = ReadFileTool(
        default_attachment_limit_bytes=10 * 1024 * 1024,
        max_attachment_limit_bytes=12 * 1024 * 1024,
    )
    assert custom_read_file.schema["properties"]["max_attachment_bytes"]["default"] == (
        10 * 1024 * 1024
    )
    assert custom_read_file.schema["properties"]["max_attachment_bytes"]["maximum"] == (
        12 * 1024 * 1024
    )
    assert WriteFileTool().schema["properties"]["max_bytes"]["default"] == 256 * 1024
    assert WriteFileTool().schema["properties"]["max_bytes"]["maximum"] == 4 * 1024 * 1024
    assert ListFilesTool().schema["properties"]["limit"]["default"] == 500
    assert ListFilesTool().schema["properties"]["limit"]["maximum"] == 10_000
    assert ListArtifactsTool().schema["properties"]["limit"]["default"] == 500
    assert ListArtifactsTool().schema["properties"]["limit"]["maximum"] == 10_000
    assert ExecCommandTool().schema["properties"]["max_output_bytes"]["default"] == 50_000
    assert ExecCommandTool().schema["properties"]["max_output_bytes"]["maximum"] == 200_000
    assert ExecCommandTool().schema["properties"]["timeout_s"]["default"] == 60
    assert ExecCommandTool().schema["properties"]["timeout_s"]["maximum"] == 600
    assert ExecCommandTool().schema["properties"]["argv"]["minItems"] == 1
    assert ExecCommandTool().schema["properties"]["argv"]["items"] == {
        "type": "string",
        "minLength": 1,
        "pattern": r"\S",
    }
    assert ExecCommandTool().schema["properties"]["shell"]["minLength"] == 1
    assert ExecCommandTool().schema["properties"]["shell"]["pattern"] == r"\S"
    assert "oneOf" not in ExecCommandTool().schema
    assert "anyOf" not in ExecCommandTool().schema
    assert "allOf" not in ExecCommandTool().schema


def test_workspace_tools_read_write_and_list_files(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    ctx = ToolContext(session_id="sess_1", workspace=workspace)

    write_result = asyncio.run(
        WriteFileTool().run(ctx, {"path": "notes/result.txt", "content": "hello"})
    )
    read_result = asyncio.run(ReadFileTool().run(ctx, {"path": "notes/result.txt"}))
    list_result = asyncio.run(ListFilesTool().run(ctx, {"pattern": "**/*.txt"}))

    assert write_result.is_error is False
    assert write_result.structured == {
        "path": "notes/result.txt",
        "bytes": 5,
        "encoding": "utf-8",
    }
    assert read_result.content == "hello"
    assert read_result.structured == {
        "source": "workspace",
        "path": "notes/result.txt",
        "bytes": 5,
        "total_bytes": 5,
        "encoding": "utf-8",
        "truncated": False,
    }
    assert list_result.content == "notes/result.txt"
    assert list_result.structured == {
        "pattern": "**/*.txt",
        "files": ["notes/result.txt"],
        "total_files": 1,
        "truncated": False,
    }


def test_workspace_tools_return_error_without_workspace():
    ctx = ToolContext(session_id="sess_1")

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "notes/result.txt"}))

    assert result.is_error is True
    assert result.content == "No workspace configured for this tool call."


def test_artifact_store_tools_read_and_list_artifacts(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact = asyncio.run(
        artifact_store.put_bytes(
            b"invoice text",
            filename="invoice.txt",
            content_type="text/plain",
            session_id="sess_1",
            agent_name="assistant",
            environment_name="local-dev",
        )
    )
    asyncio.run(
        artifact_store.put_bytes(
            b"shared notes",
            filename="shared.txt",
            content_type="text/plain",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="local-dev",
        )
    )
    ctx = ToolContext(
        session_id="sess_1",
        agent_name="assistant",
        environment_name="local-dev",
        artifact_store=artifact_store,
    )

    read_result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": artifact.id}))
    list_session_result = asyncio.run(ListArtifactsTool().run(ctx, {}))
    list_environment_result = asyncio.run(
        ListArtifactsTool().run(ctx, {"scope": ArtifactScope.ENVIRONMENT.value})
    )

    assert read_result.content == "invoice text"
    assert read_result.structured == {
        "source": "artifact",
        "artifact_id": artifact.id,
        "filename": "invoice.txt",
        "content_type": "text/plain",
        "bytes": 12,
        "total_bytes": 12,
        "size_bytes": 12,
        "scope": "session",
        "session_id": "sess_1",
        "agent_name": "assistant",
        "environment_name": "local-dev",
        "truncated": False,
        "encoding": "utf-8",
    }
    assert artifact.id in list_session_result.content
    assert list_session_result.structured["scope"] == "session"
    assert [item["artifact_id"] for item in list_session_result.structured["artifacts"]] == [
        artifact.id
    ]
    assert "shared.txt" in list_environment_result.content
    assert list_environment_result.structured["scope"] == "environment"


def test_read_file_requires_exactly_one_source(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    ctx = ToolContext(session_id="sess_1", workspace=workspace, artifact_store=artifact_store)

    with pytest.raises(ValueError, match="exactly one"):
        asyncio.run(ReadFileTool().run(ctx, {}))

    with pytest.raises(ValueError, match="exactly one"):
        asyncio.run(ReadFileTool().run(ctx, {"path": "a.txt", "artifact_id": "art_1"}))


def test_read_file_returns_provider_neutral_image_attachment_without_base64(tmp_path, monkeypatch):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact = asyncio.run(
        artifact_store.put_bytes(
            TINY_PNG_BYTES,
            filename="image.png",
            content_type="image/png",
            session_id="sess_1",
        )
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)
    monkeypatch.setattr(
        files_module,
        "_detect_image_content_type",
        lambda content: ("image/png", None),
    )

    result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": artifact.id}))

    assert result.is_error is False
    assert "Attached image artifact" in result.content
    assert result.structured["artifact_id"] == artifact.id
    assert result.structured["content_type"] == "image/png"
    assert result.structured["attachment_artifact_id"] == artifact.id
    assert result.artifacts == [
        {
            "type": "cayu.file_attachment.v1",
            "artifact_id": artifact.id,
            "kind": "image",
            "filename": "image.png",
            "content_type": "image/png",
            "size_bytes": len(TINY_PNG_BYTES),
            "metadata": {"source_artifact_id": artifact.id},
        }
    ]
    assert "base64" not in result.structured
    assert "base64" not in result.artifacts[0]


def test_tiny_png_fixture_is_valid_for_native_image_reader():
    detected_content_type, validation_error = files_module._detect_image_content_type(
        TINY_PNG_BYTES
    )

    assert validation_error is None
    assert detected_content_type == "image/png"


def test_read_file_rejects_mislabeled_image_attachment(tmp_path, monkeypatch):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact = asyncio.run(
        artifact_store.put_bytes(
            TINY_PNG_BYTES,
            filename="image.png",
            content_type="image/png",
            session_id="sess_1",
        )
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)
    monkeypatch.setattr(
        files_module,
        "_detect_image_content_type",
        lambda content: ("image/jpeg", None),
    )

    result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": artifact.id}))

    assert result.is_error is True
    assert result.artifacts == []
    assert result.content == (
        "Image 'image.png' content type mismatch: metadata says image/png, "
        "but bytes are image/jpeg."
    )


def test_read_file_rejects_empty_native_file_attachments(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    empty_image = asyncio.run(
        artifact_store.put_bytes(
            b"",
            filename="empty.png",
            content_type="image/png",
            session_id="sess_1",
        )
    )
    empty_pdf = asyncio.run(
        artifact_store.put_bytes(
            b"",
            filename="empty.pdf",
            content_type="application/pdf",
            session_id="sess_1",
        )
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    image_result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": empty_image.id}))
    pdf_result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": empty_pdf.id}))

    assert image_result.is_error is True
    assert image_result.content == "Image artifact 'empty.png' is empty and cannot be inspected."
    assert image_result.artifacts == []
    assert pdf_result.is_error is True
    assert pdf_result.content == "PDF artifact 'empty.pdf' is empty and cannot be inspected."
    assert pdf_result.artifacts == []


def test_read_file_rejects_small_corrupt_native_file_attachments(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    corrupt_image = asyncio.run(
        artifact_store.put_bytes(
            b"not an image",
            filename="bad.png",
            content_type="image/png",
            session_id="sess_1",
        )
    )
    corrupt_pdf = asyncio.run(
        artifact_store.put_bytes(
            b"not a pdf",
            filename="bad.pdf",
            content_type="application/pdf",
            session_id="sess_1",
        )
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    image_result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": corrupt_image.id}))
    pdf_result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": corrupt_pdf.id}))

    assert image_result.is_error is True
    assert "Image 'bad.png' could not be inspected:" in image_result.content
    assert image_result.artifacts == []
    assert pdf_result.is_error is True
    assert "PDF 'bad.pdf' could not be inspected:" in pdf_result.content
    assert pdf_result.artifacts == []


def test_read_file_returns_error_for_image_parser_failures(tmp_path, monkeypatch):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact = asyncio.run(
        artifact_store.put_bytes(
            b"not an image",
            filename="bad.png",
            content_type="image/png",
            session_id="sess_1",
        )
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    def fail_resize(content: bytes, *, content_type: str, max_bytes: int):
        raise ValueError("invalid image bytes")

    monkeypatch.setattr(
        files_module,
        "_detect_image_content_type",
        lambda content: ("image/png", None),
    )
    monkeypatch.setattr(files_module, "_resize_image_bytes", fail_resize)

    result = asyncio.run(
        ReadFileTool(
            default_attachment_limit_bytes=1,
            max_attachment_limit_bytes=1,
        ).run(ctx, {"artifact_id": artifact.id})
    )

    assert result.is_error is True
    assert result.content == "Image 'bad.png' could not be inspected: invalid image bytes"
    assert result.artifacts == []


def test_read_file_returns_error_for_pdf_parser_failures(tmp_path, monkeypatch):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact = asyncio.run(
        artifact_store.put_bytes(
            b"not a pdf",
            filename="bad.pdf",
            content_type="application/pdf",
            session_id="sess_1",
        )
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    def fail_extract(content: bytes, pages: str | None):
        raise ValueError("invalid PDF bytes")

    monkeypatch.setattr(files_module, "_extract_pdf_pages", fail_extract)

    result = asyncio.run(
        ReadFileTool().run(
            ctx,
            {
                "artifact_id": artifact.id,
                "pages": "1",
            },
        )
    )

    assert result.is_error is True
    assert result.content == "PDF 'bad.pdf' could not be inspected: invalid PDF bytes"
    assert result.artifacts == []


def test_read_file_extends_default_artifact_readers(tmp_path):
    (tmp_path / "workspace").mkdir()
    workspace = LocalWorkspace(tmp_path / "workspace", workspace_id="local")
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    pdf = asyncio.run(
        artifact_store.put_bytes(
            b"%PDF custom",
            filename="invoice.pdf",
            content_type="application/pdf",
            session_id="sess_1",
        )
    )
    text = asyncio.run(
        artifact_store.put_bytes(
            b"text artifact ok",
            filename="notes.txt",
            content_type="text/plain",
            session_id="sess_1",
        )
    )
    asyncio.run(workspace.write_bytes("notes.txt", b"workspace ok"))
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        artifact_store=artifact_store,
    )
    tool = ReadFileTool(extra_artifact_readers=[CustomPdfReader()])

    workspace_result = asyncio.run(tool.run(ctx, {"path": "notes.txt"}))
    pdf_result = asyncio.run(tool.run(ctx, {"artifact_id": pdf.id}))
    text_result = asyncio.run(tool.run(ctx, {"artifact_id": text.id}))

    assert workspace_result.content == "workspace ok"
    assert pdf_result.content == "custom pdf reader: invoice.pdf"
    assert pdf_result.structured["reader"] == "custom"
    assert text_result.content == "text artifact ok"
    assert text_result.structured["encoding"] == "utf-8"


def test_read_file_can_replace_artifact_readers_explicitly(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    text = asyncio.run(
        artifact_store.put_bytes(
            b"text artifact ok",
            filename="notes.txt",
            content_type="text/plain",
            session_id="sess_1",
        )
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)
    tool = ReadFileTool(artifact_readers=[CustomPdfReader()])

    result = asyncio.run(tool.run(ctx, {"artifact_id": text.id}))

    assert result.is_error is True
    assert "No built-in reader is available" in result.content


def test_read_file_accepts_empty_extra_artifact_readers(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    text = asyncio.run(
        artifact_store.put_bytes(
            b"text artifact ok",
            filename="notes.txt",
            content_type="text/plain",
            session_id="sess_1",
        )
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)
    tool = ReadFileTool(extra_artifact_readers=[])

    result = asyncio.run(tool.run(ctx, {"artifact_id": text.id}))

    assert result.is_error is False
    assert result.content == "text artifact ok"


def test_read_file_rejects_ambiguous_artifact_reader_configuration():
    with pytest.raises(ValueError, match="Use either artifact_readers"):
        ReadFileTool(
            artifact_readers=[CustomPdfReader()],
            extra_artifact_readers=[CustomPdfReader()],
        )


def test_read_file_rejects_invalid_attachment_limit_configuration():
    with pytest.raises(ValueError, match="greater than zero"):
        ReadFileTool(default_attachment_limit_bytes=0)

    with pytest.raises(ValueError, match="less than or equal"):
        ReadFileTool(
            default_attachment_limit_bytes=3,
            max_attachment_limit_bytes=2,
        )


def test_read_file_enforces_artifact_scope(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    other_session_artifact = asyncio.run(
        artifact_store.put_bytes(
            b"secret",
            filename="secret.txt",
            content_type="text/plain",
            session_id="sess_other",
        )
    )
    other_environment_artifact = asyncio.run(
        artifact_store.put_bytes(
            b"shared",
            filename="shared.txt",
            content_type="text/plain",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="prod",
        )
    )
    ctx = ToolContext(
        session_id="sess_1",
        environment_name="local-dev",
        artifact_store=artifact_store,
    )

    session_result = asyncio.run(
        ReadFileTool().run(ctx, {"artifact_id": other_session_artifact.id})
    )
    environment_result = asyncio.run(
        ReadFileTool().run(ctx, {"artifact_id": other_environment_artifact.id})
    )

    assert session_result.is_error is True
    assert session_result.content == "Artifact is not available in this session."
    assert session_result.structured == {
        "artifact_id": other_session_artifact.id,
        "scope": "session",
    }
    assert environment_result.is_error is True
    assert environment_result.content == "Artifact is not available in this environment."
    assert environment_result.structured == {
        "artifact_id": other_environment_artifact.id,
        "scope": "environment",
    }


def test_artifact_tools_return_error_without_artifact_store():
    ctx = ToolContext(session_id="sess_1")

    read_result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": "art_1"}))
    list_result = asyncio.run(ListArtifactsTool().run(ctx, {}))

    assert read_result.is_error is True
    assert read_result.content == "No artifact store configured for this tool call."
    assert list_result.is_error is True
    assert list_result.content == "No artifact store configured for this tool call."


def test_exec_command_tool_runs_process_and_reports_failures(tmp_path):
    ctx = ToolContext(session_id="sess_1", runner=LocalRunner(tmp_path))

    ok = asyncio.run(
        ExecCommandTool().run(
            ctx,
            {
                "kind": "process",
                "argv": [sys.executable, "-c", "print('ok')"],
            },
        )
    )
    failed = asyncio.run(
        ExecCommandTool().run(
            ctx,
            {
                "kind": "process",
                "argv": [sys.executable, "-c", "import sys; sys.exit(3)"],
            },
        )
    )

    assert ok.is_error is False
    assert ok.content == "ok"
    assert ok.structured["exit_code"] == 0
    assert ok.structured["stdout_truncated"] is False
    assert ok.structured["stderr_truncated"] is False
    assert failed.is_error is True
    assert failed.structured["exit_code"] == 3


def test_builtin_tools_truncate_model_facing_large_outputs(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    file_ctx = ToolContext(session_id="sess_1", workspace=workspace)
    run_ctx = ToolContext(session_id="sess_1", runner=LocalRunner(tmp_path))

    asyncio.run(WriteFileTool().run(file_ctx, {"path": "large.txt", "content": "abcdef"}))
    asyncio.run(WriteFileTool().run(file_ctx, {"path": "other.txt", "content": ""}))
    read_result = asyncio.run(ReadFileTool().run(file_ctx, {"path": "large.txt", "max_bytes": 3}))
    list_result = asyncio.run(ListFilesTool().run(file_ctx, {"pattern": "*.txt", "limit": 1}))
    command_result = asyncio.run(
        ExecCommandTool().run(
            run_ctx,
            {
                "argv": [sys.executable, "-c", "print('abcdef')"],
                "max_output_bytes": 3,
            },
        )
    )

    assert read_result.content == "abc\n\n[file truncated]"
    assert read_result.structured["truncated"] is True
    assert read_result.structured["total_bytes"] == 6
    assert list_result.content.endswith("[file list truncated]")
    assert list_result.structured["total_files"] is None
    assert list_result.structured["truncated"] is True
    assert command_result.structured["stdout"] == "abc"
    assert command_result.structured["stdout_truncated"] is True
    assert "[output truncated]" in command_result.content


def test_write_file_tool_refuses_oversized_content(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    ctx = ToolContext(session_id="sess_1", workspace=workspace)

    result = asyncio.run(
        WriteFileTool().run(
            ctx,
            {
                "path": "large.txt",
                "content": "abcdef",
                "max_bytes": 3,
            },
        )
    )

    assert result.is_error is True
    assert result.content == ("Write refused: content is 6 bytes, which exceeds max_bytes=3.")
    assert result.structured == {
        "path": "large.txt",
        "bytes": 6,
        "max_bytes": 3,
        "encoding": "utf-8",
    }
    assert not (tmp_path / "large.txt").exists()


def test_exec_command_tool_applies_default_and_max_timeout(tmp_path):
    runner = RecordingRunner()
    ctx = ToolContext(session_id="sess_1", runner=runner)

    result = asyncio.run(
        ExecCommandTool().run(
            ctx,
            {
                "argv": [sys.executable, "-c", "print('ok')"],
            },
        )
    )

    assert result.is_error is False
    assert runner.timeout_s == 60
    with pytest.raises(ValueError, match="at most 600"):
        asyncio.run(
            ExecCommandTool().run(
                ctx,
                {
                    "argv": [sys.executable, "-c", "print('ok')"],
                    "timeout_s": 601,
                },
            )
        )


def test_exec_command_tool_reports_timeout_and_cancellation():
    timed_out_runner = RecordingRunner(ExecResult(exit_code=-9, timed_out=True))
    cancelled_runner = RecordingRunner(ExecResult(exit_code=-9, cancelled=True))

    timed_out = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="sess_1", runner=timed_out_runner),
            {
                "argv": [sys.executable, "-c", "print('ok')"],
                "timeout_s": 3,
            },
        )
    )
    cancelled = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="sess_1", runner=cancelled_runner),
            {
                "argv": [sys.executable, "-c", "print('ok')"],
            },
        )
    )

    assert timed_out.is_error is True
    assert timed_out.content == "Command timed out after 3 seconds."
    assert timed_out.structured["timed_out"] is True
    assert cancelled.is_error is True
    assert cancelled.content == "Command was cancelled."
    assert cancelled.structured["cancelled"] is True


def test_exec_command_tool_returns_error_without_runner():
    ctx = ToolContext(session_id="sess_1")

    result = asyncio.run(
        ExecCommandTool().run(
            ctx,
            {
                "kind": "process",
                "argv": [sys.executable, "-c", "print('ok')"],
            },
        )
    )

    assert result.is_error is True
    assert result.content == "No runner configured for this tool call."


def test_runtime_passes_environment_services_to_tool_context(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    runner = LocalRunner(tmp_path)
    tool = ContextRecordingTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="record_context",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local-dev"),
            workspace=workspace,
            artifact_store=artifact_store,
            runner=runner,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "record context")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert tool.context is not None
    assert tool.context.environment_name == "local-dev"
    assert tool.context.workspace_id == "local"
    assert tool.context.artifact_store_id == "artifacts"
    assert tool.context.workspace is workspace
    assert tool.context.artifact_store is artifact_store
    assert tool.context.runner is runner


def test_runtime_resolves_file_attachments_only_for_provider_request(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact = asyncio.run(
        artifact_store.put_bytes(
            b"image-bytes",
            filename="invoice.png",
            content_type="image/png",
            session_id="sess_attachments",
        )
    )
    tool = AttachmentTool(artifact.id, artifact.size_bytes)
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="attach_file",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local-dev"),
            artifact_store=artifact_store,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                session_id="sess_attachments",
                agent_name="assistant",
                messages=[Message.text("user", "attach")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    first_request_options = provider.requests[0].options
    second_request_options = provider.requests[1].options
    assert first_request_options[RESOLVED_FILE_ATTACHMENTS_OPTION] == {}
    assert second_request_options[RESOLVED_FILE_ATTACHMENTS_OPTION][artifact.id] == {
        "artifact_id": artifact.id,
        "kind": "image",
        "filename": "invoice.png",
        "content_type": "image/png",
        "data_base64": "aW1hZ2UtYnl0ZXM=",
        "metadata": {},
    }


def test_runtime_does_not_resend_old_file_attachments_by_default(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    first = asyncio.run(
        artifact_store.put_bytes(
            b"first",
            filename="first.png",
            content_type="image/png",
            session_id="sess_attachments",
        )
    )
    second = asyncio.run(
        artifact_store.put_bytes(
            b"second",
            filename="second.png",
            content_type="image/png",
            session_id="sess_attachments",
        )
    )
    tool = SequencedAttachmentTool(
        [
            (first.id, first.size_bytes),
            (second.id, second.size_bytes),
        ]
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="sequenced_attachment",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="sequenced_attachment",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local-dev"),
            artifact_store=artifact_store,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                session_id="sess_attachments",
                agent_name="assistant",
                messages=[Message.text("user", "attach twice")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert provider.requests[0].options[RESOLVED_FILE_ATTACHMENTS_OPTION] == {}
    assert list(provider.requests[1].options[RESOLVED_FILE_ATTACHMENTS_OPTION]) == [first.id]
    assert list(provider.requests[2].options[RESOLVED_FILE_ATTACHMENTS_OPTION]) == [second.id]


def test_runtime_rejects_oversized_file_attachment_reference(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact = asyncio.run(
        artifact_store.put_bytes(
            b"image-bytes",
            filename="invoice.png",
            content_type="image/png",
            session_id="sess_attachments",
        )
    )
    tool = AttachmentTool(artifact.id, DEFAULT_MAX_FILE_ATTACHMENT_BYTES + 1)
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="attach_file",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local-dev"), artifact_store=artifact_store),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                session_id="sess_attachments",
                agent_name="assistant",
                messages=[Message.text("user", "attach")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_FAILED
    assert "runtime attachment byte limit" in events[-1].payload["error"]
    assert len(provider.requests) == 1


def test_runtime_allows_configured_file_attachment_byte_limit():
    size_bytes = DEFAULT_MAX_FILE_ATTACHMENT_BYTES + 1
    artifact_store = SyntheticArtifactStore(
        artifact_id="art_large",
        size_bytes=size_bytes,
    )
    tool = AttachmentTool("art_large", size_bytes)
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="attach_file",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        max_file_attachment_bytes=size_bytes,
        max_total_file_attachment_bytes=size_bytes,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local-dev"), artifact_store=artifact_store),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                session_id="sess_attachments",
                agent_name="assistant",
                messages=[Message.text("user", "attach")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert artifact_store.read_limits == [size_bytes]
    assert (
        provider.requests[1].options[RESOLVED_FILE_ATTACHMENTS_OPTION]["art_large"]["data_base64"]
        == "eA=="
    )


def test_runtime_rejects_total_file_attachment_bytes(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    first = asyncio.run(
        artifact_store.put_bytes(
            b"first",
            filename="first.png",
            content_type="image/png",
            session_id="sess_attachments",
        )
    )
    second = asyncio.run(
        artifact_store.put_bytes(
            b"second",
            filename="second.png",
            content_type="image/png",
            session_id="sess_attachments",
        )
    )
    tool = MultipleAttachmentsTool([(first.id, first.size_bytes), (second.id, second.size_bytes)])
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="multiple_attachments",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(max_total_file_attachment_bytes=first.size_bytes)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local-dev"), artifact_store=artifact_store),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                session_id="sess_attachments",
                agent_name="assistant",
                messages=[Message.text("user", "attach")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_FAILED
    assert "total attachment byte limit" in events[-1].payload["error"]
    assert len(provider.requests) == 1


def test_runtime_counts_duplicate_file_attachment_references_toward_total_limit(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact = asyncio.run(
        artifact_store.put_bytes(
            b"image-bytes",
            filename="invoice.png",
            content_type="image/png",
            session_id="sess_attachments",
        )
    )
    tool = DuplicateAttachmentReferencesTool(artifact.id, artifact.size_bytes, count=2)
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="duplicate_attachments",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(max_total_file_attachment_bytes=artifact.size_bytes)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local-dev"), artifact_store=artifact_store),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                session_id="sess_attachments",
                agent_name="assistant",
                messages=[Message.text("user", "attach")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_FAILED
    assert "total attachment byte limit" in events[-1].payload["error"]
    assert len(provider.requests) == 1


def test_runtime_rejects_file_attachment_count(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    first = asyncio.run(
        artifact_store.put_bytes(
            b"first",
            filename="first.png",
            content_type="image/png",
            session_id="sess_attachments",
        )
    )
    second = asyncio.run(
        artifact_store.put_bytes(
            b"second",
            filename="second.png",
            content_type="image/png",
            session_id="sess_attachments",
        )
    )
    tool = MultipleAttachmentsTool([(first.id, first.size_bytes), (second.id, second.size_bytes)])
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="multiple_attachments",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(max_file_attachments_per_request=1)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local-dev"), artifact_store=artifact_store),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                session_id="sess_attachments",
                agent_name="assistant",
                messages=[Message.text("user", "attach")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_FAILED
    assert "attachment count" in events[-1].payload["error"]
    assert len(provider.requests) == 1


def test_runtime_counts_duplicate_file_attachment_references_toward_count_limit(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact = asyncio.run(
        artifact_store.put_bytes(
            b"image-bytes",
            filename="invoice.png",
            content_type="image/png",
            session_id="sess_attachments",
        )
    )
    tool = DuplicateAttachmentReferencesTool(artifact.id, artifact.size_bytes, count=2)
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="duplicate_attachments",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(max_file_attachments_per_request=1)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local-dev"), artifact_store=artifact_store),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                session_id="sess_attachments",
                agent_name="assistant",
                messages=[Message.text("user", "attach")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_FAILED
    assert "attachment count" in events[-1].payload["error"]
    assert len(provider.requests) == 1


def test_runtime_rejects_conflicting_file_attachment_references(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact = asyncio.run(
        artifact_store.put_bytes(
            b"image-bytes",
            filename="invoice.png",
            content_type="image/png",
            session_id="sess_attachments",
        )
    )
    tool = ConflictingAttachmentsTool(artifact.id, artifact.size_bytes)
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="conflicting_attachments",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local-dev"), artifact_store=artifact_store),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                session_id="sess_attachments",
                agent_name="assistant",
                messages=[Message.text("user", "attach")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_FAILED
    assert "Conflicting file attachment references" in events[-1].payload["error"]
    assert len(provider.requests) == 1


async def _collect_events(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]
