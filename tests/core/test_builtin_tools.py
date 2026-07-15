from __future__ import annotations

import asyncio
import base64
import io
import sys
import threading
from collections.abc import AsyncIterator
from importlib import import_module

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
from cayu.runners import ExecCommand, ExecResult, LocalRunner, Runner, RunnerUnavailableError
from cayu.runtime import CayuApp, RunRequest
from cayu.tools import ExecCommandTool
from cayu.tools.commands import (
    DEFAULT_OUTPUT_LIMIT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_OUTPUT_LIMIT_BYTES,
    MAX_TIMEOUT_SECONDS,
    CommandPolicy,
    CommandPolicyDecision,
    CommandPolicyResult,
    CommandRequest,
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
from cayu.workspaces import LocalWorkspace, WorkspaceReadResult

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
        read_size = self.size_bytes if max_bytes is None else min(self.size_bytes, max_bytes)
        return ArtifactReadResult(
            metadata=ArtifactMetadata(
                id=artifact_id,
                filename="large.png",
                content_type="image/png",
                size_bytes=self.size_bytes,
                session_id="sess_attachments",
            ),
            content=b"x" * read_size,
            total_bytes=self.size_bytes,
            truncated=read_size < self.size_bytes,
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
        self.command: ExecCommand | None = None
        self.cwd: str | None = None
        self.env: dict[str, str] | None = None
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
        self.command = command
        self.cwd = cwd
        self.env = env
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
        "causal_budget_id": None,
        "workspace_id": "local",
        "artifact_store_id": "artifacts",
        "idempotency_key": None,
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


def test_read_file_snapshots_workspace_pdf_as_artifact_attachment(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspace(workspace_root, workspace_id="local")
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    ctx = ToolContext(
        session_id="sess_1",
        agent_name="assistant",
        environment_name="local-dev",
        workspace=workspace,
        artifact_store=artifact_store,
    )
    pdf_bytes = _tiny_pdf_bytes()

    asyncio.run(workspace.write_bytes("docs/invoice.pdf", pdf_bytes))

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "docs/invoice.pdf"}))

    assert result.is_error is False
    assert "Attached PDF artifact" in result.content
    assert "docs/invoice.pdf" in result.content
    assert "%PDF" not in result.content
    assert result.structured["source"] == "workspace"
    assert result.structured["path"] == "docs/invoice.pdf"
    assert result.structured["content_type"] == "application/pdf"
    assert result.structured["snapshot_artifact_id"].startswith("art_")
    assert result.structured["attachment_artifact_id"] == result.structured["snapshot_artifact_id"]
    assert result.artifacts[0]["artifact_id"] == result.structured["snapshot_artifact_id"]
    assert result.artifacts[0]["kind"] == "document"


def test_read_file_forwards_pages_for_workspace_pdf_snapshot(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspace(workspace_root, workspace_id="local")
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        artifact_store=artifact_store,
    )

    asyncio.run(workspace.write_bytes("docs/report.pdf", _tiny_pdf_bytes()))

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "docs/report.pdf", "pages": "1"}))

    assert result.is_error is False
    assert "showing pages 1-1 of 1" in result.content
    assert result.structured["source"] == "workspace"
    assert result.structured["path"] == "docs/report.pdf"
    assert result.structured["pages"] == "1"
    assert result.structured["snapshot_artifact_id"].startswith("art_")
    assert result.structured["attachment_artifact_id"] != result.structured["snapshot_artifact_id"]
    assert result.artifacts[0]["kind"] == "document"
    assert result.artifacts[0]["metadata"]["pages"] == "1"


def test_read_file_snapshots_workspace_image_as_artifact_attachment(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspace(workspace_root, workspace_id="local")
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    ctx = ToolContext(
        session_id="sess_1",
        agent_name="assistant",
        environment_name="local-dev",
        workspace=workspace,
        artifact_store=artifact_store,
    )

    asyncio.run(workspace.write_bytes("images/red-dot.png", TINY_PNG_BYTES))

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "images/red-dot.png"}))

    assert result.is_error is False
    assert "Attached image artifact" in result.content
    assert "images/red-dot.png" in result.content
    assert "\ufffdPNG" not in result.content
    assert result.structured["content_type"] == "image/png"
    assert result.structured["snapshot_artifact_id"].startswith("art_")
    assert result.structured["attachment_artifact_id"] == result.structured["snapshot_artifact_id"]
    assert result.artifacts[0]["artifact_id"] == result.structured["snapshot_artifact_id"]
    assert result.artifacts[0]["kind"] == "image"
    assert result.structured["total_bytes"] == len(TINY_PNG_BYTES)


def test_read_file_rejects_inspectable_workspace_binary_without_artifact_store(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    ctx = ToolContext(session_id="sess_1", workspace=workspace)

    asyncio.run(workspace.write_bytes("images/red-dot.png", TINY_PNG_BYTES))

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "images/red-dot.png"}))

    assert result.is_error is True
    assert "requires an artifact store" in result.content
    assert result.structured["content_type"] == "image/png"
    assert result.structured["binary"] is True
    assert result.structured["inspectable"] is True


def test_read_file_routes_empty_workspace_pdf_and_image_to_artifact_readers(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspace(workspace_root, workspace_id="local")
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        artifact_store=artifact_store,
    )

    asyncio.run(workspace.write_bytes("empty.pdf", b""))
    asyncio.run(workspace.write_bytes("empty.png", b""))

    pdf_result = asyncio.run(ReadFileTool().run(ctx, {"path": "empty.pdf"}))
    image_result = asyncio.run(ReadFileTool().run(ctx, {"path": "empty.png"}))

    assert pdf_result.is_error is True
    assert "PDF artifact 'empty.pdf' is empty and cannot be inspected." in pdf_result.content
    assert pdf_result.structured["source"] == "workspace"
    assert pdf_result.structured["content_type"] == "application/pdf"
    assert pdf_result.structured["binary"] is True
    assert pdf_result.structured["inspectable"] is True
    assert image_result.is_error is True
    assert "Image artifact 'empty.png' is empty and cannot be inspected." in image_result.content
    assert image_result.structured["source"] == "workspace"
    assert image_result.structured["content_type"] == "image/png"
    assert image_result.structured["binary"] is True
    assert image_result.structured["inspectable"] is True


def test_read_file_rejects_unsupported_workspace_binary_without_returning_raw_bytes(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    ctx = ToolContext(session_id="sess_1", workspace=workspace)
    binary = b"\x00\x01\x02\x03\x04binary data"

    asyncio.run(workspace.write_bytes("build/app.bin", binary))

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "build/app.bin"}))

    assert result.is_error is True
    assert "Workspace file 'build/app.bin' appears to be binary" in result.content
    assert "binary data" not in result.content
    assert result.structured["content_type"] == "application/octet-stream"
    assert result.structured["binary"] is True
    assert result.structured["inspectable"] is False


def test_read_file_rejects_binary_bytes_even_with_text_extension(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    ctx = ToolContext(session_id="sess_1", workspace=workspace)
    binary = b"looks textual first\x00\x01binary data"

    asyncio.run(workspace.write_bytes("notes/payload.txt", binary))

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "notes/payload.txt"}))

    assert result.is_error is True
    assert "appears to be binary" in result.content
    assert "binary data" not in result.content
    assert result.structured["content_type"] == "text/plain"
    assert result.structured["binary"] is True
    assert result.structured["inspectable"] is False


def test_read_file_rejects_binary_bytes_after_text_prefix(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    ctx = ToolContext(session_id="sess_1", workspace=workspace)
    binary = b"a" * (9 * 1024) + b"\x00\x01binary tail"

    asyncio.run(workspace.write_bytes("notes/payload.txt", binary))

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "notes/payload.txt"}))

    assert result.is_error is True
    assert "appears to be binary" in result.content
    assert "binary tail" not in result.content
    assert result.structured["content_type"] == "text/plain"
    assert result.structured["binary"] is True
    assert result.structured["inspectable"] is False


def test_read_file_returns_tool_error_when_workspace_attachment_changes_during_snapshot(tmp_path):
    class MutatingWorkspace(LocalWorkspace):
        def __init__(self, root):
            super().__init__(root, workspace_id="local")
            self.read_count = 0

        async def read_bytes(self, path: str, *, max_bytes: int | None = None):
            self.read_count += 1
            if self.read_count == 1:
                return await super().read_bytes(path, max_bytes=max_bytes)
            return WorkspaceReadResult(
                content=b"now text",
                total_bytes=8,
                truncated=False,
            )

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = MutatingWorkspace(workspace_root)
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        artifact_store=artifact_store,
    )

    asyncio.run(workspace.write_bytes("images/red-dot.png", TINY_PNG_BYTES))

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "images/red-dot.png"}))

    assert result.is_error is True
    assert "changed while it was being captured" in result.content
    assert "Retry read_file" in result.content
    assert result.structured["content_type"] == "image/png"
    assert result.structured["binary"] is True
    assert result.structured["inspectable"] is True


def test_read_file_still_reads_text_like_workspace_formats(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    ctx = ToolContext(session_id="sess_1", workspace=workspace)

    asyncio.run(workspace.write_bytes("data/results.csv", b"name,score\ncayu,10\n"))
    asyncio.run(workspace.write_bytes("pages/index.html", b"<h1>Cayu</h1>\n"))
    asyncio.run(workspace.write_bytes("README", b"Cayu workspace notes\n"))
    asyncio.run(workspace.write_bytes("src/app.ts", b"export const name = 'cayu';\n"))
    asyncio.run(workspace.write_bytes("src/Main.java", b"class Main {}\n"))

    csv_result = asyncio.run(ReadFileTool().run(ctx, {"path": "data/results.csv"}))
    html_result = asyncio.run(ReadFileTool().run(ctx, {"path": "pages/index.html"}))
    readme_result = asyncio.run(ReadFileTool().run(ctx, {"path": "README"}))
    typescript_result = asyncio.run(ReadFileTool().run(ctx, {"path": "src/app.ts"}))
    java_result = asyncio.run(ReadFileTool().run(ctx, {"path": "src/Main.java"}))

    assert csv_result.is_error is False
    assert csv_result.content == "name,score\ncayu,10\n"
    assert csv_result.structured["encoding"] == "utf-8"
    assert "binary" not in csv_result.structured
    assert html_result.is_error is False
    assert html_result.content == "<h1>Cayu</h1>\n"
    assert readme_result.is_error is False
    assert readme_result.content == "Cayu workspace notes\n"
    assert readme_result.structured["encoding"] == "utf-8"
    assert "binary" not in readme_result.structured
    assert typescript_result.is_error is False
    assert typescript_result.content == "export const name = 'cayu';\n"
    assert typescript_result.structured["encoding"] == "utf-8"
    assert "binary" not in typescript_result.structured
    assert java_result.is_error is False
    assert java_result.content == "class Main {}\n"
    assert java_result.structured["encoding"] == "utf-8"
    assert "binary" not in java_result.structured


def _tiny_pdf_bytes() -> bytes:
    pypdf = import_module("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _truncated_jpeg_bytes() -> bytes:
    image_module = import_module("PIL.Image")
    buffer = io.BytesIO()
    image_module.new("RGB", (1, 1), "white").save(buffer, format="JPEG")
    return buffer.getvalue()[:-1]


def _multi_page_pdf_bytes(page_count: int) -> bytes:
    pypdf = import_module("pypdf")
    writer = pypdf.PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_read_file_caps_pages_for_small_many_page_pdf(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    pdf = asyncio.run(
        artifact_store.put_bytes(
            _multi_page_pdf_bytes(12),
            filename="report.pdf",
            content_type="application/pdf",
            session_id="sess_1",
        )
    )
    # The 12-page PDF is tiny and fits comfortably under the byte cap, but the
    # 10-page limit must still be enforced instead of attaching the whole file.
    assert pdf.size_bytes < DEFAULT_ATTACHMENT_LIMIT_BYTES
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": pdf.id}))

    assert result.is_error is False
    assert "showing pages 1-10 of 12" in result.content
    attachment_id = result.structured["attachment_artifact_id"]
    assert attachment_id != pdf.id
    attachment = asyncio.run(artifact_store.read_bytes(attachment_id))
    pypdf = import_module("pypdf")
    assert len(pypdf.PdfReader(io.BytesIO(attachment.content)).pages) == 10


@pytest.mark.parametrize(
    ("pages", "message"),
    [
        ("not-a-page", "must be a page number or range"),
        ("2-1", "must be a valid 1-based page range"),
        ("1-11", "may include at most 10 pages"),
        ("13", "starts after the end of the PDF"),
    ],
)
def test_read_file_reports_invalid_pdf_page_ranges(tmp_path, pages, message):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    pdf = asyncio.run(
        artifact_store.put_bytes(
            _multi_page_pdf_bytes(12),
            filename="report.pdf",
            content_type="application/pdf",
            session_id="sess_1",
        )
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": pdf.id, "pages": pages}))

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert message in result.content


def test_read_file_dedupes_repeated_pdf_page_extraction(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    pdf = asyncio.run(
        artifact_store.put_bytes(
            _multi_page_pdf_bytes(12),
            filename="report.pdf",
            content_type="application/pdf",
            session_id="sess_1",
        )
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    first = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": pdf.id}))
    second = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": pdf.id}))

    assert first.structured["attachment_artifact_id"] == second.structured["attachment_artifact_id"]
    # Re-reading must not store a second multi-page copy: the source plus a single
    # derived attachment are the only session artifacts.
    listing = asyncio.run(artifact_store.list(scope=ArtifactScope.SESSION, session_id="sess_1"))
    assert listing.total_count == 2


def test_read_file_dedupes_across_page_selections(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    pdf = asyncio.run(
        artifact_store.put_bytes(
            _multi_page_pdf_bytes(12),
            filename="report.pdf",
            content_type="application/pdf",
            session_id="sess_1",
        )
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    first = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": pdf.id, "pages": "1-2"}))
    second = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": pdf.id, "pages": "3-4"}))
    third = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": pdf.id, "pages": "1-2"}))

    ids = {
        first.structured["attachment_artifact_id"],
        second.structured["attachment_artifact_id"],
        third.structured["attachment_artifact_id"],
    }
    # Distinct page selections are distinct derivations; the repeated selection reuses.
    assert len(ids) == 2
    assert first.structured["attachment_artifact_id"] == third.structured["attachment_artifact_id"]
    listing = asyncio.run(artifact_store.list(scope=ArtifactScope.SESSION, session_id="sess_1"))
    assert listing.total_count == 3


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

    missing_result = asyncio.run(ReadFileTool().run(ctx, {}))
    assert missing_result.is_error is True
    assert missing_result.structured == {"error": "invalid_arguments"}
    assert "exactly one" in missing_result.content

    both_result = asyncio.run(ReadFileTool().run(ctx, {"path": "a.txt", "artifact_id": "art_1"}))
    assert both_result.is_error is True
    assert both_result.structured == {"error": "invalid_arguments"}
    assert "exactly one" in both_result.content


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
    validation_threads: list[int] = []

    def detect_image_content_type(_content: bytes) -> tuple[str, None]:
        validation_threads.append(threading.get_ident())
        return "image/png", None

    monkeypatch.setattr(
        files_module,
        "_detect_image_content_type",
        detect_image_content_type,
    )

    async def read() -> tuple[int, ToolResult]:
        event_loop_thread = threading.get_ident()
        result = await ReadFileTool().run(ctx, {"artifact_id": artifact.id})
        return event_loop_thread, result

    event_loop_thread, result = asyncio.run(read())

    assert result.is_error is False
    assert validation_threads
    assert validation_threads[0] != event_loop_thread
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


def test_read_file_rejects_truncated_jpeg_that_passes_header_verification(tmp_path):
    image_module = import_module("PIL.Image")
    truncated = _truncated_jpeg_bytes()
    with image_module.open(io.BytesIO(truncated)) as image:
        image.verify()

    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact = asyncio.run(
        artifact_store.put_bytes(
            truncated,
            filename="truncated.jpg",
            content_type="image/jpeg",
            session_id="sess_1",
        )
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": artifact.id}))

    assert result.is_error is True
    assert result.artifacts == []
    assert "could not be inspected" in result.content


def test_read_file_rejects_image_decompression_bomb_warning(tmp_path, monkeypatch):
    image_module = import_module("PIL.Image")
    buffer = io.BytesIO()
    image_module.new("RGB", (15, 10), "white").save(buffer, format="PNG")
    monkeypatch.setattr(image_module, "MAX_IMAGE_PIXELS", 100)
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact = asyncio.run(
        artifact_store.put_bytes(
            buffer.getvalue(),
            filename="oversized.png",
            content_type="image/png",
            session_id="sess_1",
        )
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": artifact.id}))

    assert result.is_error is True
    assert result.artifacts == []
    assert "could not be inspected" in result.content


def test_read_file_rejects_image_over_decoded_size_limit(tmp_path, monkeypatch):
    import cayu.artifacts._images as image_validation_module

    image_module = import_module("PIL.Image")
    buffer = io.BytesIO()
    image_module.new("RGB", (15, 10), "white").save(buffer, format="PNG")
    monkeypatch.setattr(image_validation_module, "MAX_IMAGE_DECODED_BYTES", 512)
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact = asyncio.run(
        artifact_store.put_bytes(
            buffer.getvalue(),
            filename="oversized.png",
            content_type="image/png",
            session_id="sess_1",
        )
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": artifact.id}))

    assert result.is_error is True
    assert result.artifacts == []
    assert "could not be inspected" in result.content


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


def test_read_file_uses_extra_artifact_readers_for_workspace_snapshots(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspace(workspace_root, workspace_id="local")
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        artifact_store=artifact_store,
    )
    tool = ReadFileTool(extra_artifact_readers=[CustomPdfReader()])

    asyncio.run(workspace.write_bytes("docs/invoice.pdf", b"%PDF custom"))

    result = asyncio.run(tool.run(ctx, {"path": "docs/invoice.pdf"}))

    assert result.is_error is False
    assert result.content.startswith(
        "Captured workspace file 'docs/invoice.pdf' as artifact snapshot"
    )
    assert "custom pdf reader: invoice.pdf" in result.content
    assert result.structured["source"] == "workspace"
    assert result.structured["path"] == "docs/invoice.pdf"
    assert result.structured["reader"] == "custom"
    assert result.structured["snapshot_artifact_id"].startswith("art_")


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
    # A plain nonzero exit is a normal command outcome the model should read,
    # not a tool error; only timeouts and cancellations flag is_error.
    assert failed.is_error is False
    assert failed.structured["exit_code"] == 3
    assert failed.content == "Command exited with code 3."


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
    over_limit_result = asyncio.run(
        ExecCommandTool().run(
            ctx,
            {
                "argv": [sys.executable, "-c", "print('ok')"],
                "timeout_s": 601,
            },
        )
    )
    assert over_limit_result.is_error is True
    assert over_limit_result.structured == {"error": "invalid_arguments"}
    assert "at most 600" in over_limit_result.content


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


def test_exec_command_tool_preserves_runner_unavailable_diagnostic() -> None:
    diagnostic = {
        "type": "cayu.runner_unavailable.v1",
        "adapter": "microsandbox",
        "sandbox_name": "dead-agent",
        "status": "unavailable",
    }

    class UnavailableRunner(RecordingRunner):
        async def exec(self, *args, **kwargs) -> ExecResult:
            raise RunnerUnavailableError(
                "Microsandbox guest agent is unavailable.",
                diagnostic=diagnostic,
            )

    result = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="sess_1", runner=UnavailableRunner()),
            {"argv": ["pwd"]},
        )
    )

    assert result.is_error is True
    assert result.content == "Microsandbox guest agent is unavailable."
    assert result.structured == {
        "error": "runner_unavailable",
        "diagnostic": diagnostic,
    }
    assert result.artifacts == [diagnostic]


def test_runner_unavailable_diagnostic_reaches_durable_tool_event() -> None:
    diagnostic = {
        "type": "cayu.runner_unavailable.v1",
        "adapter": "microsandbox",
        "sandbox_name": "dead-agent",
        "status": "unavailable",
    }

    class UnavailableRunner(RecordingRunner):
        async def exec(self, *args, **kwargs) -> ExecResult:
            raise RunnerUnavailableError(
                "Microsandbox guest agent is unavailable.",
                diagnostic=diagnostic,
            )

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_dead_agent",
                    name="exec_command",
                    arguments={"argv": ["pwd"]},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("The sandbox must be replaced."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="dead-agent"), runner=UnavailableRunner()),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ExecCommandTool()],
    )

    async def run() -> list[Event]:
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_dead_agent",
                    messages=[Message.text("user", "run pwd")],
                )
            )
        ]
        return await app.session_store.load_events("sess_dead_agent")

    stored_events = asyncio.run(run())

    failed = next(event for event in stored_events if event.type == EventType.TOOL_CALL_FAILED)
    assert failed.payload["result"]["structured"] == {
        "error": "runner_unavailable",
        "diagnostic": diagnostic,
    }
    assert failed.payload["result"]["artifacts"] == [diagnostic]


def test_exec_command_tool_nonzero_exit_prefixes_output_with_exit_code():
    runner = RecordingRunner(ExecResult(stdout="partial\n", stderr="boom\n", exit_code=2))
    ctx = ToolContext(session_id="sess_1", runner=runner)

    result = asyncio.run(ExecCommandTool().run(ctx, {"argv": [sys.executable, "-c", "pass"]}))

    assert result.is_error is False
    assert result.content == ("Command exited with code 2.\n\nstdout:\npartial\n\nstderr:\nboom")
    assert result.structured["exit_code"] == 2


def test_exec_command_tool_rejects_argv_and_shell_together():
    ctx = ToolContext(session_id="sess_1", runner=RecordingRunner())
    tool = ExecCommandTool()

    both = asyncio.run(tool.run(ctx, {"argv": ["echo", "hi"], "shell": "echo hi"}))
    process_with_shell = asyncio.run(tool.run(ctx, {"kind": "process", "shell": "echo hi"}))
    shell_with_argv = asyncio.run(tool.run(ctx, {"kind": "shell", "argv": ["echo", "hi"]}))
    neither = asyncio.run(tool.run(ctx, {}))

    assert both.is_error is True
    assert both.structured == {"error": "invalid_arguments"}
    assert "cannot both be provided" in both.content
    assert process_with_shell.is_error is True
    assert "`shell` cannot be provided when kind is `process`" in process_with_shell.content
    assert shell_with_argv.is_error is True
    assert "`argv` cannot be provided when kind is `shell`" in shell_with_argv.content
    assert neither.is_error is True
    assert "must include `argv` or `shell`" in neither.content


def test_exec_command_tool_infers_kind_from_arguments():
    process_runner = RecordingRunner()
    shell_runner = RecordingRunner()

    process_result = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="sess_1", runner=process_runner),
            {"argv": ["echo", "hi"]},
        )
    )
    shell_result = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="sess_1", runner=shell_runner),
            {"shell": "echo hi"},
        )
    )

    assert process_result.is_error is False
    assert process_runner.command == ExecCommand.process("echo", "hi")
    assert shell_result.is_error is False
    assert shell_runner.command == ExecCommand.bash("echo hi")


class _StaticCommandPolicy(CommandPolicy):
    def __init__(self, result: CommandPolicyResult) -> None:
        self.result = result
        self.requests: list[tuple[ToolContext, CommandRequest]] = []

    async def evaluate(self, ctx: ToolContext, request: CommandRequest) -> CommandPolicyResult:
        self.requests.append((ctx, request))
        return self.result


def test_exec_command_tool_policy_receives_resolved_request_and_allows():
    runner = RecordingRunner()
    runner.default_cwd = "/workspace"
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))
    ctx = ToolContext(session_id="sess_1", runner=runner)

    result = asyncio.run(
        ExecCommandTool(policy=policy).run(
            ctx,
            {
                "argv": ["echo", "hi"],
                "cwd": "src/../tests",
                "env": {"FOO": "bar"},
                "timeout_s": 5,
            },
        )
    )

    assert result.is_error is False
    assert runner.command == ExecCommand.process("echo", "hi")
    assert len(policy.requests) == 1
    seen_ctx, seen_request = policy.requests[0]
    assert seen_ctx is ctx
    assert seen_request.command == ExecCommand.process("echo", "hi")
    assert seen_request.cwd == "src/../tests"
    assert seen_request.canonical_cwd == "/workspace/tests"
    assert seen_request.env == {"FOO": "bar"}
    assert seen_request.timeout_s == 5
    assert runner.cwd == "/workspace/tests"


@pytest.mark.parametrize(
    ("requested_cwd", "canonical_cwd"),
    [
        (None, "/workspace"),
        ("repo", "/workspace/repo"),
        ("/workspace/repo", "/workspace/repo"),
    ],
)
def test_exec_command_tool_policy_authorizes_and_executes_the_same_canonical_cwd(
    requested_cwd,
    canonical_cwd,
):
    runner = RecordingRunner()
    runner.default_cwd = "/workspace"
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))
    args = {"argv": ["pwd"]}
    if requested_cwd is not None:
        args["cwd"] = requested_cwd

    result = asyncio.run(
        ExecCommandTool(policy=policy).run(
            ToolContext(session_id="sess_1", runner=runner),
            args,
        )
    )

    assert result.is_error is False
    assert policy.requests[0][1].cwd == requested_cwd
    assert policy.requests[0][1].canonical_cwd == canonical_cwd
    assert runner.cwd == canonical_cwd


class _ResolvingRecordingRunner(RecordingRunner):
    def __init__(self) -> None:
        super().__init__()
        self.received_cwd: str | None = None
        self.executed_cwd: str | None = None

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
        self.received_cwd = cwd
        self.executed_cwd = self.resolve_cwd(cwd)
        return await super().exec(
            command,
            cwd=self.executed_cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )


class _ChangeRunnerDefaultPolicy(CommandPolicy):
    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self.request: CommandRequest | None = None

    async def evaluate(self, ctx: ToolContext, request: CommandRequest) -> CommandPolicyResult:
        self.request = request
        self.runner.default_cwd = "/other"
        return CommandPolicyResult(decision=CommandPolicyDecision.ALLOW)


def test_exec_command_tool_cannot_drift_after_authorizing_an_omitted_cwd():
    runner = _ResolvingRecordingRunner()
    runner.default_cwd = "/workspace"
    policy = _ChangeRunnerDefaultPolicy(runner)

    with pytest.raises(ValueError, match="outside the runner root"):
        asyncio.run(
            ExecCommandTool(policy=policy).run(
                ToolContext(session_id="sess_1", runner=runner),
                {"argv": ["pwd"]},
            )
        )

    assert policy.request is not None
    assert policy.request.cwd is None
    assert policy.request.canonical_cwd == "/workspace"
    assert runner.received_cwd == "/workspace"
    assert runner.executed_cwd is None
    assert runner.command is None


class _FailingCommandPolicy(CommandPolicy):
    async def evaluate(self, ctx: ToolContext, request: CommandRequest) -> CommandPolicyResult:
        del ctx, request
        raise ValueError("policy backend changed")


def test_exec_command_tool_does_not_classify_policy_value_error_as_invalid_arguments():
    runner = RecordingRunner()
    runner.default_cwd = "/workspace"

    with pytest.raises(ValueError, match="policy backend changed"):
        asyncio.run(
            ExecCommandTool(policy=_FailingCommandPolicy()).run(
                ToolContext(session_id="sess_1", runner=runner),
                {"argv": ["pwd"]},
            )
        )

    assert runner.command is None


class _FailingExecRunner(RecordingRunner):
    async def exec(self, *args, **kwargs) -> ExecResult:
        del args, kwargs
        raise ValueError("runner state changed")


def test_exec_command_tool_does_not_classify_runner_value_error_as_invalid_arguments():
    runner = _FailingExecRunner()

    with pytest.raises(ValueError, match="runner state changed"):
        asyncio.run(
            ExecCommandTool().run(
                ToolContext(session_id="sess_1", runner=runner),
                {"argv": ["pwd"]},
            )
        )


class _FailingWriteWorkspace(LocalWorkspace):
    async def write_bytes(self, path: str, content: bytes) -> None:
        del path, content
        raise ValueError("workspace backend changed")


def test_write_file_tool_does_not_classify_workspace_value_error_as_invalid_arguments(tmp_path):
    workspace = _FailingWriteWorkspace(tmp_path, workspace_id="failing-write")

    with pytest.raises(ValueError, match="workspace backend changed"):
        asyncio.run(
            WriteFileTool().run(
                ToolContext(session_id="sess_1", workspace=workspace),
                {"path": "notes.txt", "content": "hello"},
            )
        )

    assert (tmp_path / "notes.txt").exists() is False


@pytest.mark.parametrize("requested_cwd", ["../etc", "/etc"])
def test_exec_command_tool_rejects_cwd_outside_runner_root_before_policy_or_exec(
    requested_cwd,
):
    runner = RecordingRunner()
    runner.default_cwd = "/workspace"
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))

    result = asyncio.run(
        ExecCommandTool(policy=policy).run(
            ToolContext(session_id="sess_1", runner=runner),
            {"argv": ["pwd"], "cwd": requested_cwd},
        )
    )

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert policy.requests == []
    assert runner.command is None


def test_exec_command_tool_rejects_blank_cwd_before_policy_or_exec():
    runner = RecordingRunner()
    runner.default_cwd = "/workspace"
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))

    result = asyncio.run(
        ExecCommandTool(policy=policy).run(
            ToolContext(session_id="sess_1", runner=runner),
            {"argv": ["pwd"], "cwd": "   "},
        )
    )

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert policy.requests == []
    assert runner.command is None


def test_exec_command_tool_rejects_initial_invalid_cwd_without_policy():
    runner = RecordingRunner()
    runner.default_cwd = "/workspace"

    result = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="sess_1", runner=runner),
            {"argv": ["pwd"], "cwd": "../etc"},
        )
    )

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert runner.command is None


@pytest.mark.parametrize(
    ("requested_cwd", "error_type", "error_match"),
    [
        ("missing", FileNotFoundError, "does not exist"),
        ("not-a-directory", NotADirectoryError, "not a directory"),
    ],
)
def test_exec_command_tool_preserves_local_cwd_resolution_errors_before_policy_or_exec(
    tmp_path,
    requested_cwd,
    error_type,
    error_match,
):
    runner = LocalRunner(tmp_path)
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))
    marker = tmp_path / "executed"
    if error_type is NotADirectoryError:
        (tmp_path / requested_cwd).write_text("not a directory", encoding="utf-8")

    with pytest.raises(error_type, match=error_match):
        asyncio.run(
            ExecCommandTool(policy=policy).run(
                ToolContext(session_id="sess_1", runner=runner),
                {
                    "argv": [
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(marker)!r}).touch()",
                    ],
                    "cwd": requested_cwd,
                },
            )
        )

    assert policy.requests == []
    assert marker.exists() is False


def test_exec_command_tool_policy_deny_blocks_runner():
    runner = RecordingRunner()
    runner.default_cwd = "/workspace"
    policy = _StaticCommandPolicy(
        CommandPolicyResult(
            decision=CommandPolicyDecision.DENY,
            reason="Shell scripts are not allowed here.",
        )
    )

    result = asyncio.run(
        ExecCommandTool(policy=policy).run(
            ToolContext(session_id="sess_1", runner=runner),
            {"shell": "rm -rf /", "cwd": "repo"},
        )
    )

    assert result.is_error is True
    assert result.content == ("Command denied by policy. Shell scripts are not allowed here.")
    assert result.structured == {
        "error": "command_denied",
        "decision": "deny",
        "reason": "Shell scripts are not allowed here.",
    }
    assert policy.requests[0][1].cwd == "repo"
    assert policy.requests[0][1].canonical_cwd == "/workspace/repo"
    assert runner.command is None


def test_command_request_loads_policy_metadata_without_canonical_cwd():
    request = CommandRequest.model_validate(
        {
            "command": {"kind": "process", "argv": ["pwd"]},
            "cwd": "repo",
            "timeout_s": 5,
        }
    )

    assert request.cwd == "repo"
    assert request.canonical_cwd is None


def test_exec_command_tool_policy_require_approval_blocks_runner():
    runner = RecordingRunner()
    policy = _StaticCommandPolicy(
        CommandPolicyResult(decision=CommandPolicyDecision.REQUIRE_COMMAND_APPROVAL)
    )

    result = asyncio.run(
        ExecCommandTool(policy=policy).run(
            ToolContext(session_id="sess_1", runner=runner),
            {"argv": ["curl", "https://example.com"]},
        )
    )

    assert result.is_error is True
    assert result.content == "Command requires approval before it can run."
    assert result.structured == {
        "error": "command_approval_required",
        "decision": "require_command_approval",
        "reason": None,
    }
    assert runner.command is None


def test_command_approval_member_is_distinct_from_tool_policy():
    # #125 footgun 2: the command-policy approval member must NOT share a name OR a bare string with
    # the tool-policy one. The tool-policy member creates a durable pause/resume checkpoint; the
    # command-policy one only refuses the command inline (no session pause).
    from cayu.runtime import ToolPolicyDecision

    assert CommandPolicyDecision.REQUIRE_COMMAND_APPROVAL != ToolPolicyDecision.REQUIRE_APPROVAL
    assert str(CommandPolicyDecision.REQUIRE_COMMAND_APPROVAL) == "require_command_approval"
    assert str(ToolPolicyDecision.REQUIRE_APPROVAL) == "require_approval"


def test_exec_command_tool_rejects_invalid_policy_wiring():
    with pytest.raises(TypeError, match="must implement CommandPolicy"):
        ExecCommandTool(policy=object())  # type: ignore[arg-type]

    class _WrongResultPolicy(CommandPolicy):
        async def evaluate(self, ctx: ToolContext, request: CommandRequest):
            return "allow"

    with pytest.raises(TypeError, match="must return a CommandPolicyResult"):
        asyncio.run(
            ExecCommandTool(policy=_WrongResultPolicy()).run(
                ToolContext(session_id="sess_1", runner=RecordingRunner()),
                {"argv": ["echo", "hi"]},
            )
        )


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


def test_runtime_executes_in_the_local_canonical_cwd_authorized_by_policy(tmp_path):
    work = tmp_path / "repo"
    work.mkdir()
    runner = LocalRunner(tmp_path)
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="exec_command",
                    arguments={
                        "argv": [
                            sys.executable,
                            "-c",
                            "import os; print(os.getcwd())",
                        ],
                        "cwd": str(work),
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local-dev"), runner=runner),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ExecCommandTool(policy=policy)],
    )

    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "print the working directory")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    request = policy.requests[0][1]
    assert request.cwd == str(work)
    assert request.canonical_cwd == str(work)
    completed = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert completed.payload["result"]["structured"]["stdout"].strip() == str(work)


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
    encoded_content = provider.requests[1].options[RESOLVED_FILE_ATTACHMENTS_OPTION]["art_large"][
        "data_base64"
    ]
    assert len(base64.b64decode(encoded_content)) == size_bytes


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
