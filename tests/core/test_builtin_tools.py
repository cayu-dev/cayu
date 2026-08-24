from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import sys
import threading
import tracemalloc
from collections.abc import AsyncIterator
from importlib import import_module

import pytest
from pydantic import SecretStr

import cayu.tools.files as files_module
from cayu import (
    DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
    DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST,
    DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES,
    REDACTED_SECRET,
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    ArtifactMetadata,
    ArtifactReadResult,
    ArtifactScope,
    ArtifactStore,
    Environment,
    EnvironmentSpec,
    SecretRedactor,
    file_attachment,
)
from cayu.artifacts import ArtifactStoreUnavailableError, LocalArtifactStore
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.tools import (
    _POLICY_DENIAL_TEXT_MAX_BYTES,
    _POLICY_DENIAL_TRUNCATION_MARKER,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    _bound_policy_denial_text,
)
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.runners import (
    DockerRunner,
    ExecCommand,
    ExecResult,
    LocalRunner,
    Runner,
    RunnerExecutionError,
    RunnerUnavailableError,
)
from cayu.runtime import (
    AfterToolCallDecision,
    CayuApp,
    RunRequest,
    RuntimeHook,
    ToolCallHookContext,
)
from cayu.runtime._invocation_secrets import InvocationSecretTracker
from cayu.runtime._model_completion_publication import (
    LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    model_step_publication_from_checkpoint,
)
from cayu.runtime._tool_execution import run_tool
from cayu.runtime.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.tools import ExecCommandTool
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._resources import InvocationArtifactStoreHandle, InvocationWorkspaceHandle
from cayu.tools._runner import InvocationRunnerHandle
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
    DeleteFileTool,
    EditFileTool,
    ListArtifactsTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from cayu.vaults import ResolvedSecret, SecretRef, StaticVault
from cayu.workspaces import LocalWorkspace, WorkspaceReadResult

TINY_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00"
    b"\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)
MISSING_ARTIFACT_ID = f"art_{'0' * 32}"


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


async def collect_events(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


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


class FailingReadArtifactStore(ArtifactStore):
    id = "failing-read"

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def put_bytes(self, content: bytes, **kwargs):
        raise NotImplementedError

    async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
        del artifact_id, max_bytes
        raise self.error

    async def list(self, **kwargs):
        raise NotImplementedError

    async def delete(self, artifact_id: str) -> None:
        raise NotImplementedError


class DisappearingRereadArtifactStore(ArtifactStore):
    id = "disappearing-reread"

    def __init__(
        self,
        *,
        artifact_id: str,
        filename: str,
        content_type: str,
        content: bytes,
        error: FileNotFoundError,
        session_id: str = "sess_1",
    ) -> None:
        self.artifact_id = artifact_id
        self.filename = filename
        self.content_type = content_type
        self.content = content
        self.error = error
        self.session_id = session_id
        self.read_count = 0

    async def put_bytes(self, content: bytes, **kwargs):
        raise NotImplementedError

    async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
        assert artifact_id == self.artifact_id
        self.read_count += 1
        if self.read_count > 1:
            raise self.error
        content = self.content if max_bytes is None else self.content[:max_bytes]
        return ArtifactReadResult(
            metadata=ArtifactMetadata(
                id=self.artifact_id,
                filename=self.filename,
                content_type=self.content_type,
                size_bytes=len(self.content),
                session_id=self.session_id,
            ),
            content=content,
            total_bytes=len(self.content),
            truncated=len(content) < len(self.content),
        )

    async def list(self, **kwargs):
        raise NotImplementedError

    async def delete(self, artifact_id: str) -> None:
        raise NotImplementedError


class VanishingPublishedArtifactStore(FailingReadArtifactStore):
    async def put_bytes(self, content: bytes, **kwargs):
        return ArtifactMetadata(
            id=MISSING_ARTIFACT_ID,
            filename=kwargs["filename"],
            content_type=kwargs["content_type"],
            size_bytes=len(content),
            scope=kwargs["scope"],
            session_id=kwargs.get("session_id"),
            agent_name=kwargs.get("agent_name"),
            environment_name=kwargs.get("environment_name"),
            metadata=kwargs.get("metadata", {}),
        )


class BlockingReadArtifactStore(ArtifactStore):
    id = "blocking-read"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def put_bytes(self, content: bytes, **kwargs):
        raise NotImplementedError

    async def read_bytes(self, artifact_id: str, *, max_bytes: int | None = None):
        del artifact_id, max_bytes
        self.started.set()
        await self.release.wait()
        raise AssertionError("cancelled artifact read unexpectedly resumed")

    async def list(self, **kwargs):
        raise NotImplementedError

    async def delete(self, artifact_id: str) -> None:
        raise NotImplementedError


class BoundedReadWorkspace(LocalWorkspace):
    def __init__(
        self,
        root,
        *,
        cap: int | None = None,
        bound_results: list[object] | None = None,
    ) -> None:
        super().__init__(root, workspace_id="bounded")
        self.cap = cap
        self.bound_results = list(bound_results or ())
        self.bound_requests: list[int] = []
        self.read_requests: list[tuple[int, int | None]] = []

    def bounded_read_limit(self, max_bytes: int) -> int:
        self.bound_requests.append(max_bytes)
        if self.bound_results:
            return self.bound_results.pop(0)  # type: ignore[return-value]
        if self.cap is None:
            return max_bytes
        return min(max_bytes, self.cap)

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        self.read_requests.append((offset, max_bytes))
        return await super().read_bytes(path, offset=offset, max_bytes=max_bytes)


class RecordingLocalArtifactStore(LocalArtifactStore):
    def __init__(self, root) -> None:
        super().__init__(root, store_id="recording")
        self.put_count = 0

    async def put_bytes(self, content: bytes, **kwargs):
        self.put_count += 1
        return await super().put_bytes(content, **kwargs)


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
        WriteFileTool().run(ctx, {"path": "notes/result.txt", "content": "hello", "mode": "create"})
    )
    read_result = asyncio.run(ReadFileTool().run(ctx, {"path": "notes/result.txt"}))
    list_result = asyncio.run(ListFilesTool().run(ctx, {"pattern": "**/*.txt"}))

    assert write_result.is_error is False
    assert write_result.structured == {
        "path": "notes/result.txt",
        "bytes": 5,
        "encoding": "utf-8",
        "mode": "create",
        "revision": f"sha256:{hashlib.sha256(b'hello').hexdigest()}",
        "sha256": hashlib.sha256(b"hello").hexdigest(),
    }
    assert read_result.content.endswith("[/read_file metadata]\nhello")
    assert read_result.structured == {
        "source": "workspace",
        "path": "notes/result.txt",
        "bytes": 5,
        "total_bytes": 5,
        "offset": 0,
        "next_offset": None,
        "revision": f"sha256:{hashlib.sha256(b'hello').hexdigest()}",
        "sha256": hashlib.sha256(b"hello").hexdigest(),
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


def test_read_file_exposes_complete_revision_to_the_model(tmp_path):
    content = b"hello"
    (tmp_path / "notes.txt").write_bytes(content)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
    )

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt"}))

    revision = f"sha256:{hashlib.sha256(content).hexdigest()}"
    assert result.content == (
        "[read_file metadata]\n"
        f'{{"path":"notes.txt","bytes":5,"total_bytes":5,"offset":0,'
        f'"next_offset":null,"revision":"{revision}","sha256":"{hashlib.sha256(content).hexdigest()}",'
        '"truncated":false}\n'
        "[/read_file metadata]\n"
        "hello"
    )


def test_edit_file_applies_multiple_exact_replacements_atomically(tmp_path):
    path = tmp_path / "notes.txt"
    original = b"alpha = 1\nbeta = 2\ngamma = 3\n"
    path.write_bytes(original)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
    )

    result = asyncio.run(
        EditFileTool().run(
            ctx,
            {
                "path": "notes.txt",
                "expected_revision": f"sha256:{hashlib.sha256(original).hexdigest()}",
                "edits": [
                    {
                        "old_text": "alpha = 1",
                        "new_text": "alpha = 10",
                    },
                    {
                        "old_text": "gamma = 3",
                        "new_text": "gamma = 30",
                    },
                ],
            },
        )
    )

    updated = b"alpha = 10\nbeta = 2\ngamma = 30\n"
    assert result.is_error is False
    assert path.read_bytes() == updated
    assert result.structured["path"] == "notes.txt"
    assert result.structured["edit_count"] == 2
    assert result.structured["replacement_count"] == 2
    assert result.structured["before_sha256"] == hashlib.sha256(original).hexdigest()
    assert result.structured["after_sha256"] == hashlib.sha256(updated).hexdigest()
    assert result.structured["before_bytes"] == len(original)
    assert result.structured["after_bytes"] == len(updated)
    assert result.structured["diff_truncated"] is False
    assert "-alpha = 1" in result.structured["diff"]
    assert "+gamma = 30" in result.structured["diff"]


def test_edit_file_redacts_complete_diff_before_bounding_it(tmp_path):
    secret = "edit-diff-secret-canary-ABCDEFGHIJKLMNOP"
    path = tmp_path / "notes.txt"
    original = (("x" * 165) + secret + "\ntarget = old\n").encode()
    path.write_bytes(original)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    result = asyncio.run(
        EditFileTool().run(
            ctx,
            {
                "path": "notes.txt",
                "expected_revision": f"sha256:{hashlib.sha256(original).hexdigest()}",
                "edits": [{"old_text": "target = old", "new_text": "target = new"}],
                "max_diff_bytes": 256,
            },
        )
    )

    rendered = json.dumps(result.model_dump(mode="json"))
    assert result.is_error is False
    assert result.structured["diff_truncated"] is True
    assert secret not in rendered
    assert not any(secret[:size] in rendered for size in range(8, len(secret) + 1))
    assert REDACTED_SECRET in rendered
    assert path.read_bytes() == (("x" * 165) + secret + "\ntarget = new\n").encode()


def test_edit_file_rolls_back_all_replacements_when_one_precondition_fails(tmp_path):
    path = tmp_path / "notes.txt"
    original = b"alpha = 1\nbeta = 2\n"
    path.write_bytes(original)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
    )

    result = asyncio.run(
        EditFileTool().run(
            ctx,
            {
                "path": "notes.txt",
                "expected_revision": f"sha256:{hashlib.sha256(original).hexdigest()}",
                "edits": [
                    {
                        "old_text": "alpha = 1",
                        "new_text": "alpha = 10",
                    },
                    {
                        "old_text": "missing = 3",
                        "new_text": "missing = 30",
                    },
                ],
            },
        )
    )

    assert result.is_error is True
    assert result.structured == {
        "path": "notes.txt",
        "edit_index": 1,
        "expected_replacements": 1,
        "actual_replacements": 0,
    }
    assert path.read_bytes() == original


def test_edit_file_rejects_overlapping_original_snapshot_matches(tmp_path):
    path = tmp_path / "notes.txt"
    original = b"abcdef\n"
    path.write_bytes(original)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
    )

    result = asyncio.run(
        EditFileTool().run(
            ctx,
            {
                "path": "notes.txt",
                "expected_revision": f"sha256:{hashlib.sha256(original).hexdigest()}",
                "edits": [
                    {"old_text": "abc", "new_text": "ABC"},
                    {"old_text": "bc", "new_text": "BC"},
                ],
            },
        )
    )

    assert result.is_error is True
    assert result.structured["reason"] == "overlapping_edits"
    assert path.read_bytes() == original


def test_edit_file_refuses_concurrent_change_at_conditional_mutation(tmp_path):
    class RacingWorkspace(LocalWorkspace):
        async def replace_bytes(self, path, content, *, expected_revision):
            await self.write_bytes(path, b"concurrent\n")
            return await super().replace_bytes(
                path,
                content,
                expected_revision=expected_revision,
            )

    path = tmp_path / "notes.txt"
    original = b"before\n"
    path.write_bytes(original)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=RacingWorkspace(tmp_path, workspace_id="local"),
    )

    result = asyncio.run(
        EditFileTool().run(
            ctx,
            {
                "path": "notes.txt",
                "expected_revision": f"sha256:{hashlib.sha256(original).hexdigest()}",
                "edits": [{"old_text": "before", "new_text": "after"}],
            },
        )
    )

    assert result.is_error is True
    assert result.structured["reason"] == "stale_content"
    assert path.read_bytes() == b"concurrent\n"


def test_edit_file_rejects_amplified_result_before_constructing_it(tmp_path):
    path = tmp_path / "notes.txt"
    original = ("x" * 1000).encode()
    path.write_bytes(original)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
    )

    tracemalloc.start()
    result = asyncio.run(
        EditFileTool().run(
            ctx,
            {
                "path": "notes.txt",
                "expected_revision": f"sha256:{hashlib.sha256(original).hexdigest()}",
                "edits": [
                    {
                        "old_text": "x",
                        "new_text": "y" * 50_000,
                        "expected_replacements": 1000,
                    }
                ],
                "max_bytes": MAX_WRITE_LIMIT_BYTES,
            },
        )
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert result.is_error is True
    assert result.structured["reason"] == "result_too_large"
    assert result.structured["after_bytes"] == 50_000_000
    assert peak < 10 * 1024 * 1024
    assert path.read_bytes() == original


def test_delete_file_requires_the_current_content_digest(tmp_path):
    path = tmp_path / "obsolete.txt"
    content = b"remove me\n"
    path.write_bytes(content)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
    )

    stale = asyncio.run(
        DeleteFileTool().run(
            ctx,
            {
                "path": "obsolete.txt",
                "expected_revision": f"sha256:{'0' * 64}",
            },
        )
    )
    deleted = asyncio.run(
        DeleteFileTool().run(
            ctx,
            {
                "path": "obsolete.txt",
                "expected_revision": f"sha256:{hashlib.sha256(content).hexdigest()}",
            },
        )
    )

    assert stale.is_error is True
    assert stale.structured["actual_revision"] == (f"sha256:{hashlib.sha256(content).hexdigest()}")
    assert deleted.is_error is False
    assert deleted.structured == {
        "path": "obsolete.txt",
        "deleted_bytes": len(content),
        "deleted_sha256": hashlib.sha256(content).hexdigest(),
        "deleted_revision": f"sha256:{hashlib.sha256(content).hexdigest()}",
    }
    assert not path.exists()


def test_write_file_create_and_overwrite_are_conditional(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("before")
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    ctx = ToolContext(session_id="sess_1", workspace=workspace)

    duplicate = asyncio.run(
        WriteFileTool().run(
            ctx,
            {"path": "notes.txt", "content": "duplicate", "mode": "create"},
        )
    )
    revision = asyncio.run(workspace.read_bytes("notes.txt")).revision
    overwritten = asyncio.run(
        WriteFileTool().run(
            ctx,
            {
                "path": "notes.txt",
                "content": "after",
                "mode": "overwrite",
                "expected_revision": revision,
            },
        )
    )

    assert duplicate.is_error is True
    assert duplicate.structured["reason"] == "already_exists"
    assert overwritten.is_error is False
    assert overwritten.structured["mode"] == "overwrite"
    assert path.read_text() == "after"


def test_write_file_overwrite_refuses_missing_and_invalid_revision(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    ctx = ToolContext(session_id="sess_1", workspace=workspace)

    missing = asyncio.run(
        WriteFileTool().run(
            ctx,
            {
                "path": "missing.txt",
                "content": "after",
                "mode": "overwrite",
                "expected_revision": f"sha256:{'0' * 64}",
            },
        )
    )
    invalid = asyncio.run(
        WriteFileTool().run(
            ctx,
            {
                "path": "missing.txt",
                "content": "after",
                "mode": "overwrite",
                "expected_revision": "bad\0revision",
            },
        )
    )

    assert missing.is_error is True
    assert missing.structured["reason"] == "not_found"
    assert invalid.is_error is True
    assert invalid.structured == {"error": "invalid_arguments"}


def test_read_file_returns_not_found_error_for_missing_workspace_text(tmp_path):
    ctx = ToolContext(
        session_id="sess_1",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
    )

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "missing.txt"}))

    assert result.content == "Read refused: workspace file not found: missing.txt."
    assert result.structured == {"path": "missing.txt", "reason": "not_found"}
    assert result.is_error is True


def test_read_file_pages_text_and_only_complete_snapshot_has_revision(tmp_path):
    (tmp_path / "notes.txt").write_text("abcdef")
    ctx = ToolContext(
        session_id="sess_1",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
    )

    first = asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt", "max_bytes": 2}))
    suffix = asyncio.run(
        ReadFileTool().run(
            ctx,
            {"path": "notes.txt", "offset": first.structured["next_offset"], "max_bytes": 10},
        )
    )

    assert '"next_offset":2' in first.content
    assert '"truncated":true' in first.content
    assert first.content.endswith("[/read_file metadata]\nab")
    assert first.structured["offset"] == 0
    assert first.structured["next_offset"] == 2
    assert first.structured["revision"] is None
    assert suffix.content.endswith("[/read_file metadata]\ncdef")
    assert suffix.structured["offset"] == 2
    assert suffix.structured["next_offset"] is None
    assert suffix.structured["revision"] is None


def test_read_file_pages_text_with_workspace_bounded_read_limit(tmp_path):
    (tmp_path / "notes.txt").write_text("abcde")
    workspace = BoundedReadWorkspace(tmp_path, cap=2)
    ctx = ToolContext(session_id="sess_1", workspace=workspace)

    first = asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt", "max_bytes": 5}))
    second = asyncio.run(
        ReadFileTool().run(
            ctx,
            {"path": "notes.txt", "offset": first.structured["next_offset"], "max_bytes": 5},
        )
    )
    third = asyncio.run(
        ReadFileTool().run(
            ctx,
            {"path": "notes.txt", "offset": second.structured["next_offset"], "max_bytes": 5},
        )
    )

    assert workspace.bound_requests == [5, 5, 5]
    assert workspace.read_requests == [(0, 2), (2, 2), (4, 2)]
    assert [page.structured["bytes"] for page in (first, second, third)] == [2, 2, 1]
    assert [page.structured["next_offset"] for page in (first, second, third)] == [2, 4, None]
    assert [page.structured["truncated"] for page in (first, second, third)] == [
        True,
        True,
        False,
    ]
    assert (
        "".join(
            page.content.rsplit("[/read_file metadata]\n", 1)[1] for page in (first, second, third)
        )
        == "abcde"
    )


@pytest.mark.parametrize("invalid_bound", [None, True, 0, -1, 1.5, 6])
def test_read_file_rejects_invalid_workspace_bounded_read_limit(tmp_path, invalid_bound):
    (tmp_path / "notes.txt").write_text("abcde")
    workspace = BoundedReadWorkspace(tmp_path, bound_results=[invalid_bound])
    ctx = ToolContext(session_id="sess_1", workspace=workspace)

    with pytest.raises(
        RuntimeError,
        match=(
            r"BoundedReadWorkspace\.bounded_read_limit\(\) must return a positive integer "
            r"no greater than max_bytes=5\."
        ),
    ):
        asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt", "max_bytes": 5}))

    assert workspace.bound_requests == [5]
    assert workspace.read_requests == []


@pytest.mark.parametrize("offset", [5, 8])
def test_read_file_rejects_workspace_text_offset_past_eof(tmp_path, offset):
    (tmp_path / "tiny.txt").write_text("tiny")
    ctx = ToolContext(
        session_id="sess_1",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
    )

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "tiny.txt", "offset": offset}))

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}


def test_read_file_rejects_bounded_redacted_offset_past_eof(tmp_path):
    (tmp_path / "tiny.txt").write_text("tiny")
    workspace = BoundedReadWorkspace(tmp_path, cap=2)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor("TOKEN"),
    )

    result = asyncio.run(
        ReadFileTool().run(
            ctx,
            {"path": "tiny.txt", "offset": 8, "max_bytes": 5},
        )
    )

    assert workspace.bound_requests == [39]
    assert workspace.read_requests == [(0, 2)]
    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert "secret_redaction_window_unavailable" not in result.content


def test_read_file_rejects_bounded_redacted_split_utf8_offset(tmp_path):
    secret = "TOKEN"
    (tmp_path / "notes.txt").write_text("A€B")
    workspace = BoundedReadWorkspace(tmp_path, cap=3)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    result = asyncio.run(
        ReadFileTool().run(
            ctx,
            {"path": "notes.txt", "offset": 2, "max_bytes": 5},
        )
    )

    assert workspace.bound_requests == [33]
    assert workspace.read_requests == [(0, 3)]
    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert "splits a UTF-8 character" in result.content
    assert secret not in json.dumps(result.model_dump(mode="json"))


def test_read_file_accepts_workspace_text_offset_at_eof(tmp_path):
    (tmp_path / "tiny.txt").write_text("tiny")
    ctx = ToolContext(
        session_id="sess_1",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
    )

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "tiny.txt", "offset": 4}))

    assert result.is_error is False
    assert result.content.endswith("[/read_file metadata]\n")
    assert result.structured["bytes"] == 0
    assert result.structured["total_bytes"] == 4
    assert result.structured["offset"] == 4
    assert result.structured["next_offset"] is None
    assert result.structured["truncated"] is False


def test_read_file_accepts_bounded_workspace_text_offset_at_eof_with_secrets(tmp_path):
    (tmp_path / "tiny.txt").write_text("tiny")
    workspace = BoundedReadWorkspace(tmp_path, cap=2)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor("TOKEN"),
    )

    result = asyncio.run(
        ReadFileTool().run(
            ctx,
            {"path": "tiny.txt", "offset": 4, "max_bytes": 5},
        )
    )

    assert workspace.read_requests == [(0, 2)]
    assert result.is_error is False
    assert result.content.endswith("[/read_file metadata]\n")
    assert result.structured["bytes"] == 0
    assert result.structured["total_bytes"] == 4
    assert result.structured["offset"] == 4
    assert result.structured["next_offset"] is None
    assert result.structured["truncated"] is False


def test_read_file_preserves_unrelated_workspace_value_error(tmp_path):
    class FailingWorkspace(LocalWorkspace):
        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            raise ValueError("backend invariant failed")

    ctx = ToolContext(
        session_id="sess_1",
        workspace=FailingWorkspace(tmp_path, workspace_id="local"),
    )

    with pytest.raises(ValueError, match="backend invariant failed"):
        asyncio.run(ReadFileTool().run(ctx, {"path": "tiny.txt"}))


def test_read_file_pages_preserve_literal_redaction_marker_without_active_secrets(tmp_path):
    content = f"abc{REDACTED_SECRET}xyz"
    (tmp_path / "notes.txt").write_text(content)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
    )
    visible_pages: list[str] = []
    offset = 0

    while True:
        page = asyncio.run(
            ReadFileTool().run(
                ctx,
                {"path": "notes.txt", "offset": offset, "max_bytes": 5},
            )
        )
        visible_pages.append(page.content.rsplit("[/read_file metadata]\n", 1)[1])
        next_offset = page.structured["next_offset"]
        if next_offset is None:
            break
        assert next_offset > offset
        offset = next_offset

    assert "".join(visible_pages) == content


def test_read_file_pages_utf8_only_at_complete_scalar_boundaries(tmp_path):
    (tmp_path / "notes.txt").write_text("A€B")
    ctx = ToolContext(
        session_id="sess_1",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
    )

    first = asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt", "max_bytes": 2}))
    second = asyncio.run(
        ReadFileTool().run(
            ctx,
            {"path": "notes.txt", "offset": first.structured["next_offset"], "max_bytes": 3},
        )
    )
    third = asyncio.run(
        ReadFileTool().run(
            ctx,
            {"path": "notes.txt", "offset": second.structured["next_offset"], "max_bytes": 2},
        )
    )
    split_start = asyncio.run(
        ReadFileTool().run(ctx, {"path": "notes.txt", "offset": 2, "max_bytes": 3})
    )

    assert first.content.endswith("[/read_file metadata]\nA")
    assert first.structured["next_offset"] == 1
    assert second.content.endswith("[/read_file metadata]\n€")
    assert second.structured["next_offset"] == 4
    assert third.content.endswith("[/read_file metadata]\nB")
    assert split_start.is_error is True
    assert split_start.structured == {"error": "invalid_arguments"}


def test_read_file_bounded_pages_preserve_utf8_boundaries(tmp_path):
    (tmp_path / "notes.txt").write_text("A€B")
    workspace = BoundedReadWorkspace(tmp_path, cap=2)
    ctx = ToolContext(session_id="sess_1", workspace=workspace)

    first = asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt", "max_bytes": 5}))
    incomplete = asyncio.run(
        ReadFileTool().run(
            ctx,
            {"path": "notes.txt", "offset": first.structured["next_offset"], "max_bytes": 5},
        )
    )
    complete_workspace = BoundedReadWorkspace(tmp_path, cap=3)
    complete = asyncio.run(
        ReadFileTool().run(
            ToolContext(session_id="sess_1", workspace=complete_workspace),
            {"path": "notes.txt", "offset": 1, "max_bytes": 5},
        )
    )

    assert first.content.endswith("[/read_file metadata]\nA")
    assert first.structured["next_offset"] == 1
    assert incomplete.is_error is True
    assert incomplete.structured == {
        "error": "text_page_too_small",
        "offset": 1,
        "effective_read_limit_bytes": 2,
        "effective_page_limit_bytes": 2,
    }
    assert "workspace read window" in incomplete.content.lower()
    assert "retry with max_bytes" not in incomplete.content
    assert complete.content.endswith("[/read_file metadata]\n€")
    assert complete.structured["next_offset"] == 4
    assert workspace.read_requests == [(0, 2), (1, 2)]
    assert complete_workspace.read_requests == [(1, 3)]


def test_read_file_utf8_error_distinguishes_caller_limit_from_short_workspace_page(tmp_path):
    class ShortReadWorkspace(BoundedReadWorkspace):
        def __init__(self, root, *, short_limit: int = 2, **kwargs) -> None:
            super().__init__(root, **kwargs)
            self.short_limit = short_limit

        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            short_limit = None if max_bytes is None else min(max_bytes, self.short_limit)
            return await super().read_bytes(path, offset=offset, max_bytes=short_limit)

    class ProjectedReadWorkspace(BoundedReadWorkspace):
        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            result = await super().read_bytes(path, offset=offset, max_bytes=max_bytes)
            return WorkspaceReadResult(
                content=b"",
                total_bytes=result.total_bytes,
                truncated=result.truncated,
                offset=result.offset,
                source_bytes_read=result.source_bytes_read,
                redaction_truncated=True,
            )

    (tmp_path / "notes.txt").write_text("€B")
    caller_limited = asyncio.run(
        ReadFileTool().run(
            ToolContext(
                session_id="sess_1",
                workspace=LocalWorkspace(tmp_path, workspace_id="local"),
            ),
            {"path": "notes.txt", "max_bytes": 2},
        )
    )
    short_workspace = ShortReadWorkspace(tmp_path)
    short_page = asyncio.run(
        ReadFileTool().run(
            ToolContext(session_id="sess_1", workspace=short_workspace),
            {"path": "notes.txt", "max_bytes": 5},
        )
    )
    constrained_short_workspace = ShortReadWorkspace(tmp_path, cap=4)
    constrained_short_page = asyncio.run(
        ReadFileTool().run(
            ToolContext(session_id="sess_1", workspace=constrained_short_workspace),
            {"path": "notes.txt", "max_bytes": 5},
        )
    )
    caller_underfilled_workspace = ShortReadWorkspace(tmp_path, short_limit=1)
    caller_underfilled_page = asyncio.run(
        ReadFileTool().run(
            ToolContext(session_id="sess_1", workspace=caller_underfilled_workspace),
            {"path": "notes.txt", "max_bytes": 2},
        )
    )
    projected_workspace = ProjectedReadWorkspace(tmp_path, cap=2)
    projected_page = asyncio.run(
        ReadFileTool().run(
            ToolContext(session_id="sess_1", workspace=projected_workspace),
            {"path": "notes.txt", "max_bytes": 5},
        )
    )

    assert caller_limited.structured == {
        "error": "text_page_too_small",
        "offset": 0,
        "minimum_max_bytes": 4,
    }
    assert "retry with max_bytes" in caller_limited.content
    assert short_workspace.bound_requests == [5]
    assert short_page.structured == {
        "error": "text_page_too_small",
        "offset": 0,
    }
    assert "incomplete UTF-8 character" in short_page.content
    assert "workspace read limit" not in short_page.content.lower()
    assert constrained_short_workspace.bound_requests == [5]
    assert constrained_short_workspace.read_requests == [(0, 2)]
    assert constrained_short_page.structured == short_page.structured
    assert "incomplete UTF-8 character" in constrained_short_page.content
    assert "workspace read limit" not in constrained_short_page.content.lower()
    assert caller_underfilled_workspace.bound_requests == [2]
    assert caller_underfilled_workspace.read_requests == [(0, 1)]
    assert caller_underfilled_page.structured == short_page.structured
    assert "incomplete UTF-8 character" in caller_underfilled_page.content
    assert "retry with max_bytes" not in caller_underfilled_page.content
    assert projected_workspace.bound_requests == [5]
    assert projected_workspace.read_requests == [(0, 2)]
    assert projected_page.structured == short_page.structured
    assert "incomplete UTF-8 character" in projected_page.content
    assert "workspace read limit" not in projected_page.content.lower()


def test_read_file_does_not_blame_limits_for_incomplete_utf8_at_eof(tmp_path):
    (tmp_path / "notes.txt").write_bytes(b"\xe2\x82")
    bounded_workspace = BoundedReadWorkspace(tmp_path, cap=2)

    bounded = asyncio.run(
        ReadFileTool().run(
            ToolContext(session_id="sess_1", workspace=bounded_workspace),
            {"path": "notes.txt", "max_bytes": 5},
        )
    )
    caller_limited = asyncio.run(
        ReadFileTool().run(
            ToolContext(
                session_id="sess_1",
                workspace=LocalWorkspace(tmp_path, workspace_id="local"),
            ),
            {"path": "notes.txt", "max_bytes": 2},
        )
    )

    expected = {
        "error": "text_page_too_small",
        "offset": 0,
    }
    assert bounded_workspace.bound_requests == [5]
    assert bounded_workspace.read_requests == [(0, 2)]
    assert bounded.structured == expected
    assert caller_limited.structured == expected
    for result in (bounded, caller_limited):
        assert "incomplete UTF-8 character" in result.content
        assert "workspace read limit" not in result.content.lower()
        assert "retry with max_bytes" not in result.content


def test_read_file_reports_caller_utf8_limit_when_redaction_prefetch_reaches_eof(tmp_path):
    secret = "TOKEN"
    (tmp_path / "notes.txt").write_text("😀")
    workspace = BoundedReadWorkspace(tmp_path)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt", "max_bytes": 2}))

    assert workspace.bound_requests == [28]
    assert workspace.read_requests == [(0, 28)]
    assert result.structured == {
        "error": "text_page_too_small",
        "offset": 0,
        "minimum_max_bytes": 4,
    }
    assert "retry with max_bytes of at least 4" in result.content
    assert "workspace read limit" not in result.content.lower()
    assert secret not in json.dumps(result.model_dump(mode="json"))


def test_read_file_wrapper_composes_expanded_redaction_read_with_backend_bound(tmp_path):
    secret = "TOKEN"
    (tmp_path / "notes.txt").write_text("abcdef")
    workspace = BoundedReadWorkspace(tmp_path, cap=2)
    tracker = InvocationSecretTracker(SecretRedactor(secret))
    ctx = ToolContext(
        session_id="sess_1",
        workspace=InvocationWorkspaceHandle(
            workspace,
            redactor_snapshot_provider=tracker.snapshot,
            capture_observer=tracker.record_ambiguous_output_capture,
        ),
        invocation_secret_redactor=lambda: tracker.redactor,
    )

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt", "max_bytes": 5}))

    assert workspace.bound_requests == [31, 24]
    assert workspace.read_requests == [(0, 2)]
    assert result.is_error is True
    assert secret not in json.dumps(result.model_dump(mode="json"))


def test_workspace_mutation_observer_failure_does_not_replace_completed_write(tmp_path) -> None:
    observed: list[tuple[str, str]] = []

    def fail_observation(method: str, path: str, result: object) -> None:
        del result
        observed.append((method, path))
        raise ValueError("synthetic direct-evidence failure")

    workspace = InvocationWorkspaceHandle(
        LocalWorkspace(tmp_path),
        redactor_snapshot_provider=lambda: None,
        capture_observer=lambda _captured: None,
        direct_mutation_observer=fail_observation,
    )

    result = asyncio.run(workspace.create_bytes(" leading.txt", b"committed"))

    assert result.operation == "create"
    assert observed == [("create_bytes", " leading.txt")]
    assert (tmp_path / " leading.txt").read_bytes() == b"committed"


def test_read_file_resolves_combined_caller_and_workspace_utf8_limits(tmp_path):
    secret = "TOKEN"
    (tmp_path / "four-byte.txt").write_text(f"😀{'x' * 40}")
    (tmp_path / "three-byte.txt").write_text(f"€{'x' * 40}")
    four_byte_workspace = BoundedReadWorkspace(tmp_path, cap=25)
    three_byte_workspace = BoundedReadWorkspace(tmp_path, cap=25)

    def read(path: str, workspace: BoundedReadWorkspace) -> ToolResult:
        return asyncio.run(
            ReadFileTool().run(
                ToolContext(
                    session_id="sess_1",
                    workspace=workspace,
                    invocation_secret_redactor=lambda: SecretRedactor(secret),
                ),
                {"path": path, "max_bytes": 2},
            )
        )

    four_byte = read("four-byte.txt", four_byte_workspace)
    three_byte = read("three-byte.txt", three_byte_workspace)

    for workspace in (four_byte_workspace, three_byte_workspace):
        assert workspace.bound_requests == [28]
        assert workspace.read_requests == [(0, 25)]
    assert four_byte.structured == {
        "error": "text_page_too_small",
        "offset": 0,
        "effective_read_limit_bytes": 25,
        "effective_page_limit_bytes": 3,
    }
    assert "workspace read window" in four_byte.content.lower()
    assert "retry with max_bytes" not in four_byte.content
    assert three_byte.structured == {
        "error": "text_page_too_small",
        "offset": 0,
        "minimum_max_bytes": 4,
    }
    assert "retry with max_bytes of at least 4" in three_byte.content
    assert "workspace read limit" not in three_byte.content.lower()
    assert secret not in json.dumps(four_byte.model_dump(mode="json"))
    assert secret not in json.dumps(three_byte.model_dump(mode="json"))


def test_read_file_discards_workspace_secret_prefix_after_pretruncated_read(tmp_path):
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    asyncio.run(workspace.write_bytes("secret.txt", secret.encode()))
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    result = asyncio.run(
        ReadFileTool().run(
            ctx,
            {
                "path": "secret.txt",
                "max_bytes": 16,
            },
        )
    )

    rendered = json.dumps(result.model_dump(mode="json"))
    assert secret not in rendered
    assert secret[:16] not in rendered
    assert '"truncated":true' in result.content
    assert result.content.endswith("[/read_file metadata]\n")
    assert result.structured["bytes"] == 16
    assert result.structured["total_bytes"] == len(secret.encode())
    assert result.structured["truncated"] is True


def test_read_file_omits_secret_suffix_from_noninitial_workspace_page(tmp_path):
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    asyncio.run(workspace.write_bytes("secret.txt", secret.encode()))
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    result = asyncio.run(
        ReadFileTool().run(
            ctx,
            {
                "path": "secret.txt",
                "offset": 16,
                "max_bytes": 16,
            },
        )
    )

    rendered = json.dumps(result.model_dump(mode="json"))
    assert secret[16:32] not in rendered
    assert result.content.endswith("[/read_file metadata]\n")
    assert result.structured["offset"] == 16
    assert result.structured["truncated"] is True


def test_read_file_preserves_unrelated_continuation_pages_with_active_secret(tmp_path):
    secret = "unrelated-workload-secret-canary-ABCDEFGHIJKLMNOP"
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    asyncio.run(workspace.write_bytes("notes.txt", b"abcdef"))
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    first = asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt", "max_bytes": 2}))
    second = asyncio.run(
        ReadFileTool().run(
            ctx,
            {
                "path": "notes.txt",
                "offset": first.structured["next_offset"],
                "max_bytes": 2,
            },
        )
    )
    third = asyncio.run(
        ReadFileTool().run(
            ctx,
            {
                "path": "notes.txt",
                "offset": second.structured["next_offset"],
                "max_bytes": 2,
            },
        )
    )

    assert first.content.endswith("[/read_file metadata]\nab")
    assert second.content.endswith("[/read_file metadata]\ncd")
    assert third.content.endswith("[/read_file metadata]\nef")
    assert (first.structured["next_offset"], second.structured["next_offset"]) == (2, 4)
    assert third.structured["next_offset"] is None


@pytest.mark.parametrize(
    "revisioned_provider",
    [False, True],
    ids=["legacy-redactor", "revisioned-redactor"],
)
def test_read_file_retries_with_latest_redactor_after_secret_resolves_during_read(
    tmp_path,
    revisioned_provider: bool,
) -> None:
    secret = "workspace-read-race-secret-canary-ABCDEFGHIJKLMNOP"
    read_started = asyncio.Event()
    allow_read = asyncio.Event()

    class BlockingWorkspace(LocalWorkspace):
        read_calls = 0

        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            self.read_calls += 1
            if self.read_calls == 1:
                read_started.set()
                await allow_read.wait()
            return await super().read_bytes(path, offset=offset, max_bytes=max_bytes)

    workspace = BlockingWorkspace(tmp_path, workspace_id="blocking")
    asyncio.run(workspace.write_bytes("secret.txt", secret.encode()))
    tracker = InvocationSecretTracker(SecretRedactor())
    ctx = ToolContext(
        session_id="sess_read_revision",
        workspace=workspace,
        invocation_secret_redactor=lambda: tracker.redactor,
        invocation_secret_snapshot_provider=(tracker.snapshot if revisioned_provider else None),
        invocation_secret_capture_observer=(
            tracker.record_ambiguous_output_capture if revisioned_provider else None
        ),
    )

    async def run() -> ToolResult:
        task = asyncio.create_task(
            ReadFileTool().run(
                ctx,
                {"path": "secret.txt", "max_bytes": 16},
            )
        )
        await read_started.wait()
        tracker.record(
            ResolvedSecret(
                name="token",
                value=SecretStr(secret),
            )
        )
        allow_read.set()
        return await task

    result = asyncio.run(run())
    publication = tracker.seal_for_publication()

    rendered = repr(result.model_dump(mode="json"))
    assert workspace.read_calls == 2
    assert publication.unsafe_output is False
    assert secret not in rendered
    assert secret[:16] not in rendered


def test_read_file_capture_fails_closed_when_secret_resolves_before_publication(
    tmp_path,
) -> None:
    secret = "workspace-late-publication-secret-canary-ABCDEFGHIJKLMNOP"
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    asyncio.run(workspace.write_bytes("secret.txt", secret.encode()))
    tracker = InvocationSecretTracker(SecretRedactor())
    ctx = ToolContext(
        session_id="sess_read_late_publication",
        workspace=workspace,
        invocation_secret_redactor=lambda: tracker.redactor,
        invocation_secret_snapshot_provider=tracker.snapshot,
        invocation_secret_capture_observer=tracker.record_ambiguous_output_capture,
    )

    class ReadThenResolveTool(Tool):
        spec = ToolSpec(
            name="read_then_resolve",
            description="Exercise the bounded read publication boundary.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, tool_ctx: ToolContext, args: dict) -> ToolResult:
            del args
            result = await ReadFileTool().run(
                tool_ctx,
                {"path": "secret.txt", "max_bytes": 16},
            )
            tracker.record(
                ResolvedSecret(
                    name="token",
                    value=SecretStr(secret),
                )
            )
            return result

    outcome = asyncio.run(
        run_tool(
            tool=ReadThenResolveTool(),
            effect=ToolEffect.NONE,
            ctx=ctx,
            arguments={},
            redactor=lambda: tracker.redactor,
            finalize_publication=tracker.seal_for_publication,
        )
    )

    rendered = repr(outcome)
    assert outcome.result.is_error is True
    assert outcome.result.structured["terminal_outcome"] == "invalid_tool_output"
    assert secret not in rendered
    assert secret[:16] not in rendered


def test_read_file_fails_closed_after_repeated_secret_revision_changes(tmp_path) -> None:
    tracker = InvocationSecretTracker(SecretRedactor())

    class ChangingWorkspace(LocalWorkspace):
        read_calls = 0

        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            result = await super().read_bytes(path, offset=offset, max_bytes=max_bytes)
            self.read_calls += 1
            tracker.record(
                ResolvedSecret(
                    name=f"token_{self.read_calls}",
                    value=SecretStr(f"changing-secret-{self.read_calls}-ABCDEFGHIJKLMNOP"),
                )
            )
            return result

    workspace = ChangingWorkspace(tmp_path, workspace_id="changing")
    asyncio.run(workspace.write_bytes("notes.txt", b"ordinary text"))
    result = asyncio.run(
        ReadFileTool().run(
            ToolContext(
                session_id="sess_read_unstable",
                workspace=workspace,
                invocation_secret_redactor=lambda: tracker.redactor,
                invocation_secret_snapshot_provider=tracker.snapshot,
                invocation_secret_capture_observer=(tracker.record_ambiguous_output_capture),
            ),
            {"path": "notes.txt", "max_bytes": 4},
        )
    )

    assert workspace.read_calls == 2
    assert result.is_error is True
    assert result.structured == {"error": "secret_redaction_scope_unstable"}


def test_read_file_secret_crossing_page_boundary_omits_only_secret_remainder(tmp_path):
    secret = "boundary-secret-canary"
    content = ("A" * 15 + secret + "visible-tail").encode()
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    asyncio.run(workspace.write_bytes("notes.txt", content))
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    first = asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt", "max_bytes": 20}))
    pages = [first]
    while pages[-1].structured["next_offset"] is not None:
        pages.append(
            asyncio.run(
                ReadFileTool().run(
                    ctx,
                    {
                        "path": "notes.txt",
                        "offset": pages[-1].structured["next_offset"],
                        "max_bytes": 20,
                    },
                )
            )
        )

    rendered = json.dumps(
        [page.model_dump(mode="json") for page in pages],
        ensure_ascii=False,
    )
    assert secret not in rendered
    assert secret[:5] not in rendered
    assert secret[5:] not in rendered
    assert first.content.endswith("[/read_file metadata]\n" + "A" * 15)
    visible_text = "".join(page.content.rsplit("[/read_file metadata]\n", 1)[1] for page in pages)
    assert visible_text == "A" * 15 + "visible-tail"
    assert first.structured["next_offset"] == 20
    assert pages[-1].structured["next_offset"] is None


def test_read_file_pages_cannot_reconstruct_redacted_secret_when_concatenated(tmp_path):
    secret = "CE"
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    asyncio.run(workspace.write_bytes("notes.txt", b"CCEE"))
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    first = asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt", "max_bytes": 2}))
    second = asyncio.run(
        ReadFileTool().run(
            ctx,
            {
                "path": "notes.txt",
                "offset": first.structured["next_offset"],
                "max_bytes": 2,
            },
        )
    )
    visible_text = "".join(
        page.content.rsplit("[/read_file metadata]\n", 1)[1] for page in (first, second)
    )

    assert visible_text == "E"
    assert secret not in visible_text
    assert first.structured["next_offset"] == 2
    assert second.structured["next_offset"] is None


def test_read_file_redacted_pagination_preserves_utf8_boundaries_and_progress(tmp_path):
    secret = "密钥值-boundary-secret"
    content = ("αβ" + secret + "終わり").encode()
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    asyncio.run(workspace.write_bytes("notes.txt", content))
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    offset = 0
    seen_offsets: list[int] = []
    rendered_pages: list[str] = []
    while True:
        page = asyncio.run(
            ReadFileTool().run(
                ctx,
                {"path": "notes.txt", "offset": offset, "max_bytes": 8},
            )
        )
        assert page.is_error is False
        rendered_pages.append(page.content)
        next_offset = page.structured["next_offset"]
        if next_offset is None:
            break
        assert next_offset > offset
        seen_offsets.append(next_offset)
        offset = next_offset

    rendered = json.dumps(rendered_pages, ensure_ascii=False)
    assert secret not in rendered
    assert seen_offsets == sorted(set(seen_offsets))
    visible_text = "".join(page.rsplit("[/read_file metadata]\n", 1)[1] for page in rendered_pages)
    assert visible_text == "αβ終わり"


def test_read_file_shortens_bounded_page_to_preserve_secret_overlap(tmp_path):
    secret = "TOKEN"
    (tmp_path / "notes.txt").write_text(f"abc{secret}{'x' * 40}")
    workspace = BoundedReadWorkspace(tmp_path, cap=25)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt", "max_bytes": 5}))

    assert workspace.bound_requests == [31]
    assert workspace.read_requests == [(0, 25)]
    assert result.is_error is False
    assert result.content.endswith("[/read_file metadata]\nabc")
    assert result.structured["bytes"] == 3
    assert result.structured["next_offset"] == 3
    assert result.structured["truncated"] is True
    assert secret not in json.dumps(result.model_dump(mode="json"))


def test_read_file_reports_page_capacity_after_secret_overlap(tmp_path):
    secret = "TOKEN"
    (tmp_path / "notes.txt").write_text(f"😀{'x' * 40}")
    workspace = BoundedReadWorkspace(tmp_path, cap=25)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt", "max_bytes": 5}))

    assert workspace.bound_requests == [31]
    assert workspace.read_requests == [(0, 25)]
    assert result.is_error is True
    assert result.structured == {
        "error": "text_page_too_small",
        "offset": 0,
        "effective_read_limit_bytes": 25,
        "effective_page_limit_bytes": 3,
    }
    assert "workspace read window" in result.content.lower()
    assert "safely framed UTF-8 character" in result.content
    assert "retry with max_bytes" not in result.content
    assert secret not in json.dumps(result.model_dump(mode="json"))


def test_read_file_fails_closed_when_bound_cannot_preserve_secret_overlap(tmp_path):
    secret = "TOKEN"
    (tmp_path / "notes.txt").write_text(f"{secret}{'x' * 40}")
    workspace = BoundedReadWorkspace(tmp_path, cap=22)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt", "max_bytes": 5}))

    assert workspace.bound_requests == [31]
    assert workspace.read_requests == [(0, 22)]
    assert result.is_error is True
    assert result.structured == {
        "error": "secret_redaction_window_unavailable",
        "offset": 0,
        "required_window_end": 23,
        "observed_window_end": 22,
    }
    assert secret not in json.dumps(result.model_dump(mode="json"))


def test_read_file_fails_closed_when_backend_omits_required_redaction_prefix(tmp_path):
    class IncompleteWindowWorkspace(LocalWorkspace):
        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            result = await super().read_bytes(path, offset=offset, max_bytes=max_bytes)
            if offset == 0 or not result.content:
                return result
            shifted_content = result.content[1:]
            return WorkspaceReadResult(
                content=shifted_content,
                total_bytes=result.total_bytes,
                truncated=result.offset + 1 + len(shifted_content) < result.total_bytes,
                offset=result.offset + 1,
            )

    workspace = IncompleteWindowWorkspace(tmp_path, workspace_id="incomplete")
    asyncio.run(workspace.write_bytes("notes.txt", b"x" * 80))
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor("token"),
    )

    result = asyncio.run(
        ReadFileTool().run(
            ctx,
            {"path": "notes.txt", "offset": 40, "max_bytes": 8},
        )
    )

    assert result.is_error is True
    assert result.structured == {
        "error": "secret_redaction_window_unavailable",
        "offset": 40,
        "required_window_start": 15,
        "observed_window_start": 16,
    }


def test_read_file_omits_sensitive_page_suffix_across_workspace_replacement(tmp_path):
    secret = "SECRET"
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    asyncio.run(workspace.write_bytes("notes.txt", b"SECREaaaaa"))
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    first = asyncio.run(
        ReadFileTool().run(
            ctx,
            {"path": "notes.txt", "max_bytes": 5},
        )
    )
    asyncio.run(workspace.write_bytes("notes.txt", b"aaaaaTbbbb"))
    second = asyncio.run(
        ReadFileTool().run(
            ctx,
            {"path": "notes.txt", "offset": 5, "max_bytes": 5},
        )
    )

    first_text = first.content.rsplit("[/read_file metadata]\n", 1)[1]
    second_text = second.content.rsplit("[/read_file metadata]\n", 1)[1]
    assert first_text == ""
    assert second_text == "Tbbbb"
    assert secret not in first_text + second_text
    assert first.structured["next_offset"] == 5
    assert second.structured["next_offset"] is None


def test_read_file_discards_text_artifact_secret_prefix_after_pretruncated_read(tmp_path):
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact = asyncio.run(
        artifact_store.put_bytes(
            secret.encode(),
            filename="secret.txt",
            content_type="text/plain",
            session_id="sess_1",
        )
    )
    ctx = ToolContext(
        session_id="sess_1",
        artifact_store=artifact_store,
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )

    result = asyncio.run(
        ReadFileTool().run(
            ctx,
            {
                "artifact_id": artifact.id,
                "max_bytes": 16,
            },
        )
    )

    rendered = json.dumps(result.model_dump(mode="json"))
    assert secret not in rendered
    assert secret[:16] not in rendered
    assert "[file truncated]" in result.content
    assert result.structured["bytes"] == 16
    assert result.structured["total_bytes"] == len(secret.encode())
    assert result.structured["truncated"] is True


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


def test_read_file_rejects_attachment_offset_before_workspace_read(tmp_path):
    class ReadCountingWorkspace(LocalWorkspace):
        def __init__(self, root):
            super().__init__(root, workspace_id="local")
            self.read_count = 0

        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            self.read_count += 1
            return await super().read_bytes(path, offset=offset, max_bytes=max_bytes)

    workspace = ReadCountingWorkspace(tmp_path)
    asyncio.run(workspace.write_bytes("tiny.png", TINY_PNG_BYTES))
    ctx = ToolContext(session_id="sess_1", workspace=workspace)

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "tiny.png", "offset": 1}))

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert workspace.read_count == 0


def test_read_file_returns_not_found_error_for_missing_workspace_attachment(tmp_path):
    ctx = ToolContext(
        session_id="sess_1",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
    )

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "missing.pdf"}))

    assert result.content == "Read refused: workspace file not found: missing.pdf."
    assert result.structured == {"path": "missing.pdf", "reason": "not_found"}
    assert result.is_error is True


def test_read_file_returns_not_found_error_when_workspace_attachment_disappears(tmp_path):
    class DisappearingWorkspace(LocalWorkspace):
        def __init__(self, root):
            super().__init__(root, workspace_id="local")
            self.read_count = 0

        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ) -> WorkspaceReadResult:
            self.read_count += 1
            if self.read_count == 2:
                raise FileNotFoundError(path)
            return await super().read_bytes(path, offset=offset, max_bytes=max_bytes)

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = DisappearingWorkspace(workspace_root)
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        artifact_store=artifact_store,
    )
    asyncio.run(workspace.write_bytes("invoice.pdf", _tiny_pdf_bytes()))

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "invoice.pdf"}))

    assert result.content == "Read refused: workspace file not found: invoice.pdf."
    assert result.structured == {"path": "invoice.pdf", "reason": "not_found"}
    assert result.is_error is True


def test_read_file_does_not_misclassify_missing_published_workspace_snapshot(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspace(workspace_root, workspace_id="local")
    error = FileNotFoundError("published artifact disappeared")
    artifact_store = VanishingPublishedArtifactStore(error)
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        artifact_store=artifact_store,
    )
    asyncio.run(workspace.write_bytes("invoice.pdf", _tiny_pdf_bytes()))

    with pytest.raises(FileNotFoundError) as raised:
        asyncio.run(ReadFileTool().run(ctx, {"path": "invoice.pdf"}))

    assert raised.value is error


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


def test_read_file_bounds_initial_and_full_workspace_attachment_reads(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = BoundedReadWorkspace(workspace_root, cap=2)
    artifact_store = RecordingLocalArtifactStore(tmp_path / "artifacts")
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        artifact_store=artifact_store,
    )
    asyncio.run(workspace.write_bytes("tiny.png", TINY_PNG_BYTES))

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "tiny.png"}))

    assert workspace.bound_requests == [
        DEFAULT_READ_LIMIT_BYTES,
        files_module.MAX_IMAGE_SOURCE_BYTES,
    ]
    assert workspace.read_requests == [(0, 2), (0, 2)]
    assert artifact_store.put_count == 0
    assert result.is_error is True
    assert result.structured["bytes"] == 2
    assert result.structured["total_bytes"] == len(TINY_PNG_BYTES)
    assert result.structured["truncated"] is True
    assert result.structured["effective_read_limit_bytes"] == 2
    assert "effective native-inspection limit (max 2 bytes)" in result.content
    assert str(files_module.MAX_IMAGE_SOURCE_BYTES) not in result.content


def test_read_file_rejects_invalid_full_attachment_bound_before_snapshot(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = BoundedReadWorkspace(
        workspace_root,
        bound_results=[DEFAULT_READ_LIMIT_BYTES, 0],
    )
    artifact_store = RecordingLocalArtifactStore(tmp_path / "artifacts")
    ctx = ToolContext(
        session_id="sess_1",
        workspace=workspace,
        artifact_store=artifact_store,
    )
    asyncio.run(workspace.write_bytes("tiny.png", TINY_PNG_BYTES))

    with pytest.raises(
        RuntimeError,
        match=(
            r"BoundedReadWorkspace\.bounded_read_limit\(\) must return a positive integer "
            rf"no greater than max_bytes={files_module.MAX_IMAGE_SOURCE_BYTES}\."
        ),
    ):
        asyncio.run(ReadFileTool().run(ctx, {"path": "tiny.png"}))

    assert workspace.bound_requests == [
        DEFAULT_READ_LIMIT_BYTES,
        files_module.MAX_IMAGE_SOURCE_BYTES,
    ]
    assert workspace.read_requests == [(0, DEFAULT_READ_LIMIT_BYTES)]
    assert artifact_store.put_count == 0


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

        async def read_bytes(
            self,
            path: str,
            *,
            offset: int = 0,
            max_bytes: int | None = None,
        ):
            self.read_count += 1
            if self.read_count == 1:
                return await super().read_bytes(path, offset=offset, max_bytes=max_bytes)
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
    assert csv_result.content.endswith("[/read_file metadata]\nname,score\ncayu,10\n")
    assert csv_result.structured["encoding"] == "utf-8"
    assert "binary" not in csv_result.structured
    assert html_result.is_error is False
    assert html_result.content.endswith("[/read_file metadata]\n<h1>Cayu</h1>\n")
    assert readme_result.is_error is False
    assert readme_result.content.endswith("[/read_file metadata]\nCayu workspace notes\n")
    assert readme_result.structured["encoding"] == "utf-8"
    assert "binary" not in readme_result.structured
    assert typescript_result.is_error is False
    assert typescript_result.content.endswith(
        "[/read_file metadata]\nexport const name = 'cayu';\n"
    )
    assert typescript_result.structured["encoding"] == "utf-8"
    assert "binary" not in typescript_result.structured
    assert java_result.is_error is False
    assert java_result.content.endswith("[/read_file metadata]\nclass Main {}\n")
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


def test_read_file_reapplies_attachment_limit_to_reused_pdf_derivation(
    tmp_path,
    monkeypatch,
):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    pdf = asyncio.run(
        artifact_store.put_bytes(
            _multi_page_pdf_bytes(1),
            filename="report.pdf",
            content_type="application/pdf",
            session_id="sess_1",
        )
    )
    derived_pdf = b"%PDF-1.7\n" + (b"x" * 190)
    monkeypatch.setattr(
        files_module,
        "_extract_pdf_pages",
        lambda content, pages: (derived_pdf, " showing page 1"),
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)
    tool = ReadFileTool()

    permissive = asyncio.run(
        tool.run(
            ctx,
            {
                "artifact_id": pdf.id,
                "pages": "1",
                "max_attachment_bytes": len(derived_pdf),
            },
        )
    )
    strict = asyncio.run(
        tool.run(
            ctx,
            {
                "artifact_id": pdf.id,
                "pages": "1",
                "max_attachment_bytes": len(derived_pdf) - 1,
            },
        )
    )
    reconstructed_store = LocalArtifactStore(
        tmp_path / "artifacts",
        store_id="artifacts",
    )
    reconstructed_ctx = ToolContext(
        session_id="sess_1",
        artifact_store=reconstructed_store,
    )

    async def reuse_concurrently() -> tuple[ToolResult, ToolResult]:
        args = {
            "artifact_id": pdf.id,
            "pages": "1",
            "max_attachment_bytes": len(derived_pdf),
        }
        first, second = await asyncio.gather(
            tool.run(reconstructed_ctx, args),
            tool.run(reconstructed_ctx, args),
        )
        return first, second

    reused = asyncio.run(reuse_concurrently())

    assert permissive.is_error is False
    assert strict.is_error is True
    assert strict.artifacts == ()
    assert "attachment_artifact_id" not in strict.structured
    assert f"max_attachment_bytes={len(derived_pdf) - 1}" in strict.content
    assert all(result.is_error is False for result in reused)
    assert {result.structured["attachment_artifact_id"] for result in reused} == {
        permissive.structured["attachment_artifact_id"]
    }
    listing = asyncio.run(artifact_store.list(scope=ArtifactScope.SESSION, session_id="sess_1"))
    assert listing.total_count == 2


def test_read_file_can_cache_pdf_derivation_after_a_strict_rejection(
    tmp_path,
    monkeypatch,
):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    pdf = asyncio.run(
        artifact_store.put_bytes(
            _multi_page_pdf_bytes(1),
            filename="report.pdf",
            content_type="application/pdf",
            session_id="sess_1",
        )
    )
    derived_pdf = b"%PDF-1.7\n" + (b"x" * 190)
    monkeypatch.setattr(
        files_module,
        "_extract_pdf_pages",
        lambda content, pages: (derived_pdf, " showing page 1"),
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)
    tool = ReadFileTool()

    strict = asyncio.run(
        tool.run(
            ctx,
            {
                "artifact_id": pdf.id,
                "pages": "1",
                "max_attachment_bytes": len(derived_pdf) - 1,
            },
        )
    )
    permissive = asyncio.run(
        tool.run(
            ctx,
            {
                "artifact_id": pdf.id,
                "pages": "1",
                "max_attachment_bytes": len(derived_pdf),
            },
        )
    )

    assert strict.is_error is True
    assert strict.artifacts == ()
    assert permissive.is_error is False
    listing = asyncio.run(artifact_store.list(scope=ArtifactScope.SESSION, session_id="sess_1"))
    assert listing.total_count == 2


def test_read_file_rebuilds_missing_and_corrupt_pdf_derivations(
    tmp_path,
    monkeypatch,
):
    artifact_root = tmp_path / "artifacts"
    artifact_store = LocalArtifactStore(artifact_root, store_id="artifacts")
    pdf = asyncio.run(
        artifact_store.put_bytes(
            _multi_page_pdf_bytes(1),
            filename="report.pdf",
            content_type="application/pdf",
            session_id="sess_1",
        )
    )
    derived_pdf = b"%PDF-1.7\n" + (b"x" * 190)
    extraction_count = 0

    def extract(content: bytes, pages: str | None):
        nonlocal extraction_count
        extraction_count += 1
        return derived_pdf, " showing page 1"

    monkeypatch.setattr(files_module, "_extract_pdf_pages", extract)
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)
    args = {
        "artifact_id": pdf.id,
        "pages": "1",
        "max_attachment_bytes": len(derived_pdf),
    }

    first = asyncio.run(ReadFileTool().run(ctx, args))
    first_id = first.structured["attachment_artifact_id"]
    asyncio.run(artifact_store.delete(first_id))
    rebuilt_missing = asyncio.run(ReadFileTool().run(ctx, args))
    missing_id = rebuilt_missing.structured["attachment_artifact_id"]
    (artifact_root / missing_id / "content").write_bytes(b"y" * len(derived_pdf))
    strict_args = {
        **args,
        "max_attachment_bytes": len(derived_pdf) - 1,
    }
    rejected_corrupt = asyncio.run(ReadFileTool().run(ctx, strict_args))
    extraction_count_after_strict_rebuild = extraction_count
    rebuilt_corrupt = asyncio.run(ReadFileTool().run(ctx, args))

    assert rebuilt_missing.is_error is False
    assert missing_id != first_id
    assert rejected_corrupt.is_error is True
    assert rejected_corrupt.artifacts == ()
    assert extraction_count_after_strict_rebuild == 3
    assert rebuilt_corrupt.is_error is False
    assert rebuilt_corrupt.structured["attachment_artifact_id"] != missing_id
    assert extraction_count == 4


def test_read_file_reapplies_attachment_limit_to_reused_image_derivation(
    tmp_path,
    monkeypatch,
):
    artifact_store = RecordingLocalArtifactStore(tmp_path / "artifacts")
    image = asyncio.run(
        artifact_store.put_bytes(
            TINY_PNG_BYTES,
            filename="image.png",
            content_type="image/png",
            session_id="sess_1",
        )
    )
    monkeypatch.setattr(
        files_module,
        "_resize_image_bytes",
        lambda content, *, content_type, max_bytes: (TINY_PNG_BYTES, "image/png"),
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)
    args = {"artifact_id": image.id, "max_attachment_bytes": 10}

    fresh = asyncio.run(ReadFileTool().run(ctx, args))
    reused = asyncio.run(ReadFileTool().run(ctx, args))

    assert fresh.is_error is True
    assert fresh.artifacts == ()
    assert reused.is_error is True
    assert reused.artifacts == ()
    assert artifact_store.put_count == 2


def test_read_file_validates_cached_image_derivations_after_reconstruction(
    tmp_path,
    monkeypatch,
):
    image_module = import_module("PIL.Image")
    source_buffer = io.BytesIO()
    image_module.new("RGB", (64, 64), "white").save(source_buffer, format="PNG")
    artifact_root = tmp_path / "artifacts"
    artifact_store = LocalArtifactStore(artifact_root, store_id="artifacts")
    image = asyncio.run(
        artifact_store.put_bytes(
            source_buffer.getvalue(),
            filename="image.png",
            content_type="image/png",
            session_id="sess_1",
        )
    )
    resize_count = 0

    def resize(content: bytes, *, content_type: str, max_bytes: int):
        nonlocal resize_count
        resize_count += 1
        return TINY_PNG_BYTES, "image/png"

    monkeypatch.setattr(files_module, "_resize_image_bytes", resize)
    args = {
        "artifact_id": image.id,
        "max_attachment_bytes": len(TINY_PNG_BYTES),
    }
    first = asyncio.run(
        ReadFileTool().run(
            ToolContext(session_id="sess_1", artifact_store=artifact_store),
            args,
        )
    )
    first_id = first.structured["attachment_artifact_id"]
    reconstructed_store = LocalArtifactStore(artifact_root, store_id="artifacts")
    reconstructed_ctx = ToolContext(
        session_id="sess_1",
        artifact_store=reconstructed_store,
    )

    async def reuse_concurrently() -> tuple[ToolResult, ToolResult]:
        first_reuse, second_reuse = await asyncio.gather(
            ReadFileTool().run(reconstructed_ctx, args),
            ReadFileTool().run(reconstructed_ctx, args),
        )
        return first_reuse, second_reuse

    reused = asyncio.run(reuse_concurrently())
    (artifact_root / first_id / "content").write_bytes(b"x" * len(TINY_PNG_BYTES))
    rebuilt = asyncio.run(ReadFileTool().run(reconstructed_ctx, args))

    assert first.is_error is False
    assert all(result.is_error is False for result in reused)
    assert {result.structured["attachment_artifact_id"] for result in reused} == {first_id}
    assert rebuilt.is_error is False
    assert rebuilt.structured["attachment_artifact_id"] != first_id
    assert resize_count == 2


def test_read_file_admits_custom_reader_attachments_at_the_final_boundary(tmp_path):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    pdf = asyncio.run(
        artifact_store.put_bytes(
            _multi_page_pdf_bytes(1),
            filename="report.pdf",
            content_type="application/pdf",
            session_id="sess_1",
        )
    )

    class AttachingCustomPdfReader:
        def __init__(self, advertised_size: int) -> None:
            self.advertised_size = advertised_size

        def can_read(self, artifact) -> bool:
            return artifact.content_type == "application/pdf"

        async def read(self, request: ArtifactReadRequest) -> ToolResult:
            return ToolResult(
                content="custom attachment",
                artifacts=[
                    file_attachment(
                        artifact_id=request.artifact.id,
                        kind="document",
                        filename=request.artifact.filename,
                        content_type=request.artifact.content_type,
                        size_bytes=self.advertised_size,
                    )
                ],
            )

    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)
    oversized = asyncio.run(
        ReadFileTool(artifact_readers=[AttachingCustomPdfReader(pdf.size_bytes)]).run(
            ctx,
            {
                "artifact_id": pdf.id,
                "max_attachment_bytes": pdf.size_bytes - 1,
            },
        )
    )
    understated = asyncio.run(
        ReadFileTool(artifact_readers=[AttachingCustomPdfReader(pdf.size_bytes - 1)]).run(
            ctx,
            {
                "artifact_id": pdf.id,
                "max_attachment_bytes": pdf.size_bytes,
            },
        )
    )

    assert oversized.is_error is True
    assert oversized.artifacts == ()
    assert understated.is_error is True
    assert understated.artifacts == ()
    assert "changed before admission" in understated.content


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


@pytest.mark.parametrize("wrapped", [False, True], ids=["raw", "invocation-wrapped"])
def test_read_file_returns_structured_result_for_missing_local_artifact(
    tmp_path,
    wrapped: bool,
) -> None:
    local_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    artifact_store: ArtifactStore = local_store
    if wrapped:
        tracker = InvocationSecretTracker(SecretRedactor())
        artifact_store = InvocationArtifactStoreHandle(
            local_store,
            redactor_snapshot_provider=tracker.snapshot,
            capture_observer=tracker.record_ambiguous_output_capture,
        )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": MISSING_ARTIFACT_ID}))

    assert result.content == "Artifact was not found."
    assert result.structured == {
        "artifact_id": MISSING_ARTIFACT_ID,
        "reason": "not_found",
    }
    assert result.is_error is True


def test_read_file_keeps_invalid_artifact_id_distinct_from_absence(tmp_path) -> None:
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts")
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": "invalid"}))

    assert result.structured == {"error": "invalid_arguments"}
    assert "must match the local artifact id format" in result.content
    assert result.content != "Artifact was not found."
    assert result.is_error is True


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("artifact authorization failed"),
        ArtifactStoreUnavailableError("artifact backend unavailable"),
        ExceptionGroup(
            "ambiguous artifact failure",
            [FileNotFoundError("missing"), PermissionError("denied")],
        ),
    ],
    ids=["authorization", "unavailable", "grouped"],
)
def test_read_file_does_not_misclassify_non_absence_artifact_failures(
    error: BaseException,
) -> None:
    artifact_store = FailingReadArtifactStore(error)
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    with pytest.raises(type(error)) as raised:
        asyncio.run(ReadFileTool().run(ctx, {"artifact_id": MISSING_ARTIFACT_ID}))

    assert raised.value is error


def test_read_file_preserves_real_task_cancellation_during_artifact_read() -> None:
    async def exercise() -> None:
        artifact_store = BlockingReadArtifactStore()
        ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)
        task = asyncio.create_task(ReadFileTool().run(ctx, {"artifact_id": MISSING_ARTIFACT_ID}))
        await artifact_store.started.wait()

        task.cancel()

        assert task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() is True

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("filename", "content_type", "content"),
    [
        pytest.param("image.png", "image/png", TINY_PNG_BYTES, id="image"),
        pytest.param("document.pdf", "application/pdf", None, id="pdf"),
    ],
)
def test_read_file_returns_missing_result_when_native_artifact_disappears_during_reread(
    filename: str,
    content_type: str,
    content: bytes | None,
) -> None:
    if content is None:
        content = _tiny_pdf_bytes()
    backend_canary = "/private/backend/tenant-secret/disappeared-artifact"
    artifact_store = DisappearingRereadArtifactStore(
        artifact_id=MISSING_ARTIFACT_ID,
        filename=filename,
        content_type=content_type,
        content=content,
        error=FileNotFoundError(backend_canary),
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": MISSING_ARTIFACT_ID}))

    assert result.content == "Artifact was not found."
    assert result.structured == {
        "artifact_id": MISSING_ARTIFACT_ID,
        "reason": "not_found",
    }
    assert result.is_error is True
    assert artifact_store.read_count == 2
    assert backend_canary not in json.dumps(result.model_dump(mode="json"))


def test_read_file_does_not_misclassify_custom_reader_file_not_found() -> None:
    class MissingDependencyReader:
        def can_read(self, artifact: ArtifactMetadata) -> bool:
            return artifact.content_type == "application/x-custom"

        async def read(self, request: ArtifactReadRequest) -> ToolResult:
            del request
            raise FileNotFoundError("custom reader dependency was not found")

    artifact_store = DisappearingRereadArtifactStore(
        artifact_id=MISSING_ARTIFACT_ID,
        filename="custom.bin",
        content_type="application/x-custom",
        content=b"custom",
        error=FileNotFoundError("store reread should not occur"),
    )
    ctx = ToolContext(session_id="sess_1", artifact_store=artifact_store)

    with pytest.raises(FileNotFoundError, match="custom reader dependency"):
        asyncio.run(
            ReadFileTool(extra_artifact_readers=[MissingDependencyReader()]).run(
                ctx,
                {"artifact_id": MISSING_ARTIFACT_ID},
            )
        )

    assert artifact_store.read_count == 1


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

    assert workspace_result.content.endswith("[/read_file metadata]\nworkspace ok")
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


def test_exec_command_tool_redacts_at_runner_capture_before_output_bound(tmp_path) -> None:
    secret = "exec-command-boundary-secret"
    prefix = "p" * 45

    def redactor_provider() -> SecretRedactor:
        return SecretRedactor(secret)

    ctx = ToolContext(
        session_id="sess_1",
        runner=InvocationRunnerHandle(
            LocalRunner(tmp_path),
            redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                revision=0,
                redactor=redactor_provider(),
            ),
        ),
        invocation_secret_redactor=redactor_provider,
    )

    result = asyncio.run(
        ExecCommandTool().run(
            ctx,
            {
                "argv": [
                    sys.executable,
                    "-c",
                    (
                        "import os,sys,time; secret=os.environ['TOKEN']; "
                        f"sys.stdout.write({prefix!r} + secret[:10]); sys.stdout.flush(); "
                        "time.sleep(0.02); sys.stdout.write(secret[10:]); sys.stdout.flush()"
                    ),
                ],
                "env": {"TOKEN": secret},
                "max_output_bytes": 50,
            },
        )
    )

    serialized = json.dumps(result.model_dump(mode="json"))
    assert secret not in serialized
    assert secret[:10] not in serialized
    assert REDACTED_SECRET not in result.content
    assert result.structured["stdout"] == prefix
    assert result.structured["stdout_truncated"] is True


def test_exec_command_tool_reports_output_suppressed_by_invocation_runner() -> None:
    suppressed = "abcdef"
    runner = InvocationRunnerHandle(
        RecordingRunner(ExecResult(stdout=suppressed, stdout_bytes=len(suppressed))),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    result = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="sess_1", runner=runner),
            {"argv": ["emit"], "max_output_bytes": 3},
        )
    )

    assert result.structured["stdout"] == ""
    assert result.structured["stdout_truncated"] is True
    assert "[output truncated]" in result.content
    assert suppressed not in json.dumps(result.model_dump(mode="json"))


def test_exec_command_tool_preserves_normal_status_for_empty_output() -> None:
    runner = InvocationRunnerHandle(
        RecordingRunner(ExecResult()),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    result = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="sess_1", runner=runner),
            {"argv": ["true"]},
        )
    )

    assert result.content == "Command exited with code 0."
    assert result.structured["stdout_truncated"] is False
    assert result.structured["stderr_truncated"] is False


def test_builtin_tools_truncate_model_facing_large_outputs(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    file_ctx = ToolContext(session_id="sess_1", workspace=workspace)
    run_ctx = ToolContext(session_id="sess_1", runner=LocalRunner(tmp_path))

    asyncio.run(
        WriteFileTool().run(file_ctx, {"path": "large.txt", "content": "abcdef", "mode": "create"})
    )
    asyncio.run(
        WriteFileTool().run(file_ctx, {"path": "other.txt", "content": "", "mode": "create"})
    )
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

    assert '"next_offset":3' in read_result.content
    assert '"truncated":true' in read_result.content
    assert read_result.content.endswith("[/read_file metadata]\nabc")
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
                "mode": "create",
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


def test_exec_command_tool_applies_configured_default_timeout(tmp_path):
    runner = RecordingRunner()
    ctx = ToolContext(session_id="sess_1", runner=runner)
    tool = ExecCommandTool(default_timeout_seconds=420)

    result = asyncio.run(tool.run(ctx, {"argv": [sys.executable, "-c", "print('ok')"]}))
    assert result.is_error is False
    assert runner.timeout_s == 420
    assert tool.schema["properties"]["timeout_s"]["default"] == 420

    override = asyncio.run(
        tool.run(
            ctx,
            {"argv": [sys.executable, "-c", "print('ok')"], "timeout_s": 7},
        )
    )
    assert override.is_error is False
    assert runner.timeout_s == 7


@pytest.mark.parametrize("value", [0, 601, True, 1.5, "60"])
def test_exec_command_tool_rejects_invalid_configured_default_timeout(value):
    with pytest.raises(ValueError, match="default_timeout_seconds"):
        ExecCommandTool(default_timeout_seconds=value)


def test_exec_command_tool_default_timeout_is_execution_profile_material():
    assert (
        ExecCommandTool()._execution_profile_material()
        != ExecCommandTool(default_timeout_seconds=600)._execution_profile_material()
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


def test_exec_command_tool_preserves_runner_unavailable_diagnostic_from_policy_preflight() -> None:
    diagnostic = {
        "type": "cayu.runner_unavailable.v1",
        "adapter": "microsandbox",
        "status": "unavailable",
    }

    class UnavailablePreflightRunner(RecordingRunner):
        def preflight_exec(self, *args, **kwargs) -> None:
            del args, kwargs
            raise RunnerUnavailableError(
                "Microsandbox guest agent is unavailable.",
                diagnostic=diagnostic,
            )

    runner = UnavailablePreflightRunner()
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))

    result = asyncio.run(
        ExecCommandTool(policy=policy).run(
            ToolContext(session_id="sess_1", runner=runner),
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
    assert policy.requests == []
    assert runner.command is None


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
    safe_runtime_diagnostic = {
        "type": "cayu.runner_unavailable.v1",
        "adapter": "unknown",
        "status": "unavailable",
        "error_type": "RunnerUnavailableError",
    }
    assert failed.payload["result"]["structured"] == {
        "error": "runner_unavailable",
        "diagnostic": safe_runtime_diagnostic,
    }
    assert failed.payload["result"]["artifacts"] == [safe_runtime_diagnostic]


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


class _StructuralCommandRunnerHandle:
    def __init__(self) -> None:
        self.command: ExecCommand | None = None

    def resolve_cwd(self, cwd: str | None = None) -> str:
        return "/workspace" if cwd is None else cwd

    async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
        del kwargs
        self.command = command
        return ExecResult(stdout="structural runner\n")


def test_exec_command_tool_without_policy_accepts_existing_structural_runner_handle() -> None:
    runner = _StructuralCommandRunnerHandle()

    result = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="sess_1", runner=runner),
            {"argv": ["pwd"]},
        )
    )

    assert result.is_error is False
    assert result.content == "structural runner"
    assert runner.command == ExecCommand.process("pwd")


def test_exec_command_tool_policy_requires_preflight_from_structural_runner_handle() -> None:
    runner = _StructuralCommandRunnerHandle()
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))

    with pytest.raises(TypeError, match="must support command preflight"):
        asyncio.run(
            ExecCommandTool(policy=policy).run(
                ToolContext(session_id="sess_1", runner=runner),
                {"argv": ["pwd"]},
            )
        )

    assert policy.requests == []
    assert runner.command is None


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


def test_exec_command_tool_runs_selected_backend_preflight_before_policy() -> None:
    snapshot_calls = 0

    class CountingVault(StaticVault):
        def __init__(self) -> None:
            super().__init__({"runner_secret": "value"})
            self.resolve_calls = 0

        async def resolve(self, ref, *, scope=None):
            self.resolve_calls += 1
            return await super().resolve(ref, scope=scope)

    def snapshot_provider() -> InvocationRedactorSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return InvocationRedactorSnapshot(revision=0, redactor=SecretRedactor())

    vault = CountingVault()
    runner = InvocationRunnerHandle(
        DockerRunner(
            "validation-probe",
            docker_path="/unreachable/docker",
            secret_env={"TOKEN": SecretRef(name="runner_secret")},
            secret_resolver=vault,
        ),
        redactor_snapshot_provider=snapshot_provider,
    )
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))

    with pytest.raises(
        RunnerExecutionError,
        match="Runner command execution failed",
    ) as raised:
        asyncio.run(
            ExecCommandTool(policy=policy).run(
                ToolContext(session_id="sess_1", runner=runner),
                {"argv": ["true"], "env": {"#COMMENT": "value"}},
            )
        )

    assert policy.requests == []
    assert snapshot_calls == 0
    assert vault.resolve_calls == 0
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_exec_command_tool_rejects_invalid_safe_host_environment_before_policy(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "invalid\udcffvalue")

    class CountingVault(StaticVault):
        def __init__(self) -> None:
            super().__init__({"runner_secret": "value"})
            self.resolve_calls = 0

        async def resolve(self, ref, *, scope=None):
            self.resolve_calls += 1
            return await super().resolve(ref, scope=scope)

    snapshot_calls = 0

    def snapshot_provider() -> InvocationRedactorSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return InvocationRedactorSnapshot(revision=0, redactor=SecretRedactor())

    vault = CountingVault()
    runner = InvocationRunnerHandle(
        LocalRunner(
            tmp_path,
            secret_env={"API_TOKEN": SecretRef(name="runner_secret")},
            secret_resolver=vault,
        ),
        redactor_snapshot_provider=snapshot_provider,
    )
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))

    with pytest.raises(RunnerExecutionError, match="Runner command execution failed"):
        asyncio.run(
            ExecCommandTool(policy=policy).run(
                ToolContext(session_id="sess_1", runner=runner),
                {"argv": ["true"]},
            )
        )

    assert policy.requests == []
    assert snapshot_calls == 0
    assert vault.resolve_calls == 0


def test_exec_command_tool_policy_preflight_preserves_pending_caller_cancellation() -> None:
    secret = "policy-preflight-cancellation-secret-canary-ABCDEFGHIJKLMNOP"
    snapshot_calls = 0

    class RejectingPreflightRunner(RecordingRunner):
        isolation = "docker"

        def preflight_exec(self, *args, **kwargs) -> None:
            del args, kwargs
            raise ValueError("backend preflight rejected the request")

    def snapshot_provider() -> InvocationRedactorSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(secret),
        )

    raw_runner = RejectingPreflightRunner()
    runner = InvocationRunnerHandle(
        raw_runner,
        redactor_snapshot_provider=snapshot_provider,
    )
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))

    async def invoke():
        task = asyncio.current_task()
        assert task is not None
        task.cancel(secret)
        return await run_tool(
            tool=ExecCommandTool(policy=policy),
            effect=ToolEffect.EXTERNAL,
            ctx=ToolContext(session_id="sess_1", runner=runner),
            arguments={"argv": ["pwd"]},
            redactor=lambda: SecretRedactor(secret),
        )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        task = asyncio.create_task(invoke())
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        return raised.value, task.cancelling(), task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == (REDACTED_SECRET,)
    assert type(cancellation.__cause__) is RunnerExecutionError
    assert cancellation.__context__ is None
    assert policy.requests == []
    assert raw_runner.command is None
    assert snapshot_calls == 1
    assert secret not in repr(cancellation)
    assert secret not in repr(cancellation.__cause__)


def test_exec_command_tool_direct_preflight_preserves_pending_caller_cancellation() -> None:
    class RejectingPreflightRunner(RecordingRunner):
        def preflight_exec(self, *args, **kwargs) -> None:
            del args, kwargs
            raise ValueError("backend preflight rejected the request")

    runner = RejectingPreflightRunner()
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))

    async def invoke():
        task = asyncio.current_task()
        assert task is not None
        task.cancel("caller requested stop")
        return await run_tool(
            tool=ExecCommandTool(policy=policy),
            effect=ToolEffect.EXTERNAL,
            ctx=ToolContext(session_id="sess_1", runner=runner),
            arguments={"argv": ["pwd"]},
            redactor=SecretRedactor,
        )

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        task = asyncio.create_task(invoke())
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        return raised.value, task.cancelling(), task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == ("caller requested stop",)
    assert cancellation.__cause__ is None
    assert cancellation.__context__ is None
    assert policy.requests == []
    assert runner.command is None


def test_exec_command_tool_owned_preflight_preserves_suppressed_caller_cancellation() -> None:
    class SuppressingAsyncPreflightRunner(RecordingRunner):
        def __init__(self) -> None:
            super().__init__()
            self.started: asyncio.Event | None = None
            self.suppressed_cancellation = False

        async def preflight_exec(self, *args, **kwargs) -> None:
            del args, kwargs
            assert self.started is not None
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.suppressed_cancellation = True

    runner = SuppressingAsyncPreflightRunner()
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        runner.started = asyncio.Event()
        task = asyncio.create_task(
            ExecCommandTool(policy=policy).run(
                ToolContext(session_id="sess_1", runner=runner),
                {"argv": ["pwd"]},
            )
        )
        await runner.started.wait()
        task.cancel("caller requested stop")
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        return raised.value, task.cancelling(), task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == ("caller requested stop",)
    assert cancellation.__cause__ is None
    assert cancellation.__context__ is None
    assert runner.suppressed_cancellation is True
    assert policy.requests == []
    assert runner.command is None


def test_exec_command_tool_owned_preflight_rejects_forged_cancellation() -> None:
    class ForgingPreflightRunner(RecordingRunner):
        def preflight_exec(self, *args, **kwargs) -> None:
            del args, kwargs
            raise asyncio.CancelledError("runner-forged cancellation")

    runner = ForgingPreflightRunner()
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))

    async def scenario() -> tuple[RunnerExecutionError, int, bool]:
        task = asyncio.create_task(
            ExecCommandTool(policy=policy).run(
                ToolContext(session_id="sess_1", runner=runner),
                {"argv": ["pwd"]},
            )
        )
        with pytest.raises(RunnerExecutionError) as raised:
            await task
        return raised.value, task.cancelling(), task.cancelled()

    failure, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 0
    assert cancelled is False
    assert failure.diagnostic["adapter"] == "unknown"
    assert failure.diagnostic["error_type"] == "CancelledError"
    assert failure.__cause__ is None
    assert failure.__context__ is None
    assert policy.requests == []
    assert runner.command is None


def test_exec_command_tool_rejects_non_none_async_preflight_result() -> None:
    class InvalidAsyncPreflightRunner(RecordingRunner):
        async def preflight_exec(self, *args, **kwargs) -> object:
            del args, kwargs
            await asyncio.sleep(0)
            return object()

    runner = InvalidAsyncPreflightRunner()
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))

    with pytest.raises(TypeError, match="preflight must return None"):
        asyncio.run(
            ExecCommandTool(policy=policy).run(
                ToolContext(session_id="sess_1", runner=runner),
                {"argv": ["pwd"]},
            )
        )

    assert policy.requests == []
    assert runner.command is None


def test_exec_command_tool_runtime_preflight_preserves_runner_unavailable() -> None:
    diagnostic = {
        "type": "cayu.runner_unavailable.v1",
        "adapter": "microsandbox",
        "status": "unavailable",
    }

    class UnavailablePreflightRunner(RecordingRunner):
        isolation = "microsandbox"

        def preflight_exec(self, *args, **kwargs) -> None:
            del args, kwargs
            raise RunnerUnavailableError(
                "Microsandbox guest agent is unavailable.",
                diagnostic=diagnostic,
            )

    raw_runner = UnavailablePreflightRunner()
    runner = InvocationRunnerHandle(
        raw_runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))

    outcome = asyncio.run(
        run_tool(
            tool=ExecCommandTool(policy=policy),
            effect=ToolEffect.EXTERNAL,
            ctx=ToolContext(session_id="sess_1", runner=runner),
            arguments={"argv": ["pwd"]},
            redactor=SecretRedactor,
        )
    )

    safe_diagnostic = {
        "type": "cayu.runner_unavailable.v1",
        "adapter": "microsandbox",
        "status": "unavailable",
        "error_type": "RunnerUnavailableError",
    }
    assert outcome.result.content == "Runner is unavailable."
    assert outcome.result.structured == {
        "error": "runner_unavailable",
        "diagnostic": safe_diagnostic,
    }
    assert outcome.result.artifacts == [safe_diagnostic]
    assert policy.requests == []
    assert raw_runner.command is None


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
    async def create_bytes(self, path: str, content: bytes):
        del path, content
        raise ValueError("workspace backend changed")


def test_write_file_tool_does_not_classify_workspace_value_error_as_invalid_arguments(tmp_path):
    workspace = _FailingWriteWorkspace(tmp_path, workspace_id="failing-write")

    with pytest.raises(ValueError, match="workspace backend changed"):
        asyncio.run(
            WriteFileTool().run(
                ToolContext(session_id="sess_1", workspace=workspace),
                {"path": "notes.txt", "content": "hello", "mode": "create"},
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

    assert type(result) is ToolResult
    assert result.is_error is True
    assert result.content == ("Command denied by policy. Shell scripts are not allowed here.")
    assert result.structured == {
        "error": "command_denied",
        "decision": "deny",
        "reason": "Shell scripts are not allowed here.",
    }
    assert result.model_dump() == {
        "content": "Command denied by policy. Shell scripts are not allowed here.",
        "structured": {
            "error": "command_denied",
            "decision": "deny",
            "reason": "Shell scripts are not allowed here.",
        },
        "artifacts": [],
        "is_error": True,
    }
    assert dict(result) == result.model_dump()
    assert result == ToolResult(**result.model_dump())
    assert policy.requests[0][1].cwd == "repo"
    assert policy.requests[0][1].canonical_cwd == "/workspace/repo"
    assert runner.command is None


@pytest.mark.parametrize(
    "decision",
    [CommandPolicyDecision.DENY, CommandPolicyDecision.REQUIRE_COMMAND_APPROVAL],
)
def test_exec_command_tool_bounds_oversized_policy_refusal(decision):
    runner = RecordingRunner()
    reason = "policy says no 🙂 " * 600
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=decision, reason=reason))
    tool = ExecCommandTool(policy=policy)
    context = ToolContext(session_id="sess_bounded_command_denial", runner=runner)

    result = asyncio.run(tool.run(context, {"argv": ["git", "push"]}))

    assert type(result) is ToolResult
    assert result.is_error is True
    assert len(result.content.encode("utf-8")) <= _POLICY_DENIAL_TEXT_MAX_BYTES
    assert result.content.endswith(_POLICY_DENIAL_TRUNCATION_MARKER)
    assert result.structured is not None
    assert result.structured["reason"] == _bound_policy_denial_text(reason)
    denial = context._policy_denial_for(tool)
    assert denial is not None
    assert denial.reason == reason
    assert denial.result.structured is not None
    assert denial.result.structured["reason"] == reason
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


@pytest.mark.parametrize(
    ("decision", "expected_error", "policy_reason", "expected_reason"),
    [
        (
            CommandPolicyDecision.DENY,
            "command_denied",
            "Command is outside the approved workflow.",
            "Command is outside the approved workflow.",
        ),
        (
            CommandPolicyDecision.REQUIRE_COMMAND_APPROVAL,
            "command_approval_required",
            None,
            "Command requires approval before it can run.",
        ),
        (
            CommandPolicyDecision.DENY,
            "command_denied",
            "oversized command denial 🙂 " * 500,
            _bound_policy_denial_text("oversized command denial 🙂 " * 500),
        ),
    ],
)
def test_exec_command_policy_refusal_emits_one_canonical_blocked_event(
    decision,
    expected_error,
    policy_reason,
    expected_reason,
):
    runner = RecordingRunner()

    class AttemptedRewriteHook(RuntimeHook):
        def __init__(self) -> None:
            self.event_types: list[EventType | str] = []

        async def after_tool_call(self, context: ToolCallHookContext) -> AfterToolCallDecision:
            self.event_types.append(context.tool_event.type)
            return AfterToolCallDecision(
                action="modify",
                modified_result=ToolResult(content="rewritten as success"),
            )

    hook = AttemptedRewriteHook()
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=decision, reason=policy_reason))
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_policy_refusal",
                    name="exec_command",
                    arguments={
                        "argv": ["printenv", "TOP_SECRET"],
                        "env": {"TOP_SECRET": "env-secret-value"},
                        "stdin": "stdin-secret-value",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("I will use another approach."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="policy-runner"), runner=runner),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ExecCommandTool(policy=policy)],
        runtime_hooks=[hook],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_command_policy_{decision.value}",
                messages=[Message.text("user", "run the command")],
            ),
        )
    )

    blocked = [event for event in events if event.type == EventType.TOOL_CALL_BLOCKED]
    assert len(blocked) == 1
    assert all(event.type != EventType.TOOL_CALL_FAILED for event in events)
    assert all(event.type != EventType.TOOL_CALL_APPROVAL_REQUESTED for event in events)
    assert all(event.type != EventType.SESSION_INTERRUPTED for event in events)
    assert runner.command is None
    assert hook.event_types == [EventType.TOOL_CALL_BLOCKED]
    payload = blocked[0].payload
    assert payload["denied_by"] == "command_policy"
    assert payload["decision"] == decision.value
    assert payload["metadata"] == {}
    assert payload["reason"] == expected_reason
    assert payload["tool_name"] == "exec_command"
    assert payload["tool_call_id"] == "cayu_event_10:tool_call_id"
    assert payload["tool_round_id"] == "cayu_event_10:tool_round_id"
    assert payload["idempotency_key"] == "[PRIVATE_EVENT_AUTHORITY]"
    assert payload["result"]["structured"]["error"] == expected_error
    assert payload["result"]["content"] != "rewritten as success"
    assert len(payload["result"]["content"].encode("utf-8")) <= (_POLICY_DENIAL_TEXT_MAX_BYTES)
    if policy_reason is not None:
        assert payload["result"]["structured"]["reason"] == _bound_policy_denial_text(policy_reason)
    terminal_json = json.dumps(payload)
    assert "env-secret-value" not in terminal_json
    assert "stdin-secret-value" not in terminal_json
    assert "TOP_SECRET" not in terminal_json

    transcript = asyncio.run(app.session_store.load_transcript(blocked[0].session_id))
    tool_result = next(message for message in transcript if message.role == "tool").content[0]
    assert tool_result.tool_call_id == "call_policy_refusal"
    assert tool_result.is_error is True
    if policy_reason is not None and len(policy_reason.encode("utf-8")) > (
        _POLICY_DENIAL_TEXT_MAX_BYTES
    ):
        assert tool_result.content.endswith(_POLICY_DENIAL_TRUNCATION_MARKER)
    else:
        assert expected_reason in tool_result.content
    checkpoint = asyncio.run(app.session_store.load_checkpoint(blocked[0].session_id))
    assert checkpoint is not None
    assert set(checkpoint) == {
        ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
        CHECKPOINT_SCHEMA_VERSION_KEY,
        LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
    }
    assert checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
    assert model_step_publication_from_checkpoint(checkpoint) is not None


def test_command_policy_denial_is_redacted_before_runtime_bounding():
    from cayu.vaults import REDACTED_SECRET, SecretRedactor

    secret_value = "BOUNDARY_SECRET_command_value"
    raw_reason = "a" * 4050 + secret_value
    runner = RecordingRunner()
    policy = _StaticCommandPolicy(
        CommandPolicyResult(decision=CommandPolicyDecision.DENY, reason=raw_reason)
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_secret_command",
                    name="exec_command",
                    arguments={"argv": ["git", "push"]},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp(secret_redactor=SecretRedactor(secret_value), enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="redacted-policy-runner"), runner=runner),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ExecCommandTool(policy=policy)],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_redacted_command_policy_denial",
                messages=[Message.text("user", "push")],
            ),
        )
    )

    blocked = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    redacted_reason = raw_reason.replace(secret_value, REDACTED_SECRET)
    assert "BOUNDARY" not in str(blocked.payload)
    assert blocked.payload["reason"] == _bound_policy_denial_text(redacted_reason)
    assert blocked.payload["result"]["content"] == _bound_policy_denial_text(
        f"Command denied by policy. {redacted_reason}"
    )
    assert blocked.payload["result"]["structured"]["reason"] == (
        _bound_policy_denial_text(redacted_reason)
    )
    assert runner.command is None
    transcript = asyncio.run(app.session_store.load_transcript(blocked.session_id))
    tool_result = next(message for message in transcript if message.role == "tool").content[0]
    assert "BOUNDARY" not in str(tool_result.model_dump(mode="json"))


def test_command_policy_redaction_preserves_protocol_fields_that_match_secrets():
    from cayu.vaults import SecretRedactor

    secret_values = [
        "reason",
        "denied",
        "deny",
        "policy",
        "decision",
        "result",
        "error",
    ]
    redactor = SecretRedactor(secret_values)
    raw_reason = "reason denied deny command policy decision result error " * 300
    runner = RecordingRunner()
    policy = _StaticCommandPolicy(
        CommandPolicyResult(decision=CommandPolicyDecision.DENY, reason=raw_reason)
    )
    observed: dict[str, object] = {}

    class ObserveCanonicalCommandDenial(RuntimeHook):
        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            observed["payload"] = context.tool_event.payload
            observed["structured"] = context.result.structured

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_command_collision",
                    name="exec_command",
                    arguments={"argv": ["git", "push"]},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp(secret_redactor=redactor, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="protocol-runner"), runner=runner),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ExecCommandTool(policy=policy)],
        runtime_hooks=[ObserveCanonicalCommandDenial()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_protocol_collision",
                messages=[Message.text("user", "push")],
            ),
        )
    )

    expected_reason = _bound_policy_denial_text(redactor.redact_text(raw_reason))
    expected_content = _bound_policy_denial_text(
        redactor.redact_text(f"Command denied by policy. {raw_reason}")
    )
    blocked = next(event for event in events if event.type == EventType.TOOL_CALL_BLOCKED)
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert blocked.payload["tool_name"] == "exec_command"
    assert blocked.payload["denied_by"] == "command_policy"
    assert blocked.payload["decision"] == "deny"
    assert blocked.payload["metadata"] == {}
    assert blocked.payload["reason"] == expected_reason
    assert blocked.payload["result"]["content"] == expected_content
    assert blocked.payload["result"]["structured"] == {
        "error": "command_denied",
        "decision": "deny",
        "reason": expected_reason,
    }
    observed_payload = dict(observed["payload"])
    observed_payload["tool_name"] = blocked.payload["tool_name"]
    private_authority_fields = {
        "idempotency_key",
        "model_attempt_id",
        "model_step_id",
        "tool_call_id",
        "tool_round_id",
    }
    assert {
        key: value for key, value in observed_payload.items() if key not in private_authority_fields
    } == {
        key: value for key, value in blocked.payload.items() if key not in private_authority_fields
    }
    assert blocked.payload["model_step_id"] == "[PRIVATE_EVENT_AUTHORITY]"
    assert blocked.payload["model_attempt_id"] == "[PRIVATE_EVENT_AUTHORITY]"
    assert blocked.payload["idempotency_key"] == "[PRIVATE_EVENT_AUTHORITY]"
    assert blocked.payload["tool_call_id"] == "cayu_event_10:tool_call_id"
    assert blocked.payload["tool_round_id"] == "cayu_event_10:tool_round_id"
    assert observed["structured"] == blocked.payload["result"]["structured"]
    assert runner.command is None

    transcript = asyncio.run(app.session_store.load_transcript(blocked.session_id))
    tool_result = next(message for message in transcript if message.role == "tool").content[0]
    assert tool_result.content == expected_content
    assert tool_result.structured == blocked.payload["result"]["structured"]


def test_allowed_nonzero_command_remains_completed_and_not_policy_blocked():
    runner = RecordingRunner(ExecResult(stderr="not found", exit_code=7))
    policy = _StaticCommandPolicy(CommandPolicyResult(decision=CommandPolicyDecision.ALLOW))
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_nonzero",
                    name="exec_command",
                    arguments={"argv": ["missing-command"]},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="nonzero-runner"), runner=runner),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ExecCommandTool(policy=policy)],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_allowed_nonzero_command",
                messages=[Message.text("user", "try it")],
            ),
        )
    )

    completed = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
    assert len(completed) == 1
    assert completed[0].payload["result"]["structured"]["exit_code"] == 7
    assert all(event.type != EventType.TOOL_CALL_BLOCKED for event in events)
    assert all(event.type != EventType.TOOL_CALL_FAILED for event in events)
    assert runner.command == ExecCommand.process("missing-command")


def test_command_policy_exception_remains_tool_failure_not_policy_block():
    runner = RecordingRunner()

    class RaisingCommandPolicy(CommandPolicy):
        async def evaluate(self, ctx: ToolContext, request: CommandRequest) -> CommandPolicyResult:
            del ctx, request
            raise RuntimeError("command policy crashed")

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_policy_error",
                    name="exec_command",
                    arguments={"argv": ["pwd"]},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="policy-error-runner"), runner=runner),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ExecCommandTool(policy=RaisingCommandPolicy())],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_command_policy_error",
                messages=[Message.text("user", "run pwd")],
            ),
        )
    )

    failed = [event for event in events if event.type == EventType.TOOL_CALL_FAILED]
    assert len(failed) == 1
    assert failed[0].payload["result"]["content"] == "command policy crashed"
    assert all(event.type != EventType.TOOL_CALL_BLOCKED for event in events)
    assert runner.command is None


def test_command_policy_denial_resolves_once_inside_mixed_tool_round():
    runner = RecordingRunner()
    recorder = ContextRecordingTool()
    policy = _StaticCommandPolicy(
        CommandPolicyResult(
            decision=CommandPolicyDecision.DENY,
            reason="Command denied in mixed round.",
        )
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_denied",
                    name="exec_command",
                    arguments={"argv": ["git", "push"]},
                ),
                ModelStreamEvent.tool_call(
                    id="call_allowed",
                    name="record_context",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="mixed-runner"), runner=runner),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ExecCommandTool(policy=policy), recorder],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_mixed_command_policy_denial",
                messages=[Message.text("user", "use both tools")],
            ),
        )
    )

    terminal = [
        event
        for event in events
        if event.type
        in {
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
            EventType.TOOL_CALL_BLOCKED,
        }
    ]
    assert [event.type for event in terminal] == [
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_COMPLETED,
    ]
    assert all(event.payload["tool_call_id"] == f"{event.id}:tool_call_id" for event in terminal)
    assert terminal[0].payload["denied_by"] == "command_policy"
    assert terminal[0].payload["metadata"] == {}
    assert all(event.payload["tool_round_id"] == f"{event.id}:tool_round_id" for event in terminal)
    assert runner.command is None
    assert recorder.context is not None

    transcript = asyncio.run(app.session_store.load_transcript("sess_mixed_command_policy_denial"))
    tool_message = next(message for message in transcript if message.role == "tool")
    assert [part.tool_call_id for part in tool_message.content] == [
        "call_denied",
        "call_allowed",
    ]


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


def test_runtime_treats_missing_artifact_as_recoverable_safe_tool_result() -> None:
    backend_canary = "/private/backend/tenant-secret/missing-artifact"
    artifact_store = DisappearingRereadArtifactStore(
        artifact_id=MISSING_ARTIFACT_ID,
        filename="missing.png",
        content_type="image/png",
        content=TINY_PNG_BYTES,
        error=FileNotFoundError(backend_canary),
        session_id="sess_missing_artifact",
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="read_file",
                    arguments={"artifact_id": MISSING_ARTIFACT_ID},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local-dev"), artifact_store=artifact_store),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[ReadFileTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_missing_artifact",
                messages=[Message.text("user", "read the artifact")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    failed = next(event for event in events if event.type == EventType.TOOL_CALL_FAILED)
    assert failed.payload["result"]["content"] == "Artifact was not found."
    assert failed.payload["result"]["structured"] == {
        "artifact_id": MISSING_ARTIFACT_ID,
        "reason": "not_found",
    }
    assert failed.payload["result"]["is_error"] is True
    assert all(event.type != EventType.TOOL_CALL_COMPLETED for event in events)
    assert artifact_store.read_count == 2
    assert len(provider.requests) == 2

    transcript = asyncio.run(app.session_store.load_transcript("sess_missing_artifact"))
    published = json.dumps(
        {
            "events": [event.model_dump(mode="json") for event in events],
            "transcript": [message.model_dump(mode="json") for message in transcript],
            "provider_request": [
                message.model_dump(mode="json") for message in provider.requests[1].messages
            ],
        },
        sort_keys=True,
    )
    assert backend_canary not in published
    assert "Artifact was not found." in published


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
    assert isinstance(tool.context.workspace, InvocationWorkspaceHandle)
    assert tool.context.workspace is not workspace
    assert isinstance(tool.context.artifact_store, InvocationArtifactStoreHandle)
    assert tool.context.artifact_store is not artifact_store
    assert tool.context._authoritative_workspace_for_builtin() is workspace
    assert tool.context._authoritative_artifact_store_for_builtin() is artifact_store
    assert isinstance(tool.context.runner, InvocationRunnerHandle)
    assert tool.context.runner is not runner
    assert not hasattr(tool.context.runner, "close")


def test_builtin_file_tools_use_falsey_raw_runtime_authorities(tmp_path):
    class FalseyWorkspace(LocalWorkspace):
        def __bool__(self) -> bool:
            return False

    class FalseyArtifactStore(LocalArtifactStore):
        def __bool__(self) -> bool:
            return False

    async def exercise() -> tuple[ToolResult, ToolResult]:
        secret = "falsey-resource-secret-canary-ABCDEFGHIJKLMNOP"
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        workspace = FalseyWorkspace(workspace_root, workspace_id="local")
        artifact_store = FalseyArtifactStore(
            tmp_path / "artifacts",
            store_id="artifacts",
        )
        await workspace.write_bytes("secret.txt", secret.encode())
        artifact = await artifact_store.put_bytes(
            secret.encode(),
            filename="secret.txt",
            content_type="text/plain",
            session_id="sess_1",
        )
        tracker = InvocationSecretTracker(SecretRedactor(secret))
        ctx = ToolContext(
            session_id="sess_1",
            workspace=InvocationWorkspaceHandle(
                workspace,
                redactor_snapshot_provider=tracker.snapshot,
                capture_observer=tracker.record_ambiguous_output_capture,
            ),
            artifact_store=InvocationArtifactStoreHandle(
                artifact_store,
                redactor_snapshot_provider=tracker.snapshot,
                capture_observer=tracker.record_ambiguous_output_capture,
            ),
            invocation_secret_redactor=lambda: tracker.redactor,
            invocation_secret_snapshot_provider=tracker.snapshot,
            invocation_secret_capture_observer=tracker.record_ambiguous_output_capture,
        )
        ctx._bind_runtime_resource_authorities(
            workspace=workspace,
            artifact_store=artifact_store,
        )
        workspace_result = await ReadFileTool().run(ctx, {"path": "secret.txt"})
        artifact_result = await ReadFileTool().run(ctx, {"artifact_id": artifact.id})
        return workspace_result, artifact_result

    workspace_result, artifact_result = asyncio.run(exercise())

    assert workspace_result.is_error is False
    assert workspace_result.structured["source"] == "workspace"
    assert artifact_result.is_error is False
    assert artifact_result.structured["source"] == "artifact"
    assert REDACTED_SECRET in workspace_result.content
    assert REDACTED_SECRET in artifact_result.content


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
        "content_sha256": hashlib.sha256(b"image-bytes").hexdigest(),
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
