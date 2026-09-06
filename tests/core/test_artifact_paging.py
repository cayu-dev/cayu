from __future__ import annotations

import asyncio
import json

import pytest
from tests.core.test_tool_result_projection import _collect, _ReadbackProvider, _ResultTool

from cayu import ArtifactScope, Environment, EnvironmentSpec, SecretRedactor
from cayu.artifacts import ArtifactStore, LocalArtifactStore
from cayu.core import AgentSpec, EventType, Message
from cayu.core.tools import ToolContext, ToolResult
from cayu.providers import ModelStreamEvent
from cayu.runtime import CayuApp, RunRequest
from cayu.runtime.tool_result_projection import ArtifactExternalizingToolResultPolicy
from cayu.tools.files import ReadFileTool


def _page_text(result):
    if "\n[read_file " not in result.content:
        return result.content
    text, metadata = result.content.rsplit("\n[read_file ", 1)
    visible = json.loads(metadata[:-1])
    for key in ("offset", "next_offset", "total_bytes", "truncated"):
        assert visible[key] == result.structured[key]
    return text


@pytest.mark.parametrize(
    "text", ["a" * 15, "a" * 16, "a" * 17, "a" * 100 + "TAIL", "a" * 15 + "🙂é" * 30 + "TAIL"]
)
def test_text_artifact_pages_cover_exact_utf8_bytes(tmp_path, text):
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="paging")
    artifact = asyncio.run(
        store.put_bytes(text.encode(), filename="text.txt", session_id="session")
    )
    ctx = ToolContext(session_id="session", artifact_store=store)
    offset = 0
    pages = []
    while True:
        result = asyncio.run(
            ReadFileTool().run(ctx, {"artifact_id": artifact.id, "max_bytes": 16, "offset": offset})
        )
        assert not result.is_error, result
        page = _page_text(result)
        pages.append(page)
        assert len(page.encode()) == result.structured["bytes"] <= 16
        assert result.structured["offset"] == offset
        next_offset = result.structured["next_offset"]
        assert result.structured["truncated"] == (next_offset is not None)
        if next_offset is None:
            break
        assert next_offset == offset + len(page.encode()) > offset
        offset = next_offset
        assert len(pages) <= len(text)
    assert "".join(pages) == text
    eof = asyncio.run(
        ReadFileTool().run(
            ctx, {"artifact_id": artifact.id, "max_bytes": 16, "offset": len(text.encode())}
        )
    )
    assert not eof.is_error and _page_text(eof) == ""
    assert eof.structured["next_offset"] is None
    beyond = asyncio.run(
        ReadFileTool().run(ctx, {"artifact_id": artifact.id, "offset": len(text.encode()) + 1})
    )
    assert beyond.is_error and beyond.structured["error"] == "invalid_arguments"


@pytest.mark.parametrize("scope", [ArtifactScope.SESSION, ArtifactScope.ENVIRONMENT])
@pytest.mark.parametrize("offset", [0, 16])
def test_artifact_pages_preserve_scope_and_missing_behavior(tmp_path, scope, offset):
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="paging")
    artifact = asyncio.run(
        store.put_bytes(
            b"x" * 64,
            filename="text.txt",
            scope=scope,
            session_id="owner" if scope == ArtifactScope.SESSION else None,
            environment_name="owner-env",
        )
    )
    args = {"artifact_id": artifact.id, "offset": offset, "max_bytes": 16}
    denied = asyncio.run(
        ReadFileTool().run(
            ToolContext(session_id="other", environment_name="other-env", artifact_store=store),
            args,
        )
    )
    assert denied.is_error
    assert "x" * 16 not in denied.content
    allowed = asyncio.run(
        ReadFileTool().run(
            ToolContext(session_id="owner", environment_name="owner-env", artifact_store=store),
            args,
        )
    )
    assert not allowed.is_error
    asyncio.run(store.delete(artifact.id))
    missing = asyncio.run(
        ReadFileTool().run(
            ToolContext(session_id="owner", environment_name="owner-env", artifact_store=store),
            args,
        )
    )
    assert missing.is_error and missing.structured["reason"] == "not_found"


def test_artifact_page_rejects_invalid_utf8_offset_and_small_budget(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="paging")
    artifact = asyncio.run(
        store.put_bytes("🙂TAIL".encode(), filename="text.txt", session_id="session")
    )
    ctx = ToolContext(session_id="session", artifact_store=store)
    split = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": artifact.id, "offset": 1}))
    assert split.is_error and "splits a UTF-8" in split.content
    small = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": artifact.id, "max_bytes": 3}))
    assert small.is_error and small.structured["error"] == "text_page_too_small"


def test_prefix_only_store_reports_explicit_continuation_capability(tmp_path):
    class PrefixStore(LocalArtifactStore):
        read_range = ArtifactStore.read_range

    store = PrefixStore(tmp_path / "artifacts", store_id="paging")
    artifact = asyncio.run(store.put_bytes(b"x" * 64, filename="text.txt", session_id="session"))
    ctx = ToolContext(session_id="session", artifact_store=store)
    for offset in (0, 16):
        result = asyncio.run(
            ReadFileTool().run(ctx, {"artifact_id": artifact.id, "max_bytes": 16, "offset": offset})
        )
        assert result.is_error
        assert result.structured["error"] == "artifact_range_unsupported"
        assert "range-capable" in result.content
    complete = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": artifact.id, "max_bytes": 64}))
    assert complete.content == "x" * 64 and not complete.is_error


class _PagingProvider(_ReadbackProvider):
    def __init__(self, max_inline_bytes=2048):
        super().__init__()
        self.pages = []
        self.max_inline_bytes = max_inline_bytes

    async def stream(self, request):
        if len(self.requests) < 2:
            async for event in super().stream(request):
                yield event
            return
        self.requests.append(request)
        part = [
            part
            for message in request.messages
            if message.role == "tool"
            for part in message.content
        ][-1]
        assert not part.is_error, part.content
        assert len(part.content.encode()) <= self.max_inline_bytes
        assert not part.artifacts  # no recursive externalization of a readback page
        text = _page_text(part)
        self.pages.append(text)
        assert len(self.pages) < 20
        if part.structured["next_offset"] is not None:
            assert part.structured["next_offset"] > part.structured["offset"]
            yield ModelStreamEvent.tool_call(
                id=f"page_{len(self.pages)}",
                name="read_file",
                arguments={**self.readback_arguments, "offset": part.structured["next_offset"]},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
        else:
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})


@pytest.mark.parametrize("inline_bytes", [2048, 8192])
def test_runtime_externalization_pages_to_redacted_tail_and_survives_store_reconstruction(
    tmp_path, inline_bytes
):
    secret = "synthetic-private-value"
    original = "a" * 1850 + "🙂é" * 350 + secret + "z" * 5500 + "TAIL-SENTINEL"
    redacted = SecretRedactor(secret).redact_text(original)
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="paging")
    provider = _PagingProvider(inline_bytes)
    app = CayuApp(
        enable_logging=False,
        secret_redactor=SecretRedactor(secret),
        tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
            max_inline_bytes=inline_bytes, max_inline_token_estimate=None, preview_bytes=32
        ),
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=store), default=True
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[_ResultTool(ToolResult(content=original)), ReadFileTool()],
    )
    events = asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    session_id="session",
                    agent_name="assistant",
                    messages=[Message.text("user", "inspect all pages")],
                )
            )
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert "".join(provider.pages) == redacted
    assert provider.pages[-1].endswith("TAIL-SENTINEL")
    assert len(provider.pages) > 1
    for event in events:
        if event.type is EventType.TOOL_CALL_COMPLETED and event.tool_name == "read_file":
            assert event.payload["tool_result_projection"]["status"] == "unchanged"
    reconstructed = LocalArtifactStore(tmp_path / "artifacts", store_id="paging")
    ctx = ToolContext(session_id="session", environment_name="local", artifact_store=reconstructed)
    offset = 0
    observed = []
    for _ in range(20):
        result = asyncio.run(
            ReadFileTool().run(ctx, {**provider.readback_arguments, "offset": offset})
        )
        assert not result.is_error
        observed.append(_page_text(result))
        offset = result.structured["next_offset"]
        if offset is None:
            break
    assert "".join(observed) == redacted
    assert secret not in "".join(observed)


@pytest.mark.parametrize("prefix_size", range(45, 66))
@pytest.mark.parametrize("stored_redacted", [False, True])
def test_artifact_redaction_preserves_complete_spans_at_page_boundaries(
    tmp_path, prefix_size, stored_redacted
):
    secret = "synthetic-private-value"
    redactor = SecretRedactor(secret)
    source = "s" * prefix_size + secret + "🙂" * 24 + "TAIL"
    expected = redactor.redact_text(source)
    stored = expected if stored_redacted else source
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="paging")
    artifact = asyncio.run(
        store.put_bytes(stored.encode(), filename="text.txt", session_id="session")
    )
    ctx = ToolContext(
        session_id="session", artifact_store=store, invocation_secret_redactor=lambda: redactor
    )
    offset = 0
    output = []
    for _ in range(10):
        result = asyncio.run(
            ReadFileTool().run(ctx, {"artifact_id": artifact.id, "offset": offset, "max_bytes": 64})
        )
        assert not result.is_error, result
        output.append(_page_text(result))
        assert len(output[-1].encode()) <= 64
        next_offset = result.structured["next_offset"]
        if next_offset is None:
            break
        assert next_offset == offset + result.structured["bytes"] > offset
        offset = next_offset
    assert "".join(output) == expected


def test_text_continuation_rejects_native_attachment_offset(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="paging")
    artifact = asyncio.run(
        store.put_bytes(
            b"synthetic-image", filename="image.png", content_type="image/png", session_id="session"
        )
    )
    result = asyncio.run(
        ReadFileTool().run(
            ToolContext(session_id="session", artifact_store=store),
            {"artifact_id": artifact.id, "offset": 1},
        )
    )
    assert result.is_error and result.structured["error"] == "invalid_arguments"


def test_disappearing_artifact_range_remains_not_found(tmp_path):
    class DisappearingStore(LocalArtifactStore):
        async def read_range(self, artifact_id, *, offset, max_bytes):
            await self.delete(artifact_id)
            return await super().read_range(artifact_id, offset=offset, max_bytes=max_bytes)

    store = DisappearingStore(tmp_path / "artifacts", store_id="paging")
    artifact = asyncio.run(store.put_bytes(b"x" * 64, filename="text.txt", session_id="session"))
    result = asyncio.run(
        ReadFileTool().run(
            ToolContext(session_id="session", artifact_store=store),
            {"artifact_id": artifact.id, "offset": 16, "max_bytes": 16},
        )
    )
    assert result.is_error and result.structured["reason"] == "not_found"


@pytest.mark.parametrize(
    "secrets,source",
    [(["x"], "x" * 30 + "TAIL"), (["x", "c["], "abcx" * 20 + "TAIL"), (["abc"], "a" * 60 + "TAIL")],
)
def test_redaction_output_expansion_never_skips_source_bytes(tmp_path, secrets, source):
    redactor = SecretRedactor(secrets)
    expected = redactor.redact_text(source)
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="paging")
    artifact = asyncio.run(
        store.put_bytes(source.encode(), filename="text.txt", session_id="session")
    )
    ctx = ToolContext(
        session_id="session", artifact_store=store, invocation_secret_redactor=lambda: redactor
    )
    offset = 0
    output = []
    for _ in range(len(source) + 1):
        result = asyncio.run(
            ReadFileTool().run(ctx, {"artifact_id": artifact.id, "offset": offset, "max_bytes": 32})
        )
        assert not result.is_error, result
        output.append(_page_text(result))
        assert len(output[-1].encode()) <= 32
        next_offset = result.structured["next_offset"]
        if next_offset is None:
            break
        assert next_offset > offset
        offset = next_offset
    assert "".join(output) == expected


def test_redaction_progress_requires_authoritative_framing():
    redactor = SecretRedactor("secret")
    with pytest.raises(ValueError, match="framing"):
        redactor.redact_utf8_page_with_progress(
            b"prefix-sec",
            window_offset=0,
            page_offset=0,
            page_end=10,
            total_bytes=100,
            max_bytes=10,
        )


def test_prefix_adapter_cannot_claim_completion_after_redaction_omits_content(tmp_path):
    from cayu.artifacts import ArtifactReadResult

    class RedactedPrefixStore(LocalArtifactStore):
        read_range = ArtifactStore.read_range

        async def read_bytes(self, artifact_id, *, max_bytes=None):
            result = await super().read_bytes(artifact_id, max_bytes=max_bytes)
            return ArtifactReadResult(
                metadata=result.metadata,
                content=b"",
                total_bytes=result.total_bytes,
                source_bytes_read=result.total_bytes,
                redaction_truncated=True,
            )

    store = RedactedPrefixStore(tmp_path / "artifacts", store_id="paging")
    artifact = asyncio.run(
        store.put_bytes(b"redaction omitted this", filename="text.txt", session_id="session")
    )
    result = asyncio.run(
        ReadFileTool().run(
            ToolContext(session_id="session", artifact_store=store), {"artifact_id": artifact.id}
        )
    )
    assert result.is_error and result.structured["error"] == "artifact_range_unsupported"
