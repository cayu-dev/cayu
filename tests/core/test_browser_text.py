from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace

import pytest
from tests.core.test_browser_session import _context, _durable_context, _FakeBrowserBackend

from cayu.artifacts.local import LocalArtifactStore
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions.base import RunRequest, SessionIdentity
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.browser_session import (
    BrowserArtifactPayload,
    BrowserRenderedTextEvidence,
    BrowserSessionTool,
)

TEXT = ("early\n" + "é漢🙂" * 30000 + "\nlate decisive text").encode()


class TextBackend(_FakeBrowserBackend):
    async def execute(self, ctx, request):
        if request["operation"] != "export_text":
            return await super().execute(ctx, request)
        response = await super().execute(ctx, {**request, "operation": "screenshot"})
        evidence = BrowserRenderedTextEvidence(
            observed_at="2026-09-17T00:00:00+00:00",
            content_sha256=hashlib.sha256(TEXT).hexdigest(),
            size_bytes=len(TEXT),
            scope="main_document_light_dom",
            method="innerText",
            truncated=False,
            omitted_frames=0,
            complete=True,
        )
        return replace(
            response,
            observation=response.observation.model_copy(update={"rendered_text": evidence}),
            artifacts=(
                BrowserArtifactPayload(
                    kind="rendered_text",
                    filename="browser-text.txt",
                    content_type="text/plain",
                    content=TEXT,
                ),
            ),
        )


def test_text_replay_sqlite_reopen_and_source_bound_utf8_readback(tmp_path):
    async def scenario():
        path = tmp_path / "sessions.sqlite"
        store = SQLiteSessionStore(path)
        await store.create(
            RunRequest(
                session_id="parent-session",
                agent_name="assistant",
                messages=[Message.text("user", "read")],
            ),
            identity=SessionIdentity(provider_name="fixture", model="fixture"),
            interaction_started_event=Event(
                id="start",
                type=EventType.INTERACTION_STARTED,
                session_id="parent-session",
                interaction_id="interaction",
                agent_name="assistant",
            ),
            interaction_source_messages=[Message.text("user", "read")],
        )
        records = {}
        backend = TextBackend()
        tool = BrowserSessionTool._from_backend_for_testing(backend)
        opened_args = {
            "operation": "navigate",
            "operation_id": "open",
            "url": "https://example.test/",
        }
        opened = await tool.run(
            _durable_context(tmp_path, args=opened_args, records=records, session_store=store),
            opened_args,
        )
        assert not opened.is_error, opened
        state = opened.structured
        export_args = {
            "operation": "export_text",
            "operation_id": "export",
            "session_id": state["session_id"],
            "page_id": state["page_id"],
            "expected_revision": state["revision"],
            "expected_control_epoch": state["control_epoch"],
        }
        exported = await tool.run(
            _durable_context(
                tmp_path,
                args=export_args,
                records=records,
                session_store=store,
                tool_call_id="export",
            ),
            export_args,
        )
        assert not exported.is_error, exported
        await store.close()
        store = SQLiteSessionStore(path)
        try:
            replay = await BrowserSessionTool._from_backend_for_testing(TextBackend()).run(
                _durable_context(
                    tmp_path,
                    args=export_args,
                    records=records,
                    session_store=store,
                    tool_call_id="export",
                ),
                export_args,
            )
            assert not replay.is_error, replay
            assert replay.artifacts == exported.artifacts
            artifact = replay.artifacts[0]
            assert artifact["artifact_id"] in replay.content
            ctx = _context(
                tmp_path,
                artifact_store=LocalArtifactStore(
                    tmp_path / "artifacts", store_id="browser-artifacts"
                ),
            )
            args = {
                "operation": "read_text",
                "artifact_id": artifact["artifact_id"],
                "session_id": state["session_id"],
                "page_id": state["page_id"],
                "expected_revision": artifact["source"]["revision"],
            }
            reader = BrowserSessionTool()
            offset = 6
            chunks = []
            for _ in range(3):
                page = await reader.run(ctx, {**args, "offset": offset, "max_bytes": 7})
                assert not page.is_error, page
                chunks.append(page.structured["text"])
                offset = page.structured["next_offset"]
            assert "".join(chunks).encode() == TEXT[6:offset]
            late = await reader.run(ctx, {**args, "query": "late decisive"})
            assert "late decisive text" in late.content
            assert late.structured["source"] == artifact["source"]
            for key in ("session_id", "page_id", "expected_revision"):
                assert (await reader.run(ctx, {**args, key: "wrong"})).is_error
            assert (await reader.run(ctx.model_copy(update={"session_id": "other"}), args)).is_error
            assert (await reader.run(ctx, {**args, "offset": 7})).is_error
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes", [{"truncated": True}, {"omitted_frames": 1}, {"method": "document_textContent"}]
)
def test_incomplete_text_cannot_claim_complete(changes):
    with pytest.raises(ValueError, match="coverage"):
        BrowserRenderedTextEvidence.model_validate(
            {
                "observed_at": "2026-09-17T00:00:00+00:00",
                "content_sha256": "a" * 64,
                "size_bytes": 1,
                "scope": "main_document_light_dom",
                "method": "innerText",
                "truncated": False,
                "omitted_frames": 0,
                "complete": True,
                **changes,
            }
        )
