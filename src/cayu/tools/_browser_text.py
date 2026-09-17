"""Bounded, session-scoped readback of immutable browser text artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from cayu.artifacts.base import ArtifactScope, copy_artifact_read_result
from cayu.tools.base import ToolContext, ToolResult
from cayu.tools.browser import _screenshot_artifact_store


async def read_browser_text(
    ctx: ToolContext, args: dict[str, Any], *, max_artifact_bytes: int
) -> ToolResult:
    required = {"operation", "artifact_id", "session_id", "page_id", "expected_revision"}
    allowed = required | {"operation_id", "offset", "max_bytes", "query"}
    try:
        if required - args.keys() or args.keys() - allowed:
            raise ValueError
        for key in required:
            if type(args[key]) is not str or not 1 <= len(args[key]) <= 128:
                raise ValueError
        offset, limit = args.get("offset", 0), args.get("max_bytes", 16384)
        query = args.get("query")
        if type(offset) is not int or not 0 <= offset <= max_artifact_bytes:
            raise ValueError
        if type(limit) is not int or not 4 <= limit <= 65536:
            raise ValueError
        if query is not None and (type(query) is not str or not 1 <= len(query) <= 1024):
            raise ValueError
    except (TypeError, ValueError):
        return ToolResult(content="Invalid browser text read arguments.", is_error=True)
    try:
        store = _screenshot_artifact_store(ctx)
        if store is None:
            raise ValueError
        result = copy_artifact_read_result(
            await store.read_bytes(args["artifact_id"], max_bytes=max_artifact_bytes + 1),
            expected_artifact_id=args["artifact_id"],
            max_content_bytes=max_artifact_bytes + 1,
        )
        meta = result.metadata
        source = dict(meta.metadata["source"])
        if (
            meta.scope is not ArtifactScope.SESSION
            or meta.session_id != ctx.session_id
            or meta.metadata.get("kind") != "rendered_text"
            or result.truncated
            or len(result.content) > max_artifact_bytes
            or hashlib.sha256(result.content).hexdigest() != source["content_sha256"]
            or source["session_id"] != args["session_id"]
            or source["page_id"] != args["page_id"]
            or source["revision"] != args["expected_revision"]
        ):
            raise ValueError
        raw = result.content
        raw.decode("utf-8", errors="strict")
        # Offsets are UTF-8 byte positions; callers must use the returned continuation.
        raw[:offset].decode("utf-8", errors="strict")
        if offset > len(raw):
            raise ValueError
        match_offset = None
        if query is not None:
            match_offset = raw.find(query.encode("utf-8"), offset)
            offset = len(raw) if match_offset < 0 else match_offset
        text = raw[offset : offset + limit].decode("utf-8", errors="ignore")
        end = offset + len(text.encode("utf-8"))
        evidence = {
            "artifact_id": meta.id,
            "source": source,
            "historical_evidence": True,
            "offset": offset,
            "next_offset": end if end < len(raw) else None,
            "match_offset": match_offset,
            "text": text,
        }
        return ToolResult(
            content=json.dumps(evidence, ensure_ascii=False),
            structured=evidence,
        )
    except (KeyError, TypeError, ValueError, OSError, RuntimeError):
        return ToolResult(
            content="Browser text evidence is unavailable or source identity mismatches.",
            is_error=True,
        )
