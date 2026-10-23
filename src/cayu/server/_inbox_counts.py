"""Exact inbox counts from the same bounded pages that readers can list."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from hashlib import sha256

from fastapi import HTTPException

from cayu.server.inbox import InboxCounts, InboxItemKind, InboxPage


async def count_inbox(read_page: Callable[[str | None], Awaitable[InboxPage]]) -> InboxCounts:
    """Traverse to exhaustion; never publish a capped or incomplete subtotal.

    The inbox is live, so a moved item can be encountered twice. Deduplicate by
    its exact pause/proposal identity; conflicting representations mean this
    traversal cannot establish a coherent total. Cancellation and source errors
    propagate without publishing a partial count.
    """
    counts: dict[InboxItemKind, int] = {"approval": 0, "question": 0, "recovery": 0, "knowledge": 0}
    seen: dict[str, bytes] = {}
    positions: set[str] = set()
    cursor = None
    while True:
        page = await read_page(cursor)
        if page.issues:
            return InboxCounts.unavailable("source_incomplete")
        for item in page.items:
            fingerprint = sha256(item.model_dump_json().encode("utf-8")).digest()
            previous = seen.get(item.id)
            if previous is not None:
                if previous != fingerprint:
                    return InboxCounts.unavailable("source_incomplete")
                continue
            seen[item.id] = fingerprint
            counts[item.kind] += 1
        cursor = page.next_cursor
        if cursor is None:
            if page.has_more:
                return InboxCounts.unavailable("source_incomplete")
            return InboxCounts(counts=counts, total_count=sum(counts.values()), complete=True)
        if cursor in positions:
            raise HTTPException(
                status_code=409,
                detail="Inbox count scan could not advance.",
                headers={"Cache-Control": "private, no-store"},
            )
        positions.add(cursor)
        # In-memory adapters may complete awaits synchronously; yield between pages.
        await asyncio.sleep(0)
