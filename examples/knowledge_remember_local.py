from __future__ import annotations

import asyncio
import json

from cayu import (
    InMemoryKnowledgeStore,
    KnowledgeReviewWorkflow,
    KnowledgeStatus,
    RememberKnowledgePolicy,
    RememberKnowledgeTool,
    SearchKnowledgeTool,
    ToolContext,
)


async def main() -> None:
    store = InMemoryKnowledgeStore()
    ctx = ToolContext(
        session_id="remember-demo",
        agent_name="assistant",
        environment_name="local",
        workspace_id="workspace-demo",
        knowledge_store=store,
    )

    pending_write = await RememberKnowledgeTool(
        policy=RememberKnowledgePolicy(
            default_namespace="project:cayu",
            require_labels={"project": "cayu", "tenant": "trusted"},
        )
    ).run(
        ctx,
        {
            "text": "Remote sandbox Git pushes should use a brokered credential proxy.",
            "title": "Remote sandbox Git credentials",
            "kind": "procedure",
            "aspects": ["git", "credentials"],
        },
    )
    print_json("pending_write", _write_summary(pending_write.structured))

    normal_search = await SearchKnowledgeTool().run(
        ctx,
        {
            "query": "brokered credential proxy",
            "namespace": "project:cayu",
            "labels": {"project": "cayu"},
            "limit": 5,
        },
    )
    print_json("normal_tool_search", _tool_hit_ids(normal_search.structured))

    reviewer = KnowledgeReviewWorkflow(
        store,
        namespace="project:cayu",
        labels={"project": "cayu", "tenant": "trusted"},
    )
    pending_review = await reviewer.list_pending(source_type="tool", limit=5)
    print_json("reviewer_pending_entries", [item.entry.id for item in pending_review.entries])

    approved = await reviewer.approve(pending_review.entries[0].entry.id)
    print_json("approved_pending_entry", {"entry_id": approved.id, "status": approved.status.value})

    approved_search = await SearchKnowledgeTool().run(
        ctx,
        {
            "query": "brokered credential proxy",
            "namespace": "project:cayu",
            "labels": {"project": "cayu"},
            "limit": 5,
        },
    )
    print_json("normal_tool_search_after_review", _tool_hit_ids(approved_search.structured))

    active_write = await RememberKnowledgeTool(
        policy=RememberKnowledgePolicy(
            default_status=KnowledgeStatus.ACTIVE,
            allow_active_writes=True,
            default_namespace="project:cayu",
            require_labels={"project": "cayu"},
        )
    ).run(
        ctx,
        {
            "text": "Invoice refunds require approval and audit logging before payment.",
            "title": "Invoice refund approvals",
            "kind": "procedure",
            "aspects": ["invoices", "approvals"],
        },
    )
    print_json("active_write", _write_summary(active_write.structured))

    active_search = await SearchKnowledgeTool().run(
        ctx,
        {
            "query": "invoice refunds audit logging",
            "namespace": "project:cayu",
            "labels": {"project": "cayu"},
            "limit": 5,
        },
    )
    print_json("normal_tool_search_after_active_write", _tool_hit_ids(active_search.structured))


def print_json(label: str, value) -> None:
    print(label, json.dumps(value, ensure_ascii=False, sort_keys=True))


def _write_summary(structured: dict | None) -> dict:
    if structured is None:
        return {}
    entry = structured["entry"]
    return {
        "entry_id": entry["entry_id"],
        "status": structured["status"],
        "namespace": entry["namespace"],
        "labels": entry["labels"],
        "aspects": entry["aspects"],
    }


def _tool_hit_ids(structured: dict | None) -> list[str]:
    if structured is None:
        return []
    return [hit["entry_id"] for hit in structured["hits"]]


if __name__ == "__main__":
    asyncio.run(main())
