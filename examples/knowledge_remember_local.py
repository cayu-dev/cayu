from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from cayu import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeActivationDecision,
    KnowledgeActivationDisposition,
    KnowledgeActivationRequest,
    KnowledgeGovernanceConfig,
    KnowledgeGovernanceMode,
    KnowledgeReviewWorkflow,
    KnowledgeStatus,
    RememberKnowledgePolicy,
    RememberKnowledgeTool,
    SearchKnowledgeTool,
    ToolContext,
)


class DemoActivationPolicy:
    async def decide_activation(
        self,
        request: KnowledgeActivationRequest,
    ) -> KnowledgeActivationDecision:
        disposition = (
            KnowledgeActivationDisposition.ACTIVATE
            if request.mode is KnowledgeGovernanceMode.POLICY_AUTOMATIC
            else KnowledgeActivationDisposition.REJECT
        )
        return KnowledgeActivationDecision(
            request_sha256=request.fingerprint,
            disposition=disposition,
            policy_identity="example.invoice-policy",
            policy_version="1",
            code=(
                "trusted_demo_scope"
                if disposition is KnowledgeActivationDisposition.ACTIVATE
                else "autonomous_demo_rejected"
            ),
        )


async def main() -> None:
    store = InMemoryKnowledgeStore(
        access_scope=KnowledgeAccessScope.for_namespace(
            "project:cayu",
            required_labels={"project": "cayu", "tenant": "trusted"},
            allowed_statuses=[KnowledgeStatus.PENDING, KnowledgeStatus.ACTIVE],
        )
    )
    ctx = ToolContext(
        session_id="remember-demo",
        agent_name="assistant",
        environment_name="local",
        workspace_id="workspace-demo",
        knowledge_store=store,
    )

    async with RememberKnowledgeTool(
        policy=RememberKnowledgePolicy(
            default_namespace="project:cayu",
            require_labels={"project": "cayu", "tenant": "trusted"},
        )
    ) as remember:
        pending_write = await remember.run(
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

    approved = await reviewer.approve(
        pending_review.entries[0].entry.id,
        operation_id="example-reviewed-remember",
        reviewer_identity="example-reviewer",
        reviewer_version="1",
    )
    print_json(
        "approved_pending_entry",
        {"entry_id": approved.entry.id, "status": approved.entry.status.value},
    )

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

    async with RememberKnowledgeTool(
        policy=RememberKnowledgePolicy(
            governance=KnowledgeGovernanceConfig(
                mode=KnowledgeGovernanceMode.POLICY_AUTOMATIC,
                policy_identity="example.invoice-policy",
                policy_version="1",
            ),
            default_namespace="project:cayu",
            require_labels={"project": "cayu", "tenant": "trusted"},
        ),
        activation_policy=DemoActivationPolicy(),
    ) as remember:
        active_write = await remember.run(
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

    async with RememberKnowledgeTool(
        policy=RememberKnowledgePolicy(
            governance=KnowledgeGovernanceConfig(
                mode=KnowledgeGovernanceMode.AUTONOMOUS,
                policy_identity="example.invoice-policy",
                policy_version="1",
            ),
            default_namespace="project:cayu",
            require_labels={"project": "cayu", "tenant": "trusted"},
        ),
        activation_policy=DemoActivationPolicy(),
    ) as remember:
        rejected_write = await remember.run(
            ctx,
            {
                "text": "Autonomous demo candidates still need application authorization.",
                "title": "Autonomous governance",
                "kind": "fact",
                "aspects": ["governance"],
            },
        )
    print_json("autonomous_rejected_write", _write_summary(rejected_write.structured))


def print_json(label: str, value) -> None:
    print(label, json.dumps(value, ensure_ascii=False, sort_keys=True))


def _write_summary(structured: Mapping[str, Any] | None) -> dict[str, Any]:
    if structured is None:
        return {}
    entry = structured["entry"]
    if entry is None:
        return {
            "entry_id": None,
            "status": structured["status"],
            "written": structured["written"],
            "already_known": structured["already_known"],
            "activation_disposition": structured["activation_disposition"],
            "activation_code": structured["activation_code"],
        }
    return {
        "entry_id": entry["entry_id"],
        "status": structured["status"],
        "written": structured["written"],
        "already_known": structured["already_known"],
    }


def _tool_hit_ids(structured: Mapping[str, Any] | None) -> list[str]:
    if structured is None:
        return []
    return [hit["entry_id"] for hit in structured["hits"]]


if __name__ == "__main__":
    asyncio.run(main())
