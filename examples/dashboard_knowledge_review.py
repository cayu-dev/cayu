"""Run a local dashboard seeded with pending knowledge review entries.

Usage:
    PYTHONPATH=src .venv/bin/python examples/dashboard_knowledge_review.py
    # Open http://127.0.0.1:8000/cayu/knowledge
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import uvicorn

from cayu import (
    CayuApp,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeStatus,
    SQLiteKnowledgeStore,
    SQLiteSessionStore,
    SQLiteTaskStore,
)
from cayu.server import ServerConfig, create_server

WORKSPACE = Path(__file__).parent / ".examples-workspaces" / "dashboard-knowledge-review"
DATA_DIR = WORKSPACE / "data"


async def seed_knowledge(store: SQLiteKnowledgeStore) -> None:
    await store.put_entry_with_chunks(
        KnowledgeEntry(
            id="pending_remote_git_credentials",
            namespace="project:cayu",
            status=KnowledgeStatus.PENDING,
            kind="procedure",
            title="Remote sandbox Git credentials",
            text=(
                "Remote sandbox Git clone and push operations should use a brokered Git HTTP "
                "proxy. The trusted Cayu side forwards Git smart HTTP traffic to GitHub and "
                "injects the credential outside the sandbox boundary, so the raw token is never "
                "present in sandbox environment variables, files, process arguments, or command "
                "output."
            ),
            labels={"project": "cayu", "area": "sandbox-git"},
            aspects=["git", "credentials", "remote-sandbox"],
            impact_targets=["sandbox.git.push"],
            source_type="tool",
            source_id="remember_01",
            created_by_type=KnowledgeActorType.MODEL,
            created_by="assistant",
            confidence=0.91,
            metadata={
                "session_id": "sess_visual_runtime_recall_9f42",
                "tool_call_id": "call_remember_git",
                "evidence": "Captured after a successful knowledge recall trace.",
            },
        ),
        [
            KnowledgeChunk(
                id="pending_remote_git_credentials:0",
                entry_id="pending_remote_git_credentials",
                chunk_index=0,
                text=(
                    "Remote sandbox Git clone and push operations should use a brokered "
                    "Git HTTP proxy."
                ),
            ),
            KnowledgeChunk(
                id="pending_remote_git_credentials:1",
                entry_id="pending_remote_git_credentials",
                chunk_index=1,
                text=(
                    "Credential injection must happen on the trusted Cayu side, outside the "
                    "sandbox boundary."
                ),
            ),
        ],
    )
    await store.put_entry_with_chunks(
        KnowledgeEntry(
            id="pending_invoice_refund_policy",
            namespace="project:cayu",
            status=KnowledgeStatus.PENDING,
            kind="policy",
            title="Invoice refund approval policy",
            text=(
                "Invoice refunds over 500 USD require approval from finance operations before "
                "payment. The workflow should record the approver, refund reason, invoice id, "
                "and audit timestamp before marking the refund ready for settlement."
            ),
            labels={"project": "cayu", "area": "invoices"},
            aspects=["invoices", "approvals", "audit"],
            impact_targets=["invoice.refund"],
            source_type="tool",
            source_id="remember_02",
            created_by_type=KnowledgeActorType.MODEL,
            created_by="assistant",
            confidence=0.78,
            metadata={"session_id": "sess_invoice_refund_review"},
        ),
        [
            KnowledgeChunk(
                id="pending_invoice_refund_policy:0",
                entry_id="pending_invoice_refund_policy",
                chunk_index=0,
                text="Invoice refunds over 500 USD require finance operations approval.",
            ),
            KnowledgeChunk(
                id="pending_invoice_refund_policy:1",
                entry_id="pending_invoice_refund_policy",
                chunk_index=1,
                text="Audit data must include approver, reason, invoice id, and timestamp.",
            ),
        ],
    )
    await store.put_entry(
        KnowledgeEntry(
            id="active_sendgrid_proxy",
            namespace="project:cayu",
            status=KnowledgeStatus.ACTIVE,
            kind="procedure",
            title="SendGrid credential proxy",
            text="SendGrid requests should run through a trusted credential proxy.",
            labels={"project": "cayu", "area": "email"},
            aspects=["email", "credentials"],
            source_type="fixture",
        )
    )


def main() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    database = DATA_DIR / "cayu.db"
    session_store = SQLiteSessionStore(database)
    task_store = SQLiteTaskStore(database)
    knowledge_store = SQLiteKnowledgeStore(database)
    asyncio.run(seed_knowledge(knowledge_store))

    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        knowledge_store=knowledge_store,
    )
    server = create_server(app, config=ServerConfig.local_development())
    uvicorn.run(
        server,
        host=os.environ.get("CAYU_DASHBOARD_HOST", "127.0.0.1"),
        port=int(os.environ.get("CAYU_DASHBOARD_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
