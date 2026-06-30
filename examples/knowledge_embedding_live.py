from __future__ import annotations

import asyncio
import json
import os

from cayu.providers import OpenAIProvider
from cayu.storage import (
    InMemoryEmbeddingKnowledgeStore,
    KnowledgeEntry,
    KnowledgeQuery,
    KnowledgeSearchMode,
)


async def main() -> None:
    provider_name = os.environ.get("CAYU_PROVIDER", "openai")
    if provider_name != "openai":
        raise SystemExit("knowledge_embedding_live.py currently supports CAYU_PROVIDER=openai.")
    embedding_model = os.environ.get("CAYU_EMBEDDING_MODEL", "text-embedding-3-small")
    dimensions_raw = os.environ.get("CAYU_EMBEDDING_DIMENSIONS")
    dimensions = int(dimensions_raw) if dimensions_raw else None

    provider = OpenAIProvider()
    store = InMemoryEmbeddingKnowledgeStore(
        embedding_provider=provider,
        embedding_model=embedding_model,
        embedding_dimensions=dimensions,
        semantic_min_score=0.70,
    )
    await store.put_entry(
        KnowledgeEntry(
            id="remote_git_credentials",
            kind="procedure",
            labels={"area": "sandbox-git", "project": "cayu"},
            aspects=["credentials", "remote-sandbox", "git"],
            text=(
                "For GitHub clone or push from a remote sandbox, prefer a brokered "
                "Git HTTP proxy. The trusted Cayu side injects the credential outside "
                "the sandbox so the raw token is never present in environment variables, "
                "files, process arguments, or command output."
            ),
        )
    )
    await store.put_entry(
        KnowledgeEntry(
            id="invoice_approval",
            kind="procedure",
            labels={"area": "invoices", "project": "cayu"},
            aspects=["invoices", "approvals"],
            text="Invoice refunds require approval and audit logging before payment.",
        )
    )
    await store.put_entry(
        KnowledgeEntry(
            id="sendgrid_proxy",
            kind="procedure",
            labels={"area": "email", "project": "cayu"},
            aspects=["credentials", "proxy"],
            text=(
                "For SendGrid, prefer a trusted credential proxy that performs the "
                "API request outside the sandbox."
            ),
        )
    )

    query = os.environ.get(
        "CAYU_KNOWLEDGE_QUERY",
        "how should remote sandbox auth work for pushing code to GitHub",
    )
    result = await store.search(
        KnowledgeQuery(
            text=query,
            namespace="default",
            labels={"project": "cayu"},
            mode=KnowledgeSearchMode.SEMANTIC,
            limit=5,
            max_bytes=4_000,
        )
    )

    print("provider", provider_name)
    print("embedding_model", embedding_model)
    print("query", query)
    print(
        "hits",
        json.dumps(
            [
                {
                    "entry_id": hit.entry.id,
                    "score": hit.score,
                    "score_normalized": hit.score_normalized,
                    "score_kind": hit.score_kind,
                    "reason": hit.reason,
                    "preview": hit.text_preview,
                }
                for hit in result.hits
            ],
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
