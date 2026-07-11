"""Verified live OpenAI embedding and semantic-retrieval contract."""

from __future__ import annotations

import asyncio
import json
import os

from _live_checks import require
from cayu.embeddings import TextEmbeddingProvider
from cayu.providers import OpenAIProvider
from cayu.storage import (
    InMemoryEmbeddingKnowledgeStore,
    KnowledgeEntry,
    KnowledgeHit,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
)

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
QUERY = "which entry describes a brokered Git HTTP proxy for remote Git credentials"
EXPECTED_TOP_ENTRY_ID = "remote_git_credentials"
SEMANTIC_MIN_SCORE = 0.70


async def main() -> None:
    provider_name = os.environ.get("CAYU_PROVIDER", "openai").strip().lower()
    if provider_name != "openai":
        raise SystemExit("knowledge_embedding_live.py currently supports CAYU_PROVIDER=openai.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to run knowledge_embedding_live.py.")
    embedding_model = os.environ.get("CAYU_EMBEDDING_MODEL", "text-embedding-3-small")
    dimensions_raw = os.environ.get("CAYU_EMBEDDING_DIMENSIONS")
    dimensions = int(dimensions_raw) if dimensions_raw else None

    provider = OpenAIProvider()
    evidence = await _run_contract(
        provider=provider,
        embedding_model=embedding_model,
        dimensions=dimensions,
    )

    print("provider", provider_name)
    print("embedding_model", embedding_model)
    print("query", QUERY)
    print(EVIDENCE_PREFIX + json.dumps(evidence, sort_keys=True))
    print("status ok")


async def _run_contract(
    *,
    provider: TextEmbeddingProvider,
    embedding_model: str,
    dimensions: int | None,
) -> dict[str, object]:
    store = InMemoryEmbeddingKnowledgeStore(
        embedding_provider=provider,
        embedding_model=embedding_model,
        embedding_dimensions=dimensions,
        semantic_min_score=SEMANTIC_MIN_SCORE,
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

    result = await store.search(
        KnowledgeQuery(
            text=QUERY,
            namespace="default",
            labels={"project": "cayu"},
            mode=KnowledgeSearchMode.SEMANTIC,
            limit=5,
            max_bytes=4_000,
        )
    )

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
    top_hit = _validate_search_result(result)
    return {
        "provider": provider.name,
        "embedding_model": embedding_model,
        "top_entry_id": top_hit.entry.id,
        "score_kind": top_hit.score_kind,
        "score_normalized": top_hit.score_normalized,
        "hit_count": len(result.hits),
    }


def _validate_search_result(result: KnowledgeSearchResult) -> KnowledgeHit:
    require(bool(result.hits), "semantic search returned no hits")
    top_hit = result.hits[0]
    require(
        top_hit.entry.id == EXPECTED_TOP_ENTRY_ID,
        f"expected top semantic hit {EXPECTED_TOP_ENTRY_ID!r}, got {top_hit.entry.id!r}",
    )
    require(
        top_hit.score_kind == "inmemory_semantic",
        f"expected semantic score kind, got {top_hit.score_kind!r}",
    )
    require(
        top_hit.reason in {"semantic entry match", "semantic chunk match"},
        f"expected semantic match reason, got {top_hit.reason!r}",
    )
    return top_hit


if __name__ == "__main__":
    asyncio.run(main())
