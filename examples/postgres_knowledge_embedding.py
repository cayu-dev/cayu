"""Postgres pgvector-backed knowledge embedding example.

Prerequisites:
    - Postgres with pgvector available
    - OpenAI API credentials for embeddings

Run against an existing database:
    OPENAI_API_KEY=... \
    CAYU_POSTGRES_DSN=postgresql://postgres:postgres@127.0.0.1:5432/cayu \
    CAYU_PROVIDER=openai \
    PYTHONPATH=src .venv/bin/python examples/postgres_knowledge_embedding.py

The script upserts three deterministic demo knowledge entries in namespace
"demo-postgres-embedding"; it does not delete existing data.

The keyword seed store writes normal Postgres knowledge first. The embedding
store then uses schema_mode=CREATE, so the database role must be able to run
CREATE EXTENSION vector or the extension must already be installed.
"""

from __future__ import annotations

import asyncio
import json
import os

from cayu.providers import OpenAIProvider
from cayu.storage import (
    KnowledgeEntry,
    KnowledgeListQuery,
    KnowledgeQuery,
    KnowledgeSearchMode,
    PostgresEmbeddingKnowledgeStore,
    PostgresKnowledgeStore,
)
from cayu.storage.migrations import SchemaMode

NAMESPACE = "demo-postgres-embedding"
PROJECT_LABEL = {"project": "cayu-demo"}


async def main() -> None:
    provider_name = os.environ.get("CAYU_PROVIDER", "openai")
    if provider_name != "openai":
        raise SystemExit("postgres_knowledge_embedding.py currently supports CAYU_PROVIDER=openai.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to run this live embedding example.")

    dsn = os.environ.get("CAYU_POSTGRES_DSN") or os.environ.get("CAYU_TEST_POSTGRES_DSN")
    if not dsn:
        raise SystemExit("Set CAYU_POSTGRES_DSN or CAYU_TEST_POSTGRES_DSN.")

    embedding_model = os.environ.get("CAYU_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimensions = _embedding_dimensions(embedding_model)
    refresh_existing = os.environ.get("CAYU_REFRESH_EMBEDDINGS") == "1"

    seed_stats = await _seed_keyword_knowledge(dsn)

    provider = OpenAIProvider()
    store = PostgresEmbeddingKnowledgeStore(
        dsn,
        min_size=1,
        max_size=4,
        schema_mode=SchemaMode.CREATE,
        embedding_provider=provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        semantic_min_score=0.65,
    )
    try:
        backfill = await store.backfill_embeddings(
            KnowledgeListQuery(namespace=NAMESPACE, labels=PROJECT_LABEL),
            limit=100,
            refresh_existing=refresh_existing,
        )
        semantic = await store.search(
            KnowledgeQuery(
                text=os.environ.get(
                    "CAYU_KNOWLEDGE_QUERY",
                    "how should remote sandbox git auth work",
                ),
                namespace=NAMESPACE,
                labels=PROJECT_LABEL,
                mode=KnowledgeSearchMode.SEMANTIC,
                limit=5,
                max_bytes=4_000,
            )
        )
        hybrid = await store.search(
            KnowledgeQuery(
                text="GitHub credential proxy",
                namespace=NAMESPACE,
                labels=PROJECT_LABEL,
                mode=KnowledgeSearchMode.HYBRID,
                limit=5,
                max_bytes=4_000,
            )
        )
    finally:
        await store.close()

    print("provider", provider_name)
    print("embedding_model", embedding_model)
    print("embedding_dimensions", embedding_dimensions)
    print("seed", json.dumps(seed_stats, sort_keys=True))
    print(
        "backfill",
        json.dumps(
            {
                "scanned_chunks": backfill.scanned_chunks,
                "embedded_chunks": backfill.embedded_chunks,
                "skipped_current_chunks": backfill.skipped_current_chunks,
                "refresh_existing": backfill.refresh_existing,
            },
            sort_keys=True,
        ),
    )
    _print_hits("semantic_hits", semantic)
    _print_hits("hybrid_hits", hybrid)


async def _seed_keyword_knowledge(dsn: str) -> dict[str, int]:
    store = PostgresKnowledgeStore(
        dsn,
        min_size=1,
        max_size=4,
        schema_mode=SchemaMode.CREATE,
    )
    inserted_or_updated = 0
    unchanged = 0
    try:
        for entry in _entries():
            existing = await store.get_entry(entry.id)
            if existing is not None and _same_demo_entry(existing, entry):
                unchanged += 1
                continue
            await store.put_entry(entry)
            inserted_or_updated += 1
    finally:
        await store.close()
    return {"inserted_or_updated": inserted_or_updated, "unchanged": unchanged}


def _entries() -> list[KnowledgeEntry]:
    return [
        KnowledgeEntry(
            id="demo_remote_git_credentials",
            namespace=NAMESPACE,
            labels={**PROJECT_LABEL, "area": "sandbox-git"},
            kind="procedure",
            aspects=["credentials", "remote-sandbox", "git"],
            text=(
                "For GitHub clone or push from a remote sandbox, prefer a brokered "
                "Git HTTP proxy. The trusted Cayu side forwards Git smart HTTP "
                "requests to GitHub and injects the credential outside the sandbox, "
                "so the raw token is never present in environment variables, files, "
                "process arguments, or command output."
            ),
        ),
        KnowledgeEntry(
            id="demo_sendgrid_proxy",
            namespace=NAMESPACE,
            labels={**PROJECT_LABEL, "area": "email"},
            kind="procedure",
            aspects=["credentials", "proxy"],
            text=(
                "For SendGrid, prefer a trusted credential proxy that performs the "
                "API request outside the sandbox. Do not expose the SendGrid API key "
                "through generic shell access."
            ),
        ),
        KnowledgeEntry(
            id="demo_invoice_approval",
            namespace=NAMESPACE,
            labels={**PROJECT_LABEL, "area": "invoices"},
            kind="procedure",
            aspects=["invoices", "approvals"],
            text="Invoice refunds require approval and audit logging before payment.",
        ),
    ]


def _same_demo_entry(existing: KnowledgeEntry, expected: KnowledgeEntry) -> bool:
    return (
        existing.namespace == expected.namespace
        and existing.labels == expected.labels
        and existing.kind == expected.kind
        and sorted(existing.aspects) == sorted(expected.aspects)
        and existing.text == expected.text
        and existing.visibility == expected.visibility
        and existing.status == expected.status
    )


def _embedding_dimensions(model: str) -> int:
    raw = os.environ.get("CAYU_EMBEDDING_DIMENSIONS")
    if raw:
        return int(raw)
    if model == "text-embedding-3-large":
        return 3072
    if model == "text-embedding-3-small":
        return 1536
    raise SystemExit(
        "Set CAYU_EMBEDDING_DIMENSIONS for this embedding model. "
        "Postgres pgvector columns require an explicit vector dimension."
    )


def _print_hits(label: str, result) -> None:
    print(
        label,
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
