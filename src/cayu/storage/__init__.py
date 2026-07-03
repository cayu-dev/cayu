"""Storage contracts."""

from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.storage.event_watchers import SQLiteEventWatcherStore
from cayu.storage.knowledge_indexer import (
    DEFAULT_KNOWLEDGE_CHUNK_OVERLAP_BYTES,
    DEFAULT_KNOWLEDGE_CHUNK_TARGET_BYTES,
    DEFAULT_KNOWLEDGE_INDEX_MAX_CHUNKS,
    KnowledgeIndexer,
    KnowledgeIndexRequest,
    KnowledgeIndexResult,
)
from cayu.storage.knowledge_review import KnowledgeReviewWorkflow
from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore
from cayu.storage.memory import (
    BUILTIN_KNOWLEDGE_KINDS,
    DEFAULT_KNOWLEDGE_KIND,
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    DEFAULT_KNOWLEDGE_NAMESPACE,
    InMemoryEmbeddingKnowledgeStore,
    InMemoryKnowledgeStore,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeFacet,
    KnowledgeHit,
    KnowledgeListGroup,
    KnowledgeListItem,
    KnowledgeListQuery,
    KnowledgeListResult,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
    KnowledgeStatus,
    KnowledgeStore,
    KnowledgeVisibility,
)
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore

__all__ = [
    "BUILTIN_KNOWLEDGE_KINDS",
    "DEFAULT_KNOWLEDGE_CHUNK_OVERLAP_BYTES",
    "DEFAULT_KNOWLEDGE_CHUNK_TARGET_BYTES",
    "DEFAULT_KNOWLEDGE_INDEX_MAX_CHUNKS",
    "DEFAULT_KNOWLEDGE_KIND",
    "DEFAULT_KNOWLEDGE_LIMIT",
    "DEFAULT_KNOWLEDGE_MAX_BYTES",
    "DEFAULT_KNOWLEDGE_NAMESPACE",
    "InMemoryEmbeddingKnowledgeStore",
    "InMemoryKnowledgeStore",
    "KnowledgeActorType",
    "KnowledgeChunk",
    "KnowledgeEntry",
    "KnowledgeFacet",
    "KnowledgeHit",
    "KnowledgeIndexRequest",
    "KnowledgeIndexResult",
    "KnowledgeIndexer",
    "KnowledgeListGroup",
    "KnowledgeListItem",
    "KnowledgeListQuery",
    "KnowledgeListResult",
    "KnowledgeQuery",
    "KnowledgeReviewWorkflow",
    "KnowledgeSearchMode",
    "KnowledgeSearchResult",
    "KnowledgeStatus",
    "KnowledgeStore",
    "KnowledgeVisibility",
    "PostgresBudgetLedger",
    "PostgresEmbeddingBackfillResult",
    "PostgresEmbeddingKnowledgeStore",
    "PostgresEventWatcherStore",
    "PostgresKnowledgeStore",
    "PostgresSessionStore",
    "PostgresTaskStore",
    "SQLiteBudgetLedger",
    "SQLiteEventWatcherStore",
    "SQLiteKnowledgeStore",
    "SQLiteSessionStore",
    "SQLiteTaskStore",
]


def __getattr__(name: str):
    # Postgres stores require the optional ``postgres`` extra (psycopg). Import
    # them lazily so the base package import does not depend on psycopg.
    if name in {
        "PostgresBudgetLedger",
        "PostgresEmbeddingBackfillResult",
        "PostgresEmbeddingKnowledgeStore",
        "PostgresEventWatcherStore",
        "PostgresKnowledgeStore",
        "PostgresSessionStore",
        "PostgresTaskStore",
    }:
        from cayu.storage import postgres

        return getattr(postgres, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
