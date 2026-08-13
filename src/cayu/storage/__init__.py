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
    MAX_KNOWLEDGE_CHUNK_INDEX,
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
    KnowledgePublicationConflict,
    KnowledgePublicationReceipt,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
    KnowledgeStatus,
    KnowledgeStore,
    KnowledgeVisibility,
    prepare_knowledge_publication,
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
    "MAX_KNOWLEDGE_CHUNK_INDEX",
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
    "KnowledgePublicationConflict",
    "KnowledgePublicationReceipt",
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
    "PostgresEvalStore",
    "PostgresEventWatcherStore",
    "PostgresKnowledgeStore",
    "PostgresSessionStore",
    "PostgresTaskStore",
    "SQLiteBudgetLedger",
    "SQLiteEvalStore",
    "SQLiteEventWatcherStore",
    "SQLiteKnowledgeStore",
    "SQLiteSessionStore",
    "SQLiteTaskStore",
    "prepare_knowledge_publication",
]


def __getattr__(name: str):
    # Postgres stores require the optional ``postgres`` extra (psycopg). Import
    # them lazily so the base package import does not depend on psycopg.
    if name == "SQLiteEvalStore":
        from cayu.storage.evals_sqlite import SQLiteEvalStore

        return SQLiteEvalStore
    if name in {
        "PostgresBudgetLedger",
        "PostgresEmbeddingBackfillResult",
        "PostgresEmbeddingKnowledgeStore",
        "PostgresEvalStore",
        "PostgresEventWatcherStore",
        "PostgresKnowledgeStore",
        "PostgresSessionStore",
        "PostgresTaskStore",
    }:
        if name == "PostgresEvalStore":
            from cayu.storage.evals_postgres import PostgresEvalStore

            return PostgresEvalStore
        from cayu.storage import postgres

        return getattr(postgres, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
