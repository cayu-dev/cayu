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
from cayu.storage.knowledge_transition import (
    KNOWLEDGE_REVISION_RESET_POLICY_VERSION,
    KnowledgeRevisionResetRequired,
    KnowledgeRevisionTransitionAction,
    KnowledgeRevisionTransitionAssessment,
    assess_knowledge_revision_transition,
    require_empty_knowledge_revision_transition,
)
from cayu.storage.memory import (
    BUILTIN_KNOWLEDGE_KINDS,
    DEFAULT_KNOWLEDGE_KIND,
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    DEFAULT_KNOWLEDGE_NAMESPACE,
    MAX_KNOWLEDGE_CHUNK_INDEX,
    InMemoryEmbeddingKnowledgeStore,
    InMemoryKnowledgeStore,
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeChunkConflict,
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
    "KNOWLEDGE_REVISION_RESET_POLICY_VERSION",
    "MAX_KNOWLEDGE_CHUNK_INDEX",
    "InMemoryEmbeddingKnowledgeStore",
    "InMemoryKnowledgeStore",
    "KnowledgeAccessDenied",
    "KnowledgeAccessScope",
    "KnowledgeActorType",
    "KnowledgeChunk",
    "KnowledgeChunkConflict",
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
    "KnowledgeRevisionResetRequired",
    "KnowledgeRevisionTransitionAction",
    "KnowledgeRevisionTransitionAssessment",
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
    "assess_knowledge_revision_transition",
    "prepare_knowledge_publication",
    "require_empty_knowledge_revision_transition",
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
