"""Storage contracts."""

from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.storage.event_watchers import SQLiteEventWatcherStore
from cayu.storage.memory import (
    BUILTIN_KNOWLEDGE_KINDS,
    DEFAULT_KNOWLEDGE_KIND,
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    DEFAULT_KNOWLEDGE_NAMESPACE,
    InMemoryKnowledgeStore,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeHit,
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
    "DEFAULT_KNOWLEDGE_KIND",
    "DEFAULT_KNOWLEDGE_LIMIT",
    "DEFAULT_KNOWLEDGE_MAX_BYTES",
    "DEFAULT_KNOWLEDGE_NAMESPACE",
    "InMemoryKnowledgeStore",
    "KnowledgeActorType",
    "KnowledgeChunk",
    "KnowledgeEntry",
    "KnowledgeHit",
    "KnowledgeQuery",
    "KnowledgeSearchMode",
    "KnowledgeSearchResult",
    "KnowledgeStatus",
    "KnowledgeStore",
    "KnowledgeVisibility",
    "PostgresEventWatcherStore",
    "PostgresSessionStore",
    "PostgresTaskStore",
    "SQLiteBudgetLedger",
    "SQLiteEventWatcherStore",
    "SQLiteSessionStore",
    "SQLiteTaskStore",
]


def __getattr__(name: str):
    # Postgres stores require the optional ``postgres`` extra (psycopg). Import
    # them lazily so the base package import does not depend on psycopg.
    if name in {"PostgresEventWatcherStore", "PostgresSessionStore", "PostgresTaskStore"}:
        from cayu.storage import postgres

        return getattr(postgres, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
