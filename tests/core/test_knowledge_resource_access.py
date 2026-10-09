from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cayu._resource_access_binding import ResourceExecutionBinding
from cayu.resource_access import ResourceAccessPolicy, encode_scope, execution_access
from cayu.sessions.access import SessionAccessRule, SessionAccessScope, SessionAccessSelector
from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeListQuery,
    KnowledgeQuery,
)
from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore


class Policy(ResourceAccessPolicy):
    authority = "knowledge-test"

    def __init__(self, scope):
        self.scope = scope

    async def resolve(self, subject):
        return self.scope


async def conformance(store):
    prefix = uuid4().hex
    shared_namespace = prefix + "shared"
    admin = KnowledgeAccessScope.privileged()
    for suffix, organization, department, namespace in [
        ("a", "acme", "finance", shared_namespace),
        ("b", "other", "finance", shared_namespace),
        ("c", "acme", "legal", shared_namespace),
        ("d", "acme", "private", shared_namespace),
        ("e", "acme", "finance", "excluded"),
    ]:
        await store.create_entry(
            KnowledgeEntry(
                id=prefix + suffix,
                text="credential needle",
                namespace=namespace,
                labels={"organization": organization, "department": department},
            ),
            access_scope=admin,
        )
    embedding = type(store).__name__ in {
        "InMemoryEmbeddingKnowledgeStore",
        "PostgresEmbeddingKnowledgeStore",
    }
    if embedding:
        await store.backfill_embeddings(access_scope=admin)
    rules = tuple(
        SessionAccessRule(
            selectors=(
                SessionAccessSelector(key="organization", values=("acme",)),
                SessionAccessSelector(key="department", values=(department,)),
            )
        )
        for department in ("finance", "legal")
    )
    scope = SessionAccessScope(read=rules, execute=rules)
    policy = Policy(scope)
    binding = ResourceExecutionBinding(
        authority=policy.authority, subject="alice", admitted_json=encode_scope(scope)
    )
    native = KnowledgeAccessScope.for_namespace(shared_namespace)
    if embedding:
        async with execution_access(
            binding, policy, {"organization": "acme", "department": "finance"}
        ):
            semantic = await store.search(
                KnowledgeQuery(text="credential", namespace=shared_namespace, mode="semantic"),
                access_scope=native,
            )
            assert {hit.entry.id for hit in semantic.hits} == {prefix + "a", prefix + "c"}

    from cayu.knowledge.access import ScopedKnowledgeAccess

    handle = ScopedKnowledgeAccess(store, binding=binding, policy=policy, access_scope=native)
    assert await handle.get_entry(prefix + "b") is None
    assert (await handle.get_entry(prefix + "a")).id == prefix + "a"
    assert {
        item.entry.id for item in (await handle.list_entries(KnowledgeListQuery(limit=100))).entries
    } == {prefix + "a", prefix + "c"}

    async with execution_access(binding, policy, {"organization": "acme", "department": "finance"}):
        results = await store.list_entries(KnowledgeListQuery(limit=100), access_scope=native)
        assert {item.entry.id for item in results.entries} == {prefix + "a", prefix + "c"}
        search = await store.search(
            KnowledgeQuery(text="needle", namespace=shared_namespace), access_scope=native
        )
        assert {hit.entry.id for hit in search.hits} == {prefix + "a", prefix + "c"}
        assert await store.get_entry(prefix + "b", access_scope=native) is None
        all_rule = SessionAccessRule(allow_all=True)
        policy.scope = SessionAccessScope(read=(all_rule,), execute=(all_rule,))
        assert await store.get_entry(prefix + "b", access_scope=native) is None
        policy.scope = SessionAccessScope()
        from cayu._resource_access_errors import ResourceAccessDenied

        with pytest.raises(ResourceAccessDenied):
            await store.list_entries(KnowledgeListQuery(), access_scope=native)
    assert (await store.get_entry(prefix + "b", access_scope=admin)).id == prefix + "b"


@pytest.mark.parametrize("backend", ["memory", "sqlite", "memory_embedding"])
def test_knowledge_resource_access(backend, tmp_path):
    async def run():
        if backend == "memory":
            store = InMemoryKnowledgeStore()
        elif backend == "sqlite":
            store = SQLiteKnowledgeStore(tmp_path / "knowledge.db")
        else:
            from tests.core.test_knowledge_store import KeywordEmbeddingProvider

            from cayu.storage.memory import InMemoryEmbeddingKnowledgeStore

            store = InMemoryEmbeddingKnowledgeStore(
                embedding_provider=KeywordEmbeddingProvider(),
                embedding_model="test",
                embedding_dimensions=3,
            )
        try:
            await conformance(store)
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("embedding", [False, True])
def test_postgres_knowledge_resource_access(postgres_dsn, embedding):
    from cayu.storage.knowledge_postgres import PostgresKnowledgeStore
    from cayu.storage.migrations import SchemaMode

    async def run():
        if embedding:
            from tests.core.test_knowledge_store import KeywordEmbeddingProvider

            from cayu.storage.postgres import PostgresEmbeddingKnowledgeStore

            store = PostgresEmbeddingKnowledgeStore(
                postgres_dsn,
                schema_mode=SchemaMode.CREATE,
                embedding_provider=KeywordEmbeddingProvider(),
                embedding_model="test",
                embedding_dimensions=3,
            )
        else:
            store = PostgresKnowledgeStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await conformance(store)
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["keyword", "auto"])
def test_postgres_embedding_bound_scope_keyword_fallback(postgres_dsn, mode):
    from tests.core.test_knowledge_store import KeywordEmbeddingProvider

    from cayu.knowledge.access import ScopedKnowledgeAccess
    from cayu.storage.memory import KnowledgeAccessDenied, KnowledgeRevisionRef
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresEmbeddingKnowledgeStore

    async def run():
        namespace = uuid4().hex
        native = KnowledgeAccessScope.for_namespace(namespace)
        store = PostgresEmbeddingKnowledgeStore(
            postgres_dsn,
            schema_mode=SchemaMode.CREATE,
            access_scope=native,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="test",
            embedding_dimensions=3,
        )
        try:
            for suffix, organization in [("own", "acme"), ("foreign", "other")]:
                await store.create_entry(
                    KnowledgeEntry(
                        id=namespace + suffix,
                        namespace=namespace,
                        text="credential needle",
                        aspects=["credentials"],
                        labels={"organization": organization},
                    )
                )
            frontier = (await store.read_changes()).high_water_sequence
            rules = (
                SessionAccessRule(
                    selectors=(SessionAccessSelector(key="organization", values=("acme",)),)
                ),
            )
            scope = SessionAccessScope(read=rules, execute=rules)
            policy = Policy(scope)
            binding = ResourceExecutionBinding(
                authority=policy.authority,
                subject="alice",
                admitted_json=encode_scope(scope),
            )
            # Aspect-only AUTO queries exercise the no-positive-terms keyword fallback.
            query = KnowledgeQuery(
                text="needle" if mode == "keyword" else None,
                mode=mode,
                namespace=namespace,
                aspect_groups=[["credentials"]] if mode == "auto" else [],
            )
            handle = ScopedKnowledgeAccess(store, binding=binding, policy=policy)
            assert [hit.entry.id for hit in (await handle.search(query)).hits] == [
                namespace + "own"
            ]
            async with execution_access(binding, policy, {"organization": "acme"}):
                results = [
                    await store.search(query),
                    await store.search_at_frontier(
                        query,
                        knowledge_sequence=frontier,
                        index_readiness_sequence=0,
                    ),
                    await store.search_revisions(
                        query,
                        [
                            KnowledgeRevisionRef(entry_id=namespace + suffix, revision=1)
                            for suffix in ("own", "foreign")
                        ],
                        knowledge_sequence=frontier,
                        index_readiness_sequence=0,
                    ),
                ]
                for result in results:
                    assert [hit.entry.id for hit in result.hits] == [namespace + "own"]
                with pytest.raises(KnowledgeAccessDenied, match="access_scope_override"):
                    await store.search(query, access_scope=KnowledgeAccessScope.privileged())
            assert {hit.entry.id for hit in (await store.search(query)).hits} == {
                namespace + "own",
                namespace + "foreign",
            }
        finally:
            await store.close()

    asyncio.run(run())
