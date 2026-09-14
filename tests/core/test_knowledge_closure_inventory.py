import asyncio
import json

import pytest
from tests.core.test_knowledge_store import KeywordEmbeddingProvider

from cayu.storage._knowledge_closure import (
    KnowledgeClosureInventory,
    KnowledgeClosureQuery,
)
from cayu.storage.memory import (
    InMemoryEmbeddingKnowledgeStore,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeEvidence,
)


def test_knowledge_artifact_references_survive_partial_closure_retry(tmp_path):
    from tests.core.session_closure_conformance import create_closure_session

    from cayu import CayuApp, LocalArtifactStore
    from cayu.runtime.session_closure import ArtifactSessionClosureStore

    class FailingArtifacts(LocalArtifactStore):
        calls = 0

        async def delete_session_closure_artifact(self, claim, artifact_id):
            self.calls += 1
            if self.calls == 2:
                raise OSError("second artifact deletion failed")
            await super().delete_session_closure_artifact(claim, artifact_id)

    async def run():
        artifacts = FailingArtifacts(tmp_path / "artifacts")
        knowledge = InMemoryKnowledgeStore(access_scope=KnowledgeAccessScope.privileged())
        app = CayuApp(
            knowledge_store=knowledge,
            session_closure_stores=(ArtifactSessionClosureStore(artifacts),),
        )
        await create_closure_session(app.session_store, "artifact-source-session")
        items = [
            await artifacts.put_bytes(
                b"owned", filename=f"{i}.txt", session_id="artifact-source-session"
            )
            for i in range(2)
        ]
        await knowledge.create_entry(
            KnowledgeEntry(id="artifact-derived", text="shared"),
            evidence=[
                KnowledgeEvidence(
                    id=f"artifact-source-{i}",
                    entry_id="artifact-derived",
                    source_type="artifact",
                    source_id=item.id,
                    source_hash="hash",
                )
                for i, item in enumerate(items)
            ],
        )
        first = await app.erase_session_closure("artifact-source-session")
        assert not first.complete
        assert (await artifacts.list(session_id="artifact-source-session")).total_count == 1
        resumed = CayuApp(
            session_store=app.session_store,
            knowledge_store=knowledge,
            session_closure_stores=(
                ArtifactSessionClosureStore(LocalArtifactStore(artifacts.root)),
            ),
        )
        inspection = await resumed.inspect_session_closure("artifact-source-session")
        assert (
            next(item for item in inspection.records if item.store_id == "knowledge-store").count
            == 2
        )
        result = await resumed.erase_session_closure("artifact-source-session")
        assert result.complete
        assert (
            next(
                item for item in result.manifest.records if item.store_id == "knowledge-store"
            ).count
            == 2
        )
        assert (await knowledge.read_evidence("artifact-derived")).total_evidence_known == 2

    asyncio.run(run())


def test_public_closure_refuses_unsupported_knowledge_inventory():
    from tests.core.session_closure_conformance import create_closure_session

    from cayu import CayuApp

    class UnsupportedKnowledge(InMemoryKnowledgeStore):
        async def inspect_closure_sources(self, query):
            raise NotImplementedError

    async def run():
        app = CayuApp(
            knowledge_store=UnsupportedKnowledge(access_scope=KnowledgeAccessScope.privileged())
        )
        await create_closure_session(app.session_store, "unsupported-knowledge")
        manifest = await app.inspect_session_closure("unsupported-knowledge")
        assert not manifest.complete
        record = next(item for item in manifest.records if item.store_id == "knowledge-store")
        assert record.disposition.value == "unsupported"
        report = await app.erase_session_closure("unsupported-knowledge")
        assert not report.complete
        assert await app.session_store.load("unsupported-knowledge") is not None

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_public_closure_exports_and_retains_knowledge_sources(tmp_path, request, backend):
    from tests.core.session_closure_conformance import create_closure_session

    from cayu import CayuApp
    from cayu.sessions.base import InMemorySessionStore
    from cayu.storage.memory import (
        KnowledgeIndexReadinessUpdate,
        KnowledgeIndexState,
        knowledge_chunk_embedding_identity,
    )

    if backend == "postgres":
        dsn = request.getfixturevalue("postgres_dsn")

    async def run():
        scope = KnowledgeAccessScope.privileged()
        if backend == "memory":
            sessions = InMemorySessionStore()
            knowledge = InMemoryKnowledgeStore(access_scope=scope)
        elif backend == "sqlite":
            from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore
            from cayu.storage.sqlite import SQLiteSessionStore

            sessions = SQLiteSessionStore(tmp_path / "sessions.sqlite")
            knowledge = SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite", access_scope=scope)
        else:
            from cayu.storage.migrations import SchemaMode
            from cayu.storage.postgres import PostgresKnowledgeStore, PostgresSessionStore

            sessions = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
            knowledge = PostgresKnowledgeStore(
                dsn, schema_mode=SchemaMode.CREATE, access_scope=scope
            )
        try:
            root = "knowledge-closure-public"
            await create_closure_session(sessions, root)
            await knowledge.create_entry(
                KnowledgeEntry(
                    id="public-shared-entry",
                    text="private shared text",
                    source_type="tool",
                    source_id=root,
                    source_uri=f"cayu://sessions/{root}",
                    source_hash="hash",
                ),
                evidence=[
                    KnowledgeEvidence(
                        id="public-shared-evidence",
                        entry_id="public-shared-entry",
                        source_type="session_event",
                        source_id=f"{root}-completed",
                        source_hash="hash",
                    )
                ],
            )
            app = CayuApp(session_store=sessions, knowledge_store=knowledge)
            chunk = (await knowledge.read_chunks("public-shared-entry"))[0]
            identity = knowledge_chunk_embedding_identity(
                chunk, embedding_model="test-embedding", dimensions=3
            )
            pending = await knowledge.publish_index_readiness(
                KnowledgeIndexReadinessUpdate(
                    identity=identity, state=KnowledgeIndexState.PENDING, attempt_id="attempt"
                ),
                expected_sequence=None,
                operation_id="closure-pending",
            )
            failed = await knowledge.publish_index_readiness(
                KnowledgeIndexReadinessUpdate(
                    identity=identity,
                    state=KnowledgeIndexState.FAILED,
                    attempt_id="attempt",
                    failure_code="private-index-failure",
                ),
                expected_sequence=pending.sequence,
                operation_id="closure-failed",
            )
            manifest = await app.inspect_session_closure(root)
            record = next(item for item in manifest.records if item.store_id == "knowledge-store")
            assert record.disposition.value == "shared" and record.count == 4
            exported = await app.export_session_closure(root)
            assert exported.manifest.complete
            references = exported.session_records["knowledge-store"]
            assert references["counts"] == {
                "knowledge_revisions": 1,
                "knowledge_evidence": 1,
                "knowledge_projections": 0,
                "knowledge_index_readiness": 2,
            }
            assert "private shared text" not in exported.to_bytes().decode()
            assert "private-index-failure" not in exported.to_bytes().decode()
            with pytest.raises(ValueError, match="bounds"):
                await knowledge.inspect_closure_sources(
                    KnowledgeClosureQuery(
                        sources=(("tool", root), ("session_event", f"{root}-completed")),
                        max_records=3,
                    )
                )
            report = await app.erase_session_closure(root)
            assert report.complete
            assert await sessions.load(root) is None
            assert (await knowledge.read_evidence("public-shared-entry")).total_evidence_known == 1
            assert await knowledge.load_index_readiness(identity) == failed
        finally:
            if backend != "memory":
                await knowledge.close()
                await sessions.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_closure_inventories_exact_sources_without_shared_content(tmp_path, request, backend):
    if backend == "postgres":
        dsn = request.getfixturevalue("postgres_dsn")

    def create():
        if backend == "memory":
            return InMemoryKnowledgeStore(access_scope=KnowledgeAccessScope.privileged())
        if backend == "sqlite":
            from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore

            return SQLiteKnowledgeStore(
                tmp_path / "knowledge.sqlite", access_scope=KnowledgeAccessScope.privileged()
            )
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresKnowledgeStore

        return PostgresKnowledgeStore(
            dsn, schema_mode=SchemaMode.CREATE, access_scope=KnowledgeAccessScope.privileged()
        )

    async def run():
        store = create()
        try:
            await check(store)
            expected = await store.inspect_closure_sources(
                KnowledgeClosureQuery(sources=(("session", "root"),))
            )
            expected_tool = await store.inspect_closure_sources(
                KnowledgeClosureQuery(sources=(("tool", "root"),))
            )
        finally:
            if backend != "memory":
                await store.close()
        if backend != "memory":
            reopened = create()
            try:
                assert (
                    await reopened.inspect_closure_sources(
                        KnowledgeClosureQuery(sources=(("session", "root"),))
                    )
                    == expected
                )
                assert (
                    await reopened.inspect_closure_sources(
                        KnowledgeClosureQuery(sources=(("tool", "root"),))
                    )
                    == expected_tool
                )
            finally:
                await reopened.close()

    async def check(store):
        from cayu.storage.knowledge_indexer import KnowledgeIndexer, KnowledgeIndexRequest

        indexed = KnowledgeIndexer().build(
            KnowledgeIndexRequest(
                text="private indexed knowledge text",
                entry_id="indexed-tool-entry",
                source_type="tool",
                source_id="root",
                source_uri="cayu://sessions/root",
                metadata={"private": "private indexed metadata"},
            )
        )
        await store.create_entry(indexed.entry, chunks=indexed.chunks)
        indexed_references = await store.inspect_closure_sources(
            KnowledgeClosureQuery(
                sources=(("tool", "root"),),
                source_uris=(("tool", "cayu://sessions/root"),),
                max_records=1,
            )
        )
        assert indexed_references["counts"] == {
            "knowledge_revisions": 1,
            "knowledge_evidence": 0,
            "knowledge_projections": 0,
            "knowledge_index_readiness": 0,
        }
        assert "private" not in json.dumps(indexed_references)
        assert "indexed-tool-entry" not in json.dumps(indexed_references)
        for index, source_type, source_id in (
            (1, "session", "root"),
            (2, "artifact", "root"),
            (3, "session", "other"),
            (4, "session", None),
        ):
            await store.create_entry(
                KnowledgeEntry(id=f"entry-{index}", text="private-shared-content"),
                evidence=[
                    KnowledgeEvidence(
                        id=f"evidence-{index}",
                        entry_id=f"entry-{index}",
                        source_type=source_type,
                        source_id=source_id,
                        source_uri=(
                            "cayu://sessions/root" if index != 3 else "cayu://sessions/other"
                        ),
                        source_hash="private-source-hash",
                        metadata={"private": "private-metadata"},
                    )
                ],
            )
        query = KnowledgeClosureQuery(sources=(("session", "root"),))
        result = await store.inspect_closure_sources(query)
        assert result["complete"] is True
        assert result["counts"] == {
            "knowledge_revisions": 0,
            "knowledge_evidence": 1,
            "knowledge_projections": 0,
            "knowledge_index_readiness": 0,
        }
        serialized = json.dumps(result)
        assert (
            "private" not in serialized
            and "entry-1" not in serialized
            and "evidence-1" not in serialized
        )
        assert await store.inspect_closure_sources(query) == result
        uri_query = KnowledgeClosureQuery(
            sources=(), source_uris=(("session", "cayu://sessions/root"),), max_records=2
        )
        by_uri = await store.inspect_closure_sources(uri_query)
        assert by_uri["counts"] == {
            "knowledge_revisions": 0,
            "knowledge_evidence": 2,
            "knowledge_projections": 0,
            "knowledge_index_readiness": 0,
        }
        assert (
            await store.inspect_closure_sources(
                KnowledgeClosureQuery(
                    sources=query.sources,
                    source_uris=tuple(("session", f"a://unused/{i:03}") for i in range(100))
                    + uri_query.source_uris,
                    max_records=2,
                )
            )
            == by_uri
        )
        with pytest.raises(ValueError, match="bounds"):
            await store.inspect_closure_sources(
                KnowledgeClosureQuery(
                    sources=query.sources, source_uris=uri_query.source_uris, max_records=1
                )
            )
        assert (await store.read_evidence("entry-1")).total_evidence_known == 1
        assert (await store.inspect_closure_sources(KnowledgeClosureQuery(sources=())))["counts"][
            "knowledge_evidence"
        ] == 0

    asyncio.run(run())


def test_knowledge_closure_aggregates_evidence_and_projection_bounds():
    collector = KnowledgeClosureInventory(KnowledgeClosureQuery(sources=(), max_records=2))
    collector.add("knowledge_evidence", {"id": "one"})
    collector.add("knowledge_projections", {"id": "two"})
    collector.add("knowledge_projections", {"id": "two"})
    assert collector.document()["counts"] == {
        "knowledge_revisions": 0,
        "knowledge_evidence": 1,
        "knowledge_projections": 1,
        "knowledge_index_readiness": 0,
    }
    with pytest.raises(ValueError, match="bounds"):
        collector.add("knowledge_projections", {"id": "three"})
    with pytest.raises(ValueError, match="already failed"):
        collector.document()


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_knowledge_closure_counts_across_source_batches(tmp_path, request, backend):
    if backend == "postgres":
        dsn = request.getfixturevalue("postgres_dsn")

    async def run():
        if backend == "memory":
            store = InMemoryKnowledgeStore(access_scope=KnowledgeAccessScope.privileged())
        elif backend == "sqlite":
            from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore

            store = SQLiteKnowledgeStore(
                tmp_path / "batched-knowledge.sqlite",
                access_scope=KnowledgeAccessScope.privileged(),
            )
        else:
            from cayu.storage.migrations import SchemaMode
            from cayu.storage.postgres import PostgresKnowledgeStore

            store = PostgresKnowledgeStore(
                dsn, schema_mode=SchemaMode.CREATE, access_scope=KnowledgeAccessScope.privileged()
            )
        sources = tuple(("session", f"batch-root-{index:03}") for index in range(101))
        try:
            await store.create_entry(
                KnowledgeEntry(id="batch-entry", text="retained shared content"),
                evidence=[
                    KnowledgeEvidence(
                        id=f"batch-evidence-{index}",
                        entry_id="batch-entry",
                        source_type=source_type,
                        source_id=source_id,
                        source_hash="hash",
                    )
                    for index, (source_type, source_id) in enumerate(sources)
                ],
            )
            exact = await store.inspect_closure_sources(
                KnowledgeClosureQuery(sources=sources, max_records=101)
            )
            assert exact["counts"]["knowledge_evidence"] == 101
            # Duplicate and reordered selectors must not renew the bound or
            # multiply records when selectors cross database-query batches.
            assert (
                await store.inspect_closure_sources(
                    KnowledgeClosureQuery(
                        sources=tuple(reversed(sources)) + sources, max_records=101
                    )
                )
                == exact
            )
            with pytest.raises(ValueError, match="bounds"):
                await store.inspect_closure_sources(
                    KnowledgeClosureQuery(sources=sources, max_records=100)
                )
            assert (
                await store.inspect_closure_sources(
                    KnowledgeClosureQuery(sources=sources, max_records=102)
                )
                == exact
            )
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_knowledge_closure_rejects_large_evidence_before_reconstruction(
    tmp_path, request, monkeypatch, backend
):
    if backend == "postgres":
        dsn = request.getfixturevalue("postgres_dsn")

    async def run():
        if backend == "sqlite":
            from cayu.storage import knowledge_sqlite as module

            store = module.SQLiteKnowledgeStore(
                tmp_path / "bounded-knowledge.sqlite",
                access_scope=KnowledgeAccessScope.privileged(),
            )
            converter = "_evidence_from_row"
        else:
            from cayu.storage import postgres as module
            from cayu.storage.migrations import SchemaMode

            store = module.PostgresKnowledgeStore(
                dsn,
                schema_mode=SchemaMode.CREATE,
                access_scope=KnowledgeAccessScope.privileged(),
            )
            converter = "_knowledge_evidence_from_row"
        try:
            await store.create_entry(
                KnowledgeEntry(id="bounded-evidence-entry", text="shared"),
                evidence=[
                    KnowledgeEvidence(
                        id="bounded-evidence",
                        entry_id="bounded-evidence-entry",
                        source_type="session",
                        source_id="bounded-root",
                        source_hash="hash",
                        metadata={"payload": "x" * 4096},
                    )
                ],
            )
            with monkeypatch.context() as patch:

                def forbidden_reconstruction(*args, **kwargs):
                    pytest.fail("Oversized evidence reached row reconstruction")

                patch.setattr(module, converter, forbidden_reconstruction)
                with pytest.raises(ValueError, match="bounds"):
                    await store.inspect_closure_sources(
                        KnowledgeClosureQuery(
                            sources=(("session", "bounded-root"),), max_bytes=1024
                        )
                    )
            # Inspection must not modify the shared source or its evidence.
            retained = await store.read_evidence("bounded-evidence-entry")
            assert retained.total_evidence_known == 1
        finally:
            await store.close()

    asyncio.run(run())


def test_knowledge_closure_rejects_conflicting_duplicate_evidence():
    collector = KnowledgeClosureInventory(KnowledgeClosureQuery(sources=()))
    collector.add("knowledge_evidence", {"id": "same", "source_hash": "first"})
    with pytest.raises(ValueError, match="conflicting content"):
        collector.add("knowledge_evidence", {"id": "same", "source_hash": "second"})
    with pytest.raises(ValueError, match="already failed"):
        collector.document()


def test_memory_knowledge_closure_refuses_oversized_inventory():
    async def run():
        store = InMemoryKnowledgeStore(access_scope=KnowledgeAccessScope.privileged())
        entry = KnowledgeEntry(id="entry", text="shared")
        await store.create_entry(
            entry,
            evidence=[
                KnowledgeEvidence(
                    id=f"evidence-{index}",
                    entry_id=entry.id,
                    source_type="session",
                    source_id="root",
                    source_hash="hash",
                )
                for index in range(2)
            ],
        )
        with pytest.raises(ValueError, match="bounds"):
            await store.inspect_closure_sources(
                KnowledgeClosureQuery(sources=(("session", "root"),), max_records=1)
            )
        assert len(store._evidence[(entry.id, 1)]) == 2

    asyncio.run(run())


@pytest.mark.parametrize("field", ["sources", "source_uris"])
def test_knowledge_closure_revalidates_mutated_selectors_without_diagnostics(
    capsys, caplog, recwarn, field
):
    class Private:
        def __repr__(self):
            return "private-closure-source-canary"

    async def run():
        query = KnowledgeClosureQuery(sources=())
        object.__setattr__(query, field, (("session", Private()),))
        with pytest.raises(ValueError) as error:
            await InMemoryKnowledgeStore().inspect_closure_sources(query)
        assert "private-closure-source-canary" not in str(error.value)

    asyncio.run(run())
    captured = capsys.readouterr()
    assert "private-closure-source-canary" not in captured.out + captured.err + caplog.text
    assert all("private-closure-source-canary" not in str(item.message) for item in recwarn)


def test_knowledge_closure_includes_real_embedding_projection_once():
    async def run():
        provider = KeywordEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        await store.create_entry(
            KnowledgeEntry(id="shared", text="GitHub proxy."),
            evidence=[
                KnowledgeEvidence(
                    id="evidence",
                    entry_id="shared",
                    source_type="session",
                    source_id="root",
                    source_hash="source-hash",
                )
            ],
        )
        await store.process_embedding_changes("closure-test", "worker")
        assert len(store._chunk_embeddings) == 1
        assert len(store._chunk_embedding_history) == 1
        calls = list(provider.calls)
        query = KnowledgeClosureQuery(sources=(("session", "root"),), max_records=4)
        result = await store.inspect_closure_sources(query)
        assert result["counts"] == {
            "knowledge_revisions": 0,
            "knowledge_evidence": 1,
            "knowledge_projections": 1,
            "knowledge_index_readiness": 2,
        }
        assert "GitHub" not in json.dumps(result)
        assert "vector" not in json.dumps(result)
        assert provider.calls == calls
        with pytest.raises(ValueError, match="bounds"):
            await store.inspect_closure_sources(
                KnowledgeClosureQuery(sources=query.sources, max_records=3)
            )
        assert len(store._chunk_embeddings) == 1 and len(store._entries) == 1

    asyncio.run(run())


def test_postgres_plain_handle_inventories_persisted_embedding_projection(postgres_dsn):
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresEmbeddingKnowledgeStore, PostgresKnowledgeStore

    async def run():
        provider = KeywordEmbeddingProvider()
        store = PostgresEmbeddingKnowledgeStore(
            postgres_dsn,
            schema_mode=SchemaMode.CREATE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        query = KnowledgeClosureQuery(sources=(("session", "projection-root"),), max_records=4)
        try:
            await store.create_entry(
                KnowledgeEntry(id="projected-shared", text="GitHub proxy."),
                evidence=[
                    KnowledgeEvidence(
                        id="projected-evidence",
                        entry_id="projected-shared",
                        source_type="session",
                        source_id="projection-root",
                        source_hash="source-hash",
                    )
                ],
            )
            await store.process_embedding_changes("closure-projection", "worker")
            expected = await store.inspect_closure_sources(query)
            assert expected["counts"] == {
                "knowledge_revisions": 0,
                "knowledge_evidence": 1,
                "knowledge_projections": 1,
                "knowledge_index_readiness": 2,
            }
        finally:
            await store.close()
        plain = PostgresKnowledgeStore(
            postgres_dsn,
            schema_mode=SchemaMode.VALIDATE,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        try:
            assert await plain.inspect_closure_sources(query) == expected
            with pytest.raises(ValueError, match="bounds"):
                await plain.inspect_closure_sources(
                    KnowledgeClosureQuery(sources=query.sources, max_records=3)
                )
            assert (await plain.read_evidence("projected-shared")).total_evidence_known == 1
        finally:
            await plain.close()

    asyncio.run(run())
