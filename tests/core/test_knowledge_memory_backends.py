"""Memory backend boundaries preserve public identity and shared input rules."""

import asyncio
import importlib
import os
import pickle
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

import cayu
from cayu.knowledge import changes as change_rules
from cayu.knowledge import maintenance_contracts
from cayu.knowledge import records as identity_rules


@pytest.mark.parametrize("module", ["cayu", "cayu.storage", "cayu.storage.memory"])
@pytest.mark.parametrize("name", ["InMemoryKnowledgeStore", "InMemoryEmbeddingKnowledgeStore"])
def test_memory_backend_public_imports_and_saved_class_globals_share_identity(module, name):
    from cayu.storage import memory

    cls = getattr(importlib.import_module(module), name)
    assert cls is getattr(memory, name)
    assert pickle.loads(f"ccayu.storage.memory\n{name}\n.".encode()) is cls
    assert pickle.loads(pickle.dumps(cls)) is cls


def test_memory_embedding_backend_preserves_inheritance():
    from cayu import InMemoryEmbeddingKnowledgeStore, InMemoryKnowledgeStore, KnowledgeStore

    assert InMemoryEmbeddingKnowledgeStore.__bases__ == (InMemoryKnowledgeStore,)
    assert InMemoryKnowledgeStore.__bases__ == (KnowledgeStore,)
    assert InMemoryEmbeddingKnowledgeStore.get_entry is InMemoryKnowledgeStore.get_entry


@pytest.mark.parametrize("value", ["route", "a" * 256, "猫" * 85, "café"])
def test_memory_shared_operation_identity_preserves_valid_text(value):
    assert identity_rules._knowledge_semantic_watch_identity(value, "operation_id") == value


@pytest.mark.parametrize("value", ["", " ", " route", "route ", "a" * 257, "猫" * 86])
def test_memory_shared_operation_identity_preserves_validation(value):
    with pytest.raises(ValueError, match="operation_id"):
        identity_rules._knowledge_semantic_watch_identity(value, "operation_id")


@pytest.mark.parametrize("offset", [-300, 0, 330])
def test_memory_shared_change_time_normalizes_offsets(offset):
    value = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(minutes=offset)))
    result = change_rules._knowledge_change_now(value)
    assert result == value
    assert result.tzinfo is UTC


def test_memory_shared_change_time_rejects_naive_values():
    with pytest.raises(ValueError, match="`now` must be timezone-aware"):
        change_rules._knowledge_change_now(datetime(2026, 1, 1))


def test_memory_shared_change_time_reads_clock_only_for_absent_values(monkeypatch):
    calls = []
    now = datetime(2026, 1, 1, tzinfo=UTC)

    class Clock:
        @staticmethod
        def now(tz):
            calls.append(tz)
            return now

    monkeypatch.setattr(change_rules, "datetime", Clock)
    assert change_rules._knowledge_change_now(now) is now
    assert calls == []
    assert change_rules._knowledge_change_now(None) is now
    assert calls == [UTC]


def test_memory_governance_metadata_key_preserves_public_value():
    import cayu
    from cayu.knowledge import maintenance_governance
    from cayu.storage import memory

    key = maintenance_contracts.KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY
    assert key == "cayu_knowledge_maintenance_governance"
    assert key is cayu.KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY
    assert key is memory.KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY
    assert key is maintenance_governance.KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY


@pytest.mark.parametrize(
    "statement,allowed",
    [
        ("from cayu import InMemoryKnowledgeStore; InMemoryKnowledgeStore()", ["knowledge_memory"]),
        (
            "from cayu.storage import InMemoryKnowledgeStore; InMemoryKnowledgeStore()",
            ["knowledge_memory"],
        ),
        (
            "from cayu.storage.knowledge_memory import InMemoryKnowledgeStore; InMemoryKnowledgeStore()",
            ["knowledge_memory"],
        ),
        (
            "from cayu import InMemoryEmbeddingKnowledgeStore",
            ["knowledge_memory", "knowledge_embedding_memory"],
        ),
        ("from cayu import KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY", []),
        ("import cayu.knowledge.maintenance_governance", []),
        ("import cayu.storage.knowledge_sqlite", []),
        ("import cayu.storage.postgres", []),
    ],
    ids=[
        "root",
        "storage",
        "ordinary",
        "embedding",
        "metadata",
        "governance",
        "sqlite",
        "postgres",
    ],
)
def test_memory_backend_consumers_load_only_required_implementations(statement, allowed):
    script = f"""
import sys
{statement}
owners = {{"cayu.storage.memory", "cayu.storage.knowledge_memory", "cayu.storage.knowledge_embedding_memory"}}
assert owners.intersection(sys.modules) == {{"cayu.storage." + name for name in {allowed!r}}}
"""
    _check_import_process(script)


@pytest.mark.parametrize("first", ["memory", "knowledge_memory", "knowledge_embedding_memory"])
def test_memory_backend_import_order_preserves_canonical_identity(first):
    _check_import_process(f"""
import importlib
importlib.import_module("cayu.storage.{first}")
from cayu.storage import memory, knowledge_memory, knowledge_embedding_memory
from cayu import InMemoryKnowledgeStore, InMemoryEmbeddingKnowledgeStore
assert InMemoryKnowledgeStore is memory.InMemoryKnowledgeStore is knowledge_memory.InMemoryKnowledgeStore
assert InMemoryEmbeddingKnowledgeStore is memory.InMemoryEmbeddingKnowledgeStore is knowledge_embedding_memory.InMemoryEmbeddingKnowledgeStore
assert InMemoryEmbeddingKnowledgeStore.__bases__ == (InMemoryKnowledgeStore,)
""")


def _check_import_process(script):
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_memory_backend_declarations_have_one_owner():
    from cayu.knowledge import changes, maintenance_contracts, records
    from cayu.storage import knowledge_embedding_memory, knowledge_memory, memory, postgres

    assert knowledge_memory.InMemoryKnowledgeStore.__module__ == knowledge_memory.__name__
    assert (
        knowledge_embedding_memory.InMemoryEmbeddingKnowledgeStore.__module__
        == knowledge_embedding_memory.__name__
    )
    for owner, names in [
        (knowledge_memory, ["_knowledge_facets", "_next_updated_at"]),
        (
            knowledge_embedding_memory,
            ["_StoredChunkEmbedding", "_cosine_similarity", "_normalize_cosine_similarity"],
        ),
        (changes, ["_knowledge_change_now"]),
        (records, ["_knowledge_semantic_watch_identity"]),
    ]:
        for name in names:
            assert getattr(owner, name).__module__ == owner.__name__
            assert not hasattr(memory, name)
    assert postgres._knowledge_change_now is changes._knowledge_change_now
    assert postgres._knowledge_semantic_watch_identity is records._knowledge_semantic_watch_identity
    assert (
        memory.KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY
        is maintenance_contracts.KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY
    )


@pytest.mark.parametrize("embedding", [False, True], ids=["ordinary", "embedding"])
@pytest.mark.parametrize("custom", [False, True], ids=["builtin", "subclass"])
def test_memory_backend_support_schema_readiness_preserves_exact_builtin_detection(
    embedding, custom
):
    from cayu import InMemoryEmbeddingKnowledgeStore, InMemoryKnowledgeStore
    from cayu.embeddings import TextEmbeddingProvider
    from cayu.support_bundles import _store_schema_readiness

    class UnusedProvider(TextEmbeddingProvider):
        name = "unused"

        async def embed_texts(self, request):
            raise AssertionError("Schema inspection must not call the embedding provider.")

    cls = InMemoryEmbeddingKnowledgeStore if embedding else InMemoryKnowledgeStore
    if custom:
        cls = type("CustomStore", (cls,), {})
    options = (
        {
            "embedding_provider": UnusedProvider(),
            "embedding_model": "test",
            "embedding_dimensions": 2,
        }
        if embedding
        else {}
    )
    store = cls(**options)
    assert asyncio.run(_store_schema_readiness(store)) == (
        "unavailable" if custom else "not_applicable"
    )
