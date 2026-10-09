"""Custom knowledge stores compose independently and retain access boundaries."""

import ast
import asyncio
import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import pytest

import cayu
from cayu.knowledge.base import KnowledgeStore
from cayu.knowledge.records import KnowledgeEntry
from cayu.knowledge.scopes import (
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    copy_knowledge_access_scope,
)
from cayu.knowledge.search import KnowledgeSearchMode


class _ReadOnlyStore(KnowledgeStore):
    def __init__(self, scope=None):
        self._default_access_scope = None if scope is None else copy_knowledge_access_scope(scope)

    async def get_entry(self, entry_id, *, access_scope=None, **kwargs):
        scope = self._operation_access_scope(access_scope)
        if "example" not in scope.allowed_namespaces:
            return None
        return KnowledgeEntry(id=entry_id, namespace="example", text="Custom storage")

    async def _unsupported(self, *args, **kwargs):
        raise NotImplementedError("Read-only test store.")

    create_entry = _unsupported
    append_entry_revision = _unsupported
    transition_entry_status = _unsupported
    delete_entry = _unsupported
    read_evidence = _unsupported
    read_chunks = _unsupported
    search = _unsupported
    list_entries = _unsupported
    read_changes = _unsupported
    initialize_change_consumer = _unsupported
    claim_change = _unsupported
    acknowledge_change = _unsupported
    release_change = _unsupported
    load_change_consumer_state = _unsupported


@pytest.mark.parametrize("module_name", ("cayu", "cayu.storage", "cayu.knowledge.base"))
def test_custom_knowledge_store_composes_without_builtin_backends(module_name):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import importlib
import sys

api = importlib.import_module(sys.argv[1])
interface = api.KnowledgeStore
from tests.core.test_knowledge_store_interface import _ReadOnlyStore
from cayu.knowledge.scopes import KnowledgeAccessScope

store = _ReadOnlyStore(KnowledgeAccessScope.for_namespace("example"))
assert isinstance(store, interface)
assert asyncio.run(store.get_entry("entry")).text == "Custom storage"
assert not {
    "cayu.storage.memory", "cayu.storage.knowledge_sqlite", "cayu.storage.postgres",
    "cayu.storage.knowledge_indexer", "cayu.knowledge.maintenance_persistence",
    "cayu.knowledge.maintenance_governance", "cayu.knowledge.semantic_watch",
}.intersection(sys.modules)
""",
            module_name,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_knowledge_store_preserves_import_identity_subclasses_and_annotations():
    import cayu.storage as storage
    import cayu.storage.memory as legacy
    from cayu.storage._knowledge_closure import KnowledgeClosureQuery
    from cayu.storage.knowledge_embedding_postgres import PostgresEmbeddingKnowledgeStore
    from cayu.storage.knowledge_postgres import PostgresKnowledgeStore
    from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore

    assert legacy.KnowledgeStore is KnowledgeStore
    assert pickle.loads(b"ccayu.storage.memory\nKnowledgeStore\n.") is KnowledgeStore
    assert pickle.loads(pickle.dumps(KnowledgeStore)) is KnowledgeStore
    for backend in (
        legacy.InMemoryKnowledgeStore,
        legacy.InMemoryEmbeddingKnowledgeStore,
        SQLiteKnowledgeStore,
        PostgresKnowledgeStore,
        PostgresEmbeddingKnowledgeStore,
    ):
        assert issubclass(backend, KnowledgeStore)
        assert backend.__abstractmethods__ == frozenset()
    with pytest.raises(TypeError, match="abstract"):
        KnowledgeStore()
    for package in (cayu, storage):
        assert package.KnowledgeStore is KnowledgeStore
        exports = importlib.import_module(package.__name__ + "._exports").EXPORTS
        assert exports["KnowledgeStore"] == ("cayu.knowledge.base", "KnowledgeStore")
        stub = ast.parse(Path(package.__file__).with_suffix(".pyi").read_text())
        owners = {
            node.module
            for node in stub.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if (alias.asname or alias.name) == "KnowledgeStore"
        }
        assert owners == {"cayu.knowledge.base"}
    assert get_type_hints(KnowledgeStore.inspect_closure_sources)["query"] is KnowledgeClosureQuery
    assert get_type_hints(KnowledgeStore.get_entry)["return"] == KnowledgeEntry | None


def test_knowledge_store_scope_defaults_copy_and_reject_overrides():
    unbound = _ReadOnlyStore()
    assert unbound.bound_access_scope() is None
    with pytest.raises(TypeError, match="requires `access_scope`"):
        unbound._operation_access_scope(None)
    scope = KnowledgeAccessScope.for_namespace("example", required_labels={"team": "support"})
    explicit = unbound._operation_access_scope(scope)
    assert explicit == scope and explicit is not scope
    explicit.required_labels["team"] = "changed"
    assert scope.required_labels == {"team": "support"}

    bound = _ReadOnlyStore(scope)
    for copy in (bound.bound_access_scope(), bound._operation_access_scope(None)):
        assert copy == scope and copy is not scope
        copy.required_labels["team"] = "changed"
        copy.allowed_namespaces.append("foreign")
    assert bound.bound_access_scope() == scope
    with pytest.raises(KnowledgeAccessDenied, match="access_scope_override"):
        bound._operation_access_scope(KnowledgeAccessScope.privileged())
    assert bound._operation_access_scope(scope) == scope


def test_knowledge_store_intersects_ambient_constraints_and_rejects_overflow():
    from cayu._resource_access_errors import ResourceAccessDenied
    from cayu.knowledge.access import _constraints
    from cayu.resource_access import encode_scope
    from cayu.sessions.access import SessionAccessRule, SessionAccessScope, SessionAccessSelector

    constraints = tuple(
        encode_scope(
            SessionAccessScope(
                read=(
                    SessionAccessRule(selectors=(SessionAccessSelector(key=key, values=("a",)),)),
                )
            )
        )
        for key in ("team", "region", "account", "project", "department")
    )
    scope = KnowledgeAccessScope.for_namespace("example").model_copy(
        update={"resource_constraints": constraints[:1]}
    )
    store = _ReadOnlyStore(scope)
    token = _constraints.set(constraints[:4])
    try:
        effective = store._operation_access_scope(None)
        assert effective.resource_constraints == constraints[:4]
        assert store.bound_access_scope().resource_constraints == constraints[:1]
        _constraints.set(constraints[1:])
        with pytest.raises(ResourceAccessDenied):
            store._operation_access_scope(None)
    finally:
        _constraints.reset(token)
    assert store._operation_access_scope(None) == scope


def test_knowledge_store_requires_intersection_inside_resource_execution():
    from cayu._resource_access_binding import ResourceExecutionBinding
    from cayu._resource_access_errors import ResourceAccessDenied
    from cayu.resource_access import ResourceAccessPolicy, encode_scope, execution_access
    from cayu.sessions.access import SessionAccessRule, SessionAccessScope

    all_rule = SessionAccessRule(allow_all=True)
    access = SessionAccessScope(read=(all_rule,), execute=(all_rule,))

    class Policy(ResourceAccessPolicy):
        authority = "custom-knowledge-store-test"

        async def resolve(self, subject):
            return access

    policy = Policy()
    binding = ResourceExecutionBinding(
        authority=policy.authority, subject="reader", admitted_json=encode_scope(access)
    )
    store = _ReadOnlyStore(KnowledgeAccessScope.for_namespace("example"))

    async def run():
        async with execution_access(binding, policy, {}):
            with pytest.raises(ResourceAccessDenied):
                await store.get_entry("entry")
        assert (await store.get_entry("entry")).id == "entry"

    asyncio.run(run())


def test_custom_knowledge_store_can_decline_optional_operations():
    store = _ReadOnlyStore(KnowledgeAccessScope.for_namespace("example"))
    assert store.supported_search_modes() == (KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD)

    async def run():
        with pytest.raises(NotImplementedError, match="owned revision publication"):
            await store.publish_entry_revision(
                KnowledgeEntry(id="entry", text="content"), [], operation_id="publish"
            )
        with pytest.raises(NotImplementedError, match="embedding backfill"):
            await store.backfill_embeddings()
        with pytest.raises(NotImplementedError, match="prune_expired"):
            await store.prune_expired()

    asyncio.run(run())
