"""Independent indexing contracts and compatibility with existing package imports."""

import ast
import importlib
import inspect
import os
import pickle
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import get_type_hints

import pytest

import cayu

_CLASSES = (
    "KnowledgeEmbeddingIdentity",
    "KnowledgeEmbeddingProjection",
    "KnowledgeEmbeddingProjectionWriteResult",
    "KnowledgeEmbeddingBackfillResult",
    "KnowledgeIndexState",
    "KnowledgeIndexReadinessConflict",
    "KnowledgeEmbeddingProjectionConflict",
    "KnowledgeIndexReadinessUpdate",
    "KnowledgeIndexReadiness",
    "KnowledgeIndexReadinessBatch",
    "KnowledgeIndexCoverage",
    "KnowledgeEmbeddingWorkerResult",
)
_HELPERS = (
    "copy_knowledge_embedding_identity",
    "copy_knowledge_embedding_projection",
    "_copy_knowledge_embedding_projections",
    "copy_knowledge_index_readiness_update",
    "copy_knowledge_index_readiness",
    "copy_knowledge_index_coverage",
    "_knowledge_embedding_identity_sha256",
    "_knowledge_embedding_vector_sha256",
    "_knowledge_index_readiness_update_sha256",
    "_validate_knowledge_index_readiness_transition",
    "_bounded_knowledge_index_identity",
    "_validate_knowledge_index_sequence",
    "_validate_knowledge_index_readiness_limit",
    "_validate_knowledge_embedding_work_record_limit",
    "_bounded_knowledge_embedding_backfill_cursor",
    "knowledge_chunk_embedding_identity",
    "_knowledge_chunk_content_hash",
)
_CONSTANTS = (
    "DEFAULT_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT",
    "MAX_KNOWLEDGE_EMBEDDING_DIMENSIONS",
    "MAX_KNOWLEDGE_INDEX_READINESS_LIMIT",
    "MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT",
    "_MAX_KNOWLEDGE_EMBEDDING_BACKFILL_CURSOR_BYTES",
    "KNOWLEDGE_CHUNK_TEXT_PROJECTION",
    "KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION",
    "KNOWLEDGE_CHUNK_TEXT_GENERATOR",
    "KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION",
    "KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION",
)

_PUBLIC_NAMES = (
    *_CLASSES,
    *(name for name in _CONSTANTS if not name.startswith("_")),
    "knowledge_chunk_embedding_identity",
)


@pytest.mark.parametrize("module_name", ("cayu", "cayu.storage", "cayu.knowledge.indexing"))
def test_indexing_contracts_compose_without_storage_implementations(module_name):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys
from datetime import UTC, datetime

api = importlib.import_module(sys.argv[1])
from cayu.knowledge.records import KnowledgeChunk
from cayu.knowledge.indexing import (
    _knowledge_embedding_identity_sha256,
    _knowledge_embedding_vector_sha256,
    _knowledge_index_readiness_update_sha256,
    _validate_knowledge_index_readiness_transition,
    copy_knowledge_embedding_projection,
)

chunk = KnowledgeChunk(id="chunk", entry_id="entry", entry_revision=2, chunk_index=0, text="content")
identity = api.knowledge_chunk_embedding_identity(chunk, embedding_model="embedding-v1", dimensions=3)
assert type(identity) is api.KnowledgeEmbeddingIdentity
assert identity.entry_revision == 2 and identity.chunk_id == chunk.id
assert identity.projection_content_hash == "sha256:ed7002b439e9ac845f22357d822bac1444730fbdb6016d3ec9432297b9ec9f73"
assert identity.projection_type == api.KNOWLEDGE_CHUNK_TEXT_PROJECTION
assert _knowledge_embedding_identity_sha256(identity) != _knowledge_embedding_identity_sha256(
    identity.model_copy(update={"generator_version": "next"})
)
now = datetime(2026, 1, 1, tzinfo=UTC)
pending = api.KnowledgeIndexReadinessUpdate(identity=identity, state="pending", attempt_id="attempt")
_validate_knowledge_index_readiness_transition(None, pending, expected_sequence=None)
record = api.KnowledgeIndexReadiness(
    identity=identity, state="pending", attempt_id="attempt", sequence=1,
    operation_id="operation", published_at=now,
)
ready = api.KnowledgeIndexReadinessUpdate(identity=identity, state="ready", attempt_id="attempt")
_validate_knowledge_index_readiness_transition(record, ready, expected_sequence=1)
assert _knowledge_index_readiness_update_sha256(pending) != _knowledge_index_readiness_update_sha256(ready)
try:
    _validate_knowledge_index_readiness_transition(record, ready, expected_sequence=0)
except api.KnowledgeIndexReadinessConflict as error:
    assert error.reason == "stale_sequence"
else:
    raise AssertionError("Accepted stale readiness publication")
vector = [1.0, -0.0, 0.0]
projection = api.KnowledgeEmbeddingProjection(identity=identity, readiness_sequence=1, attempt_id="attempt", vector=vector)
vector[0] = 2.0
assert projection.vector == [1.0, 0.0, 0.0]
copied = copy_knowledge_embedding_projection(projection)
assert copied == projection and copied is not projection
assert copied.identity is not projection.identity and copied.vector is not projection.vector
assert _knowledge_embedding_vector_sha256(copied.vector) == _knowledge_embedding_vector_sha256([1.0, 0.0, 0.0])
page = api.KnowledgeIndexReadinessBatch(readiness=[record], next_after_sequence=1, high_water_sequence=2, truncated=True, limit=1)
assert page.readiness[0] == record and page.readiness[0] is not record
assert page.readiness[0].identity is not record.identity
accepted = api.KnowledgeEmbeddingProjectionWriteResult(submitted_records=1, stored_identities=[identity])
assert accepted.stored_identities[0] is not identity
custom = api.KnowledgeEmbeddingIdentity(**(identity.model_dump() | {
    "chunk_id": None, "projection_type": "customer-document", "generator": "customer-generator",
}))
assert custom.chunk_id is None and custom.generator == "customer-generator"
assert not {"cayu.storage.memory", "cayu.storage.knowledge_sqlite", "cayu.storage.postgres"}.intersection(sys.modules)
""",
            module_name,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_indexing_contracts_preserve_aliases_stubs_and_annotations():
    import cayu.storage as storage
    import cayu.storage.memory as legacy
    from cayu.knowledge import indexing

    for name in (*_CLASSES, *_HELPERS, *_CONSTANTS):
        canonical = getattr(indexing, name)
        assert getattr(legacy, name) is canonical
        if name in _CONSTANTS:
            continue
        assert canonical.__module__ == indexing.__name__
        assert pickle.loads(f"ccayu.storage.memory\n{name}\n.".encode()) is canonical
        get_type_hints(canonical)
        if inspect.isclass(canonical):
            for method in vars(canonical).values():
                if isinstance(method, classmethod | staticmethod):
                    method = method.__func__
                elif isinstance(method, property):
                    method = method.fget
                if inspect.isfunction(method):
                    get_type_hints(method)
    for package in (cayu, storage):
        manifest = importlib.import_module(package.__name__ + "._exports").EXPORTS
        stub = ast.parse(Path(package.__file__).with_suffix(".pyi").read_text())
        imports = {
            alias.asname or alias.name: node.module
            for node in stub.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        for name in _PUBLIC_NAMES:
            assert manifest[name] == (indexing.__name__, name)
            assert imports[name] == indexing.__name__
            assert getattr(package, name) is getattr(indexing, name)


def test_indexing_values_round_trip_through_canonical_pickle_paths():
    from cayu.knowledge import indexing as api
    from cayu.knowledge.records import KnowledgeChunk

    chunk = KnowledgeChunk(id="chunk", entry_id="entry", text="content", chunk_index=0)
    identity = api.knowledge_chunk_embedding_identity(
        chunk, embedding_model="embedding-v1", dimensions=3
    )
    update = api.KnowledgeIndexReadinessUpdate(
        identity=identity, state="pending", attempt_id="attempt"
    )
    record = api.KnowledgeIndexReadiness(
        identity=identity,
        state="pending",
        attempt_id="attempt",
        sequence=1,
        operation_id="operation",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    projection = api.KnowledgeEmbeddingProjection(
        identity=identity, readiness_sequence=1, attempt_id="attempt", vector=[1.0, 0.0, 0.0]
    )
    batch = api.KnowledgeIndexReadinessBatch(
        readiness=[record], next_after_sequence=1, high_water_sequence=1, limit=1
    )
    coverage = api.KnowledgeIndexCoverage(
        **identity.model_dump(
            include={
                "projection_type",
                "embedding_model",
                "dimensions",
                "preprocessing_version",
                "generator",
                "generator_version",
                "index_representation_version",
            }
        ),
        eligible_records=1,
        ready_records=0,
        pending_records=1,
        failed_records=0,
        high_water_sequence=1,
        complete=False,
    )
    accepted = api.KnowledgeEmbeddingProjectionWriteResult(
        submitted_records=1, stored_identities=[identity]
    )
    backfill = api.KnowledgeEmbeddingBackfillResult(
        scanned_records=1,
        indexed_records=1,
        failed_records=0,
        skipped_records=0,
        limit=1,
        refresh_existing=False,
        next_cursor="cursor",
    )
    worker = api.KnowledgeEmbeddingWorkerResult(
        consumer_id="consumer",
        worker_id="worker",
        claimed_changes=1,
        acknowledged_changes=1,
        indexed_records=1,
        failed_records=0,
        removed_records=0,
        limit=1,
        processed_records=1,
        record_limit=1,
    )
    for value in (
        identity,
        update,
        record,
        projection,
        batch,
        coverage,
        accepted,
        backfill,
        worker,
        api.KnowledgeIndexState.PENDING,
    ):
        encoded = pickle.dumps(value)
        assert api.__name__.encode() in encoded
        restored = pickle.loads(encoded)
        assert type(restored) is type(value)
        assert restored == value
