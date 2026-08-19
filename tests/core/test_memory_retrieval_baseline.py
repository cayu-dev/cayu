from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from cayu.evals.memory_baseline import (
    MemoryRetrievalBaselineResult,
    load_memory_retrieval_corpus,
    run_memory_retrieval_baseline,
)
from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    SQLiteKnowledgeStore,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_PATH = _REPOSITORY_ROOT / "benchmarks/memory/corpus-v1.json"
_RESULTS_PATH = _REPOSITORY_ROOT / "benchmarks/memory/baseline-results-v1.json"


def test_public_memory_retrieval_baseline_is_reproducible_across_builtin_backends(
    tmp_path: Path,
) -> None:
    async def run() -> list[MemoryRetrievalBaselineResult]:
        corpus = load_memory_retrieval_corpus(_CORPUS_PATH)
        memory = await run_memory_retrieval_baseline(
            corpus,
            InMemoryKnowledgeStore(),
            backend="memory",
        )
        sqlite_store = SQLiteKnowledgeStore(tmp_path / "baseline.sqlite")
        try:
            sqlite = await run_memory_retrieval_baseline(
                corpus,
                sqlite_store,
                backend="sqlite",
            )
        finally:
            await sqlite_store.close()
        return [memory, sqlite]

    actual = asyncio.run(run())
    checked_payload = json.loads(_RESULTS_PATH.read_text(encoding="utf-8"))
    checked = [
        MemoryRetrievalBaselineResult.model_validate(result)
        for result in checked_payload["results"]
    ]

    assert checked_payload["corpus_revision"] == "public-hermetic-v1"
    assert [result.backend for result in actual] == ["memory", "sqlite"]
    for result, frozen in zip(actual, checked, strict=True):
        assert _without_latency(result) == _without_latency(frozen)
        assert result.metrics.latency_p50_ms >= 0.0
        assert result.metrics.latency_p95_ms >= result.metrics.latency_p50_ms
        assert result.metrics.authorization_leak_rate == 0.0
        assert result.metrics.stale_result_rate == 0.0
        assert result.metrics.false_injection_rate == 0.0
        assert result.metrics.recall_at_k == 1.0
        assert result.metrics.citation_correctness == 1.0
        assert [slice_.language for slice_ in result.language_slices] == ["en", "es", "fr"]


def test_memory_retrieval_baseline_requires_an_empty_store() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore()
        await store.create_entry(
            KnowledgeEntry(id="existing", text="existing knowledge"),
            access_scope=KnowledgeAccessScope.privileged(),
        )
        with pytest.raises(ValueError, match="requires an empty knowledge store"):
            await run_memory_retrieval_baseline(
                load_memory_retrieval_corpus(_CORPUS_PATH),
                store,
                backend="memory",
            )

    asyncio.run(run())


def test_memory_retrieval_baseline_preserves_required_runner_configuration() -> None:
    async def run() -> None:
        corpus = load_memory_retrieval_corpus(_CORPUS_PATH)
        result = await run_memory_retrieval_baseline(
            corpus,
            InMemoryKnowledgeStore(),
            backend="memory",
            configuration={"experiment": "candidate-v2"},
        )
        assert result.configuration == {
            "cutoffs": [5],
            "experiment": "candidate-v2",
            "source_set": ["knowledge"],
            "token_estimator": "ceil(utf8_json_bytes/4)",
        }

        with pytest.raises(ValueError, match="runner-owned field 'source_set'"):
            await run_memory_retrieval_baseline(
                corpus,
                InMemoryKnowledgeStore(),
                backend="memory",
                configuration={"source_set": ["transcript"]},
            )

    asyncio.run(run())


def test_external_private_corpus_uses_the_same_bounded_import_contract(
    tmp_path: Path,
) -> None:
    payload = json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))
    payload["origin"] = "external_private"
    private_path = tmp_path / "private-corpus.json"
    private_path.write_text(json.dumps(payload), encoding="utf-8")

    corpus = load_memory_retrieval_corpus(private_path)

    assert corpus.origin == "external_private"
    assert corpus.cases[0].trajectory_id == "public-ops-trajectory"
    assert corpus.cases[0].turn_index == 120


def _without_latency(result: MemoryRetrievalBaselineResult) -> dict:
    payload = result.model_dump(mode="json")
    payload["metrics"].pop("latency_p50_ms")
    payload["metrics"].pop("latency_p95_ms")
    for case in payload["cases"]:
        case.pop("latency_ms")
    return payload
