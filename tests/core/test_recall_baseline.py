from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from cayu.evals.recall_baseline import (
    RecallBaselineResult,
    load_recall_baseline_corpus,
    run_recall_baseline,
)
from cayu.runtime import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.storage import (
    InMemoryKnowledgeStore,
    SQLiteKnowledgeStore,
    SQLiteSessionStore,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_PATH = _REPOSITORY_ROOT / "benchmarks/memory/recall-corpus-v1.json"
_RESULTS_PATH = _REPOSITORY_ROOT / "benchmarks/memory/recall-baseline-results-v1.json"


def test_public_recall_baseline_is_reproducible_across_builtin_backends(
    tmp_path: Path,
) -> None:
    async def run() -> list[RecallBaselineResult]:
        corpus = load_recall_baseline_corpus(_CORPUS_PATH)
        memory = await run_recall_baseline(
            corpus,
            InMemoryKnowledgeStore(),
            InMemorySessionStore(),
            backend="memory",
        )
        database_path = tmp_path / "recall-baseline.sqlite"
        sqlite_knowledge = SQLiteKnowledgeStore(database_path)
        sqlite_sessions = SQLiteSessionStore(database_path)
        try:
            sqlite = await run_recall_baseline(
                corpus,
                sqlite_knowledge,
                sqlite_sessions,
                backend="sqlite",
            )
        finally:
            await sqlite_sessions.close()
            await sqlite_knowledge.close()
        return [memory, sqlite]

    actual = asyncio.run(run())
    checked_payload = json.loads(_RESULTS_PATH.read_text(encoding="utf-8"))
    checked = [RecallBaselineResult.model_validate(item) for item in checked_payload["results"]]

    assert checked_payload["schema_version"] == "cayu.recall_baseline_matrix.v1"
    assert checked_payload["corpus_revision"] == "public-cross-source-v1"
    assert [result.backend for result in actual] == ["memory", "sqlite"]
    for result, frozen in zip(actual, checked, strict=True):
        assert _without_latency(result) == _without_latency(frozen)
        assert result.metrics.latency_p50_ms >= 0.0
        assert result.metrics.latency_p95_ms >= result.metrics.latency_p50_ms
        assert result.metrics.recall_at_k == 1.0
        assert result.metrics.false_result_rate == 0.0
        assert result.metrics.stale_knowledge_rate == 0.0
        assert result.metrics.authorization_leak_rate == 0.0
        assert result.metrics.locator_correctness == 1.0
        assert result.metrics.false_complete_rate == 0.0
        assert result.metrics.partial_source_count == result.metrics.case_count + 1
        by_case = {case.case_id: case for case in result.cases}
        assert by_case["token-boundary-normalization"].candidate_count == 1
        assert by_case["relevance-before-identity"].continuation_channels == ("transcript.lexical",)
        assert by_case["scan-overflow-fails-closed"].source_statuses["transcript"] == ("partial")
        assert by_case["scan-overflow-fails-closed"].source_failure_codes["transcript"] == (
            "scan_limit"
        )
        assert by_case["scan-overflow-fails-closed"].continuation_channels == ()
        byte_case = by_case["byte-truncation-without-false-continuation"]
        assert byte_case.truncated is True
        assert byte_case.continuation_channels == ()
    assert [case.selected_identities for case in actual[0].cases] == [
        case.selected_identities for case in actual[1].cases
    ]


def test_recall_baseline_requires_empty_stores_and_runner_owned_configuration() -> None:
    async def run() -> None:
        corpus = load_recall_baseline_corpus(_CORPUS_PATH)
        sessions = InMemorySessionStore()
        await sessions.create(
            RunRequest(agent_name="existing", messages=[]),
            identity=SessionIdentity(provider_name="hermetic", model="none"),
        )
        with pytest.raises(ValueError, match="requires empty knowledge and session stores"):
            await run_recall_baseline(
                corpus,
                InMemoryKnowledgeStore(),
                sessions,
                backend="memory",
            )

        with pytest.raises(ValueError, match="runner-owned field 'source_set'"):
            await run_recall_baseline(
                corpus,
                InMemoryKnowledgeStore(),
                InMemorySessionStore(),
                backend="memory",
                configuration={"source_set": ["knowledge"]},
            )

    asyncio.run(run())


def _without_latency(result: RecallBaselineResult) -> dict:
    payload = result.model_dump(mode="json")
    payload["metrics"].pop("latency_p50_ms")
    payload["metrics"].pop("latency_p95_ms")
    for case in payload["cases"]:
        case.pop("latency_ms")
    return payload
