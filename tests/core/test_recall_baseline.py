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
from cayu.runtime.sessions import TRANSCRIPT_SEARCH_INDEX_VERSION
from cayu.storage import (
    InMemoryKnowledgeStore,
    SQLiteKnowledgeStore,
    SQLiteSessionStore,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_PATH = _REPOSITORY_ROOT / "benchmarks/memory/recall-corpus-v2.json"
_RESULTS_PATH = _REPOSITORY_ROOT / "benchmarks/memory/recall-baseline-results-v2.json"
_TRANSCRIPT_INDEX_PREFIX = "cayu.transcript.text.v1+cayu.transcript.tokenizer.v1+unicode-"


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

    assert checked_payload["schema_version"] == "cayu.recall_baseline_matrix.v2"
    assert checked_payload["corpus_revision"] == "public-cross-source-admission-v2"
    assert [result.backend for result in actual] == ["memory", "sqlite"]
    for result, frozen in zip(actual, checked, strict=True):
        assert _without_latency(result) == _without_latency(frozen)
        assert {case.channel_index_versions["transcript.lexical"] for case in result.cases} == {
            TRANSCRIPT_SEARCH_INDEX_VERSION
        }
        assert result.metrics.latency_p50_ms >= 0.0
        assert result.metrics.latency_p95_ms >= result.metrics.latency_p50_ms
        assert result.metrics.recall_at_k == 1.0
        assert result.metrics.false_result_rate == 0.0
        assert result.metrics.stale_knowledge_rate == 0.0
        assert result.metrics.authorization_leak_rate == 0.0
        assert result.metrics.locator_correctness == 1.0
        assert result.metrics.false_complete_rate == 0.0
        assert result.metrics.injected_precision == 1.0
        assert result.metrics.offer_precision == 1.0
        assert result.metrics.silent_precision == 1.0
        assert result.metrics.false_injection_rate == 0.0
        assert result.metrics.stale_injection_rate == 0.0
        assert result.metrics.unauthorized_injection_rate == 0.0
        assert result.metrics.mean_injected_source_diversity > 0.0
        assert result.metrics.admission_truncated_case_count >= 2
        assert result.metrics.admission_latency_p50_ms >= 0.0
        assert result.metrics.admission_latency_p95_ms >= result.metrics.admission_latency_p50_ms
        assert result.metrics.required_source_failure_closed is True
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
        duplicate_case = by_case["duplicate-provenance"]
        assert len(duplicate_case.injected_identities) == 1
        assert len(duplicate_case.offered_identities) == 1
        assert duplicate_case.admission_truncated is True
        clipping_case = by_case["current-knowledge-and-historical-transcript"]
        assert len(clipping_case.injected_identities) == 1
        assert len(clipping_case.offered_identities) == 1
        assert clipping_case.admission_truncated is True
        threshold_case = by_case["admission-threshold-ladder"]
        assert len(threshold_case.injected_identities) == 1
        assert len(threshold_case.offered_identities) == 1
        assert len(threshold_case.silent_identities) == 1
        malicious_case = by_case["malicious-recalled-text"]
        assert len(malicious_case.injected_identities) == 1
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
    payload["metrics"].pop("admission_latency_p50_ms")
    payload["metrics"].pop("admission_latency_p95_ms")
    for case in payload["cases"]:
        case.pop("latency_ms")
        case.pop("admission_latency_ms")
        transcript_version = case["channel_index_versions"]["transcript.lexical"]
        unicode_version = transcript_version.removeprefix(_TRANSCRIPT_INDEX_PREFIX)
        assert transcript_version.startswith(_TRANSCRIPT_INDEX_PREFIX)
        assert len(unicode_version.split(".")) == 3
        assert all(part.isdigit() for part in unicode_version.split("."))
        case["channel_index_versions"]["transcript.lexical"] = (
            f"{_TRANSCRIPT_INDEX_PREFIX}<runtime>"
        )
    return payload
