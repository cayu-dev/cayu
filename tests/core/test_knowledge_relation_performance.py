from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BASELINE = _ROOT / "benchmarks" / "memory" / "knowledge-relation-performance-v1.json"
_RUNNER = runpy.run_path(str(_ROOT / "scripts" / "run_knowledge_relation_performance.py"))


def test_knowledge_relation_performance_baseline_is_complete_and_bounded() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))

    assert report["schema_version"] == "cayu.knowledge_relation_performance.v1"
    assert report["workload"] == {
        "matched_relation_count": 50,
        "unrelated_relation_count": 5_000,
        "published_relation_count": 5_050,
        "batch_size": 10,
        "query_iterations": 30,
        "provider_calls": 0,
    }
    assert report["control"] == {
        "kind": "current_runtime_zero_relations",
        "historical_pre_feature": False,
        "relation_count": 0,
        "entry_count": 5_053,
    }
    assert report["ceilings"] == _RUNNER["_CEILINGS"]
    assert report["ceiling_findings"] == []
    assert report["within_ceilings"] is True
    assert _RUNNER["_ceiling_findings"](report["results"]) == []
    assert [result["backend"] for result in report["results"]] == [
        "memory",
        "sqlite",
    ]

    for result in report["results"]:
        for metric in (
            "zero_relation_entry_publish_latency",
            "preparation_latency_per_relation",
            "relation_publish_latency_per_relation",
            "bounded_query_latency",
            "bounded_lineage_query_latency",
            "unrelated_lookup_latency",
            "unrelated_lineage_lookup_latency",
        ):
            assert result[metric]["p50_ms"] <= result[metric]["p95_ms"]
        assert result["matched_relation_count"] == 50
        assert result["unrelated_relation_count"] == 5_000
        assert result["published_relation_count"] == 5_050
        assert result["batch_count"] == 505

    sqlite = report["results"][1]
    assert sqlite["storage_byte_overhead"] > 0
    assert (
        sqlite["storage_bytes_per_relation"]
        <= report["ceilings"]["sqlite_storage_bytes_per_relation"]
    )


def test_knowledge_relation_performance_check_detects_every_regression_lane() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))
    regressed = deepcopy(report["results"])
    for result in regressed:
        result["zero_relation_entry_publish_latency"]["p95_ms"] = 10_000
        result["preparation_latency_per_relation"]["p95_ms"] = 10_000
        result["relation_publish_latency_per_relation"]["p95_ms"] = 10_000
        result["bounded_query_latency"]["p95_ms"] = 10_000
        result["bounded_lineage_query_latency"]["p95_ms"] = 10_000
        result["unrelated_lookup_latency"]["p95_ms"] = 10_000
        result["unrelated_lineage_lookup_latency"]["p95_ms"] = 10_000
    regressed[1]["storage_bytes_per_relation"] = 100_000

    findings = _RUNNER["_ceiling_findings"](regressed)
    assert {finding["metric"] for finding in findings} == set(report["ceilings"])
