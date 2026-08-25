from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BASELINE = _ROOT / "benchmarks" / "memory" / "knowledge-maintenance-performance-v1.json"
_RUNNER = runpy.run_path(str(_ROOT / "scripts" / "run_knowledge_maintenance_performance.py"))


def test_knowledge_maintenance_performance_baseline_is_complete_and_bounded() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))

    assert report["schema_version"] == "cayu.knowledge_maintenance_performance.v1"
    assert report["workload"] == {
        "decision_count": 20,
        "sources_per_decision": 20,
        "applied_source_count": 400,
        "entry_count": 420,
        "provider_calls": 0,
    }
    assert report["control"] == {
        "kind": "current_runtime_zero_maintenance_decisions",
        "historical_pre_feature": False,
        "decision_count": 0,
        "entry_count": 420,
    }
    assert report["ceilings"] == _RUNNER["_CEILINGS"]
    assert report["ceiling_findings"] == []
    assert report["within_ceilings"] is True
    assert _RUNNER["_ceiling_findings"](report["results"]) == []
    assert [result["backend"] for result in report["results"]] == ["memory", "sqlite"]

    for result in report["results"]:
        for metric in (
            "zero_decision_entry_publish_latency",
            "preparation_latency_per_source",
            "application_latency_per_source",
            "exact_replay_latency",
            "receipt_load_latency",
        ):
            assert result[metric]["p50_ms"] <= result[metric]["p95_ms"]
        assert result["decision_count"] == 20
        assert result["sources_per_decision"] == 20
        assert result["applied_source_count"] == 400
        assert result["entry_count"] == 420

    sqlite = report["results"][1]
    assert sqlite["storage_byte_overhead"] > 0
    assert (
        sqlite["storage_bytes_per_applied_source"]
        <= report["ceilings"]["sqlite_storage_bytes_per_applied_source"]
    )


def test_knowledge_maintenance_performance_check_detects_every_regression_lane() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))
    regressed = deepcopy(report["results"])
    for result in regressed:
        result["zero_decision_entry_publish_latency"]["p95_ms"] = 10_000
        result["preparation_latency_per_source"]["p95_ms"] = 10_000
        result["application_latency_per_source"]["p95_ms"] = 10_000
        result["exact_replay_latency"]["p95_ms"] = 10_000
        result["receipt_load_latency"]["p95_ms"] = 10_000
    regressed[1]["storage_bytes_per_applied_source"] = 100_000

    findings = _RUNNER["_ceiling_findings"](regressed)
    assert {finding["metric"] for finding in findings} == set(report["ceilings"])
