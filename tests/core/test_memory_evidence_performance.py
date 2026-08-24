from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BASELINE = _ROOT / "benchmarks" / "memory" / "memory-evidence-performance-v1.json"
_RUNNER = runpy.run_path(str(_ROOT / "scripts" / "run_memory_evidence_performance.py"))


def test_memory_evidence_performance_baseline_is_complete_and_within_ceilings() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))

    assert report["schema_version"] == "cayu.memory_evidence_performance.v1"
    assert report["workload"] == {
        "items_per_pair": 1,
        "provider_calls": 0,
        "receipt_exposure_pairs": 50,
    }
    assert report["control"] == {
        "historical_pre_feature": False,
        "kind": "current_runtime_zero_memory_records",
        "receipt_exposure_pairs": 0,
    }
    assert report["ceilings"] == _RUNNER["_CEILINGS"]
    assert report["within_ceilings"] is True
    assert report["ceiling_findings"] == []
    assert _RUNNER["_ceiling_findings"](report["results"]) == []
    assert [result["backend"] for result in report["results"]] == ["memory", "sqlite"]

    for result in report["results"]:
        assert result["preparation_latency"]["p50_ms"] <= result["preparation_latency"]["p95_ms"]
        assert result["persistence_latency"]["p50_ms"] <= result["persistence_latency"]["p95_ms"]
        assert (
            result["zero_record_runtime_evidence_latency"]["p50_ms"]
            <= result["zero_record_runtime_evidence_latency"]["p95_ms"]
        )
        assert (
            result["populated_runtime_evidence_latency"]["p50_ms"]
            <= result["populated_runtime_evidence_latency"]["p95_ms"]
        )
        assert result["incremental_projection_bytes"] > 0
        assert (
            result["projection_bytes_per_pair"] <= report["ceilings"]["projection_bytes_per_pair"]
        )

    sqlite = report["results"][1]
    assert sqlite["storage_byte_overhead"] > 0
    assert sqlite["storage_bytes_per_pair"] <= report["ceilings"]["sqlite_storage_bytes_per_pair"]


def test_memory_evidence_performance_check_detects_each_regression_lane() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))
    regressed = deepcopy(report["results"])
    regressed[0]["preparation_latency"]["p95_ms"] = 10_000
    regressed[0]["persistence_latency"]["p95_ms"] = 10_000
    regressed[0]["zero_record_runtime_evidence_latency"]["p95_ms"] = 10_000
    regressed[0]["incremental_projection_latency"]["p95_ms"] = 10_000
    regressed[0]["projection_bytes_per_pair"] = 10_000
    regressed[1]["persistence_latency"]["p95_ms"] = 10_000
    regressed[1]["zero_record_runtime_evidence_latency"]["p95_ms"] = 10_000
    regressed[1]["storage_bytes_per_pair"] = 100_000

    findings = _RUNNER["_ceiling_findings"](regressed)
    assert {finding["metric"] for finding in findings} == {
        "memory_persistence_p95_ms",
        "memory_zero_record_runtime_evidence_p95_ms",
        "preparation_p95_ms",
        "projection_bytes_per_pair",
        "projection_overhead_p95_ms_per_pair",
        "sqlite_persistence_p95_ms",
        "sqlite_zero_record_runtime_evidence_p95_ms",
        "sqlite_storage_bytes_per_pair",
    }
