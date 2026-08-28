from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BASELINE = _ROOT / "benchmarks" / "memory" / "checkpoint-recall-performance-v1.json"
_RUNNER = runpy.run_path(str(_ROOT / "scripts" / "run_checkpoint_recall_performance.py"))


def test_checkpoint_recall_performance_baseline_is_complete_and_bounded() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))

    assert report["schema_version"] == "cayu.checkpoint_recall_performance.v1"
    assert report["workload"] == {
        "backends": ["memory", "sqlite"],
        "existing_records": 500,
        "max_delta_records": 249,
        "max_delta_revision_refs": 250,
        "provider_calls": 0,
        "samples": 50,
    }
    assert report["control"] == {
        "historical_pre_feature": False,
        "kind": "current_checkpoint_recall_with_no_provider_calls",
    }
    assert report["ceilings"] == _RUNNER["_CEILINGS"]
    assert report["within_ceilings"] is True
    assert report["ceiling_findings"] == []
    assert _RUNNER["_ceiling_findings"](report["results"]) == []
    assert [result["backend"] for result in report["results"]] == ["memory", "sqlite"]

    for result in report["results"]:
        assert result["sample_count"] == 50
        assert result["existing_record_count"] == 500
        assert result["max_delta_record_count"] == 249
        assert result["max_delta_revision_ref_count"] == 250
        for lane in (
            "full_index_latency",
            "delta_latency",
            "max_delta_latency",
            "no_work_latency",
        ):
            assert result[lane]["p50_ms"] <= result[lane]["p95_ms"]


def test_checkpoint_recall_performance_detects_every_regression_lane() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))
    regressed = deepcopy(report["results"])
    for result in regressed:
        result["full_index_latency"]["p95_ms"] = 10_000
        result["delta_latency"]["p95_ms"] = 10_000
        result["max_delta_latency"]["p95_ms"] = 10_000
        result["no_work_latency"]["p95_ms"] = 10_000
    regressed[1]["full_index_latency"]["p50_ms"] = 10_000
    regressed[1]["delta_latency"]["p50_ms"] = 10_000
    regressed[1]["max_delta_latency"]["p50_ms"] = 10_000
    regressed[1]["no_work_latency"]["p50_ms"] = 10_000

    findings = _RUNNER["_ceiling_findings"](regressed)
    assert {finding["metric"] for finding in findings} == set(report["ceilings"])
