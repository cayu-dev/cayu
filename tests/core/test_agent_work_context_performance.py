from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BASELINE = _ROOT / "benchmarks" / "memory" / "agent-work-context-performance-v1.json"
_RUNNER = runpy.run_path(str(_ROOT / "scripts" / "run_agent_work_context_performance.py"))


def test_agent_work_context_performance_baseline_is_complete_and_bounded() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))

    assert report["schema_version"] == "cayu.agent_work_context_performance.v1"
    assert report["workload"] == {
        "backends": ["memory", "sqlite"],
        "provider_calls": 0,
        "samples": 50,
    }
    assert report["control"] == {
        "historical_pre_feature": False,
        "kind": "current_store_with_zero_work_context_records",
    }
    assert report["ceilings"] == _RUNNER["_CEILINGS"]
    assert report["within_ceilings"] is True
    assert report["ceiling_findings"] == []
    assert _RUNNER["_ceiling_findings"](report["results"]) == []
    assert [result["backend"] for result in report["results"]] == ["memory", "sqlite"]

    for result in report["results"]:
        assert result["sample_count"] == 50
        for lane in (
            "zero_record_construction_latency",
            "current_read_latency",
            "revision_append_latency",
            "checkpoint_advance_latency",
        ):
            assert result[lane]["p50_ms"] <= result[lane]["p95_ms"]

    sqlite = report["results"][1]
    assert sqlite["incremental_storage_bytes"] > 0
    assert sqlite["populated_storage_bytes"] > sqlite["zero_record_storage_bytes"]
    assert (
        sqlite["storage_bytes_per_durable_record"]
        <= report["ceilings"]["sqlite_storage_bytes_per_durable_record"]
    )


def test_agent_work_context_performance_detects_every_regression_lane() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))
    regressed = deepcopy(report["results"])
    for result in regressed:
        result["zero_record_construction_latency"]["p95_ms"] = 10_000
        result["current_read_latency"]["p95_ms"] = 10_000
        result["revision_append_latency"]["p95_ms"] = 10_000
        result["checkpoint_advance_latency"]["p95_ms"] = 10_000
    regressed[1]["zero_record_construction_latency"]["p50_ms"] = 10_000
    regressed[1]["revision_append_latency"]["p50_ms"] = 10_000
    regressed[1]["checkpoint_advance_latency"]["p50_ms"] = 10_000
    regressed[1]["storage_bytes_per_durable_record"] = 100_000

    findings = _RUNNER["_ceiling_findings"](regressed)
    assert {finding["metric"] for finding in findings} == set(report["ceilings"])
