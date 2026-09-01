from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BASELINE = _ROOT / "benchmarks/memory/knowledge-maintenance-governance-performance-v1.json"
_RUNNER = runpy.run_path(str(_ROOT / "scripts/run_knowledge_maintenance_governance_performance.py"))


def test_maintenance_governance_performance_baseline_is_complete_and_bounded() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))

    assert report["schema_version"] == ("cayu.knowledge_maintenance_governance_performance.v1")
    assert report["workload"] == {
        "operations_per_mode": 5,
        "operation_count": 10,
        "reviewed_route_operations": 5,
        "automatic_reject_operations": 5,
        "provider_calls": 0,
    }
    assert report["control"] == {
        "kind": "identical_evaluated_proposals_without_governance_outcomes",
        "historical_pre_feature": False,
        "operation_count": 10,
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
            "reviewed_route_latency",
            "automatic_reject_latency",
            "receipt_replay_latency",
        ):
            assert result[metric]["p50_ms"] <= result[metric]["p95_ms"]
        assert result["operations_per_mode"] == 5
        assert result["operation_count"] == 10

    sqlite = report["results"][1]
    assert sqlite["storage_byte_overhead"] > 0
    assert (
        sqlite["storage_bytes_per_governance_outcome"]
        <= report["ceilings"]["sqlite_storage_bytes_per_governance_outcome"]
    )


def test_maintenance_governance_performance_detects_every_regression_lane() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))
    regressed = deepcopy(report["results"])
    for result in regressed:
        result["reviewed_route_latency"]["p95_ms"] = 10_000
        result["automatic_reject_latency"]["p95_ms"] = 10_000
        result["receipt_replay_latency"]["p95_ms"] = 10_000
    regressed[1]["storage_bytes_per_governance_outcome"] = 1_000_000

    findings = _RUNNER["_ceiling_findings"](regressed)
    assert {finding["metric"] for finding in findings} == set(report["ceilings"])
