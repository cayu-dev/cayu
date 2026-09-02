from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BASELINE = _ROOT / "benchmarks/memory/knowledge-semantic-watch-performance-v1.json"
_RUNNER = runpy.run_path(str(_ROOT / "scripts/run_knowledge_semantic_watch_performance.py"))


def test_semantic_watch_performance_baseline_is_complete_and_bounded() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))

    assert report["schema_version"] == "cayu.knowledge_semantic_watch_performance.v1"
    assert report["workload"] == {
        "operation_count": 10,
        "policy_calls": 10,
        "provider_calls": 0,
        "replay_policy_calls": 0,
    }
    assert report["control"] == {
        "kind": "identical_current_runtime_knowledge_without_watch_outcomes",
        "historical_pre_feature": False,
        "operation_count": 10,
    }
    assert report["ceilings"] == _RUNNER["_CEILINGS"]
    assert report["ceiling_findings"] == []
    assert report["within_ceilings"] is True
    assert _RUNNER["_ceiling_findings"](report["results"]) == []
    assert [result["backend"] for result in report["results"]] == ["memory", "sqlite"]

    for result in report["results"]:
        assert result["evaluation_latency"]["p50_ms"] <= result["evaluation_latency"]["p95_ms"]
        assert (
            result["receipt_replay_latency"]["p50_ms"] <= result["receipt_replay_latency"]["p95_ms"]
        )
        assert result["operation_count"] == 10
        assert result["max_receipt_json_bytes"] <= 384_000

    sqlite = report["results"][1]
    assert sqlite["storage_byte_overhead"] > 0
    assert (
        sqlite["storage_bytes_per_outcome"]
        <= report["ceilings"]["sqlite_storage_bytes_per_outcome"]
    )


def test_semantic_watch_performance_detects_every_regression_lane() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))
    regressed = deepcopy(report["results"])
    for result in regressed:
        result["evaluation_latency"]["p95_ms"] = 10_000
        result["receipt_replay_latency"]["p95_ms"] = 10_000
    regressed[1]["storage_bytes_per_outcome"] = 1_000_000

    findings = _RUNNER["_ceiling_findings"](regressed)
    assert {finding["metric"] for finding in findings} == set(report["ceilings"])
