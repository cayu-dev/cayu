from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BASELINE = _ROOT / "benchmarks" / "memory" / "recall-subscription-performance-v1.json"
_RUNNER = runpy.run_path(str(_ROOT / "scripts" / "run_recall_subscription_performance.py"))


def test_recall_subscription_performance_baseline_is_complete_and_bounded() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))

    assert report["schema_version"] == "cayu.recall_subscription_performance.v1"
    assert report["workload"] == {
        "backends": ["memory", "sqlite"],
        "provider_calls": 0,
        "samples": 50,
    }
    assert report["control"] == {
        "historical_pre_feature": False,
        "kind": "current_store_with_bounded_claimed_subscriptions",
    }
    assert report["ceilings"] == _RUNNER["_CEILINGS"]
    assert report["within_ceilings"] is True
    assert report["ceiling_findings"] == []
    assert _RUNNER["_ceiling_findings"](report["results"]) == []
    assert [result["backend"] for result in report["results"]] == ["memory", "sqlite"]

    for result in report["results"]:
        assert result["sample_count"] == 50
        for lane in (
            "zero_due_latency",
            "silent_commit_latency",
            "wake_publish_latency",
            "wake_claim_latency",
            "wake_acknowledgement_latency",
        ):
            assert result[lane]["p50_ms"] <= result[lane]["p95_ms"]


def test_recall_subscription_performance_detects_every_regression_lane() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))
    regressed = deepcopy(report["results"])
    for result in regressed:
        for lane in (
            "zero_due_latency",
            "silent_commit_latency",
            "wake_publish_latency",
            "wake_claim_latency",
            "wake_acknowledgement_latency",
        ):
            result[lane]["p50_ms"] = 10_000
            result[lane]["p95_ms"] = 10_000

    findings = _RUNNER["_ceiling_findings"](regressed)
    assert {finding["metric"] for finding in findings} == set(report["ceilings"])
