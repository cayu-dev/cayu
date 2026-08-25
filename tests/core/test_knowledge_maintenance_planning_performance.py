from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BASELINE = _ROOT / "benchmarks" / "memory" / "knowledge-maintenance-planning-performance-v1.json"
_RUNNER = runpy.run_path(
    str(_ROOT / "scripts" / "run_knowledge_maintenance_planning_performance.py")
)


def test_knowledge_maintenance_planning_performance_is_complete_and_bounded() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))

    assert report["schema_version"] == ("cayu.knowledge_maintenance_planning_performance.v1")
    assert report["workload"] == {
        "candidate_count": 50,
        "iterations": 30,
        "planner_calls_per_attempt": 1,
        "evaluator_calls_per_attempt": 1,
        "source_revalidations_per_attempt": 3,
        "provider_calls": 0,
        "model_calls": 0,
        "cost_micro_usd": 0,
        "store_writes_during_planning": 0,
    }
    assert report["control"] == {
        "kind": "current_runtime_zero_candidates",
        "historical_pre_feature": False,
        "candidate_count": 0,
        "store_reads": 0,
        "planner_calls": 0,
        "evaluator_calls": 0,
    }
    assert report["ceilings"] == _RUNNER["_CEILINGS"]
    assert report["ceiling_findings"] == []
    assert report["within_ceilings"] is True
    assert _RUNNER["_ceiling_findings"](report["results"]) == []
    assert [result["backend"] for result in report["results"]] == ["memory", "sqlite"]

    for result in report["results"]:
        assert result["candidate_count"] == 50
        assert result["iteration_count"] == 30
        assert result["source_revalidation_reads_per_attempt"] == 150
        assert result["planner_calls_per_attempt"] == 1
        assert result["evaluator_calls_per_attempt"] == 1
        assert result["provider_calls"] == result["model_calls"] == 0
        assert result["cost_micro_usd"] == 0
        assert result["knowledge_revision_mutations"] == 0
        assert result["planner_input_bytes"] > result["plan_bytes"] > 0
        assert result["evaluator_input_bytes"] > result["planner_input_bytes"]
        for metric in (
            "zero_candidate_planning_latency",
            "planning_latency",
            "planning_latency_per_candidate",
        ):
            assert result[metric]["p50_ms"] <= result[metric]["p95_ms"]


def test_knowledge_maintenance_planning_performance_detects_every_regression_lane() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))
    regressed = deepcopy(report["results"])
    for result in regressed:
        result["zero_candidate_planning_latency"]["p95_ms"] = 100_000
        result["planning_latency"]["p95_ms"] = 100_000
        result["planning_latency_per_candidate"]["p95_ms"] = 100_000
        result["planner_input_bytes_per_candidate"] = 100_000
        result["plan_bytes_per_candidate"] = 100_000
        result["evaluator_input_bytes_per_candidate"] = 100_000

    findings = _RUNNER["_ceiling_findings"](regressed)
    assert {finding["metric"] for finding in findings} == set(report["ceilings"])
