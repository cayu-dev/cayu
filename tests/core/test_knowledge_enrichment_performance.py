from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BASELINE = _ROOT / "benchmarks/memory/knowledge-enrichment-jobs-performance-v1.json"
_RUNNER = runpy.run_path(str(_ROOT / "scripts/run_knowledge_enrichment_performance.py"))


def test_knowledge_enrichment_performance_baseline_is_complete_and_bounded() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))

    assert report["schema_version"] == "cayu.knowledge_enrichment_jobs_performance.v1"
    assert report["workload"] == {
        "candidate_count_per_operation": 1,
        "operation_count_per_backend": 20,
        "provider_calls": 0,
    }
    assert report["ceilings"] == _RUNNER["_CEILINGS"]
    assert report["ceiling_findings"] == []
    assert report["within_ceilings"] is True
    assert _RUNNER["_ceiling_findings"](report["results"]) == []
    assert report["correctness"] == {
        "acknowledgement_loss_reconciled": True,
        "exact_replay_component_calls": 0,
        "preparation_write_failure_did_not_redispatch_semantics": True,
        "provider_calls": 0,
        "worker_loss_recovered_after_store_reopen_and_lease_expiry": True,
    }
    assert [result["backend"] for result in report["results"]] == ["memory", "sqlite"]

    for result in report["results"]:
        for metric in (
            "enqueue_latency",
            "processing_latency",
            "exact_replay_latency",
            "empty_poll_latency",
        ):
            assert result[metric]["p50_ms"] <= result[metric]["p95_ms"]
        assert result["operation_count"] == 20
        assert result["generator_calls"] == 20
        assert result["evaluator_calls"] == 20
        assert result["replay_generator_calls"] == 0
        assert result["replay_evaluator_calls"] == 0
        assert result["max_job_result_json_bytes"] <= 64 * 1024

    sqlite = report["results"][1]
    assert sqlite["fresh_store_reopen_before_processing"] is True
    assert sqlite["populated_storage_bytes"] > sqlite["control_storage_bytes"]
    assert sqlite["storage_bytes_per_job"] <= report["ceilings"]["sqlite_storage_bytes_per_job"]


def test_knowledge_enrichment_performance_detects_every_regression_lane() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))
    regressed = deepcopy(report["results"])
    for result in regressed:
        result["enqueue_latency"]["p95_ms"] = 10_000
        result["processing_latency"]["p95_ms"] = 10_000
        result["exact_replay_latency"]["p95_ms"] = 10_000
        result["empty_poll_latency"]["p95_ms"] = 10_000
    regressed[1]["storage_bytes_per_job"] = 1_000_000

    findings = _RUNNER["_ceiling_findings"](regressed)
    assert {finding["metric"] for finding in findings} == set(report["ceilings"])
