from __future__ import annotations

import asyncio
import json
import runpy
from copy import deepcopy
from pathlib import Path

from cayu import InMemorySessionStore, SQLiteSessionStore

_ROOT = Path(__file__).resolve().parents[2]
_BASELINE = _ROOT / "benchmarks" / "memory" / "memory-reanchor-evaluation-v1.json"
_RUNNER = runpy.run_path(str(_ROOT / "scripts" / "run_memory_reanchor_evaluation.py"))


def test_memory_reanchor_evaluation_parses_base_and_delta_items() -> None:
    base = (
        '<cayu_automatic_memory version="2">\n'
        '{"focus":{"items":[{"read":{"entry_id":"base","revision":1}}]}}\n'
        "</cayu_automatic_memory>"
    )
    delta = (
        '<cayu_memory_delta version="2" sequence="1">\n'
        '{"items":[{"reason":"reanchored_current_revision",'
        '"read":{"entry_id":"delta","revision":2}}]}\n'
        "</cayu_memory_delta>"
    )

    assert _RUNNER["_memory_manifest_items"](base) == (
        {"read": {"entry_id": "base", "revision": 1}},
    )
    assert _RUNNER["_memory_manifest_items"](delta) == (
        {
            "reason": "reanchored_current_revision",
            "read": {"entry_id": "delta", "revision": 2},
        },
    )


def test_memory_reanchor_evaluation_is_complete_and_passes() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))

    assert report["schema_version"] == "cayu.memory_reanchor_evaluation.v1"
    assert report["methodology"] == {
        "backends": ["memory", "sqlite"],
        "cases": [
            "current_relevant",
            "disabled_control",
            "retained_control",
            "irrelevant_context",
            "same_entity_unrelated_aspect",
            "superseded_revision",
        ],
        "kind": "hermetic_real_runtime_projection_loss_matrix",
        "provider": "scripted",
        "provider_calls": 546,
        "history_limit_probe_prior_exposures": 32,
        "samples_per_case": 20,
        "semantic_model_claim": False,
        "token_measurement": (
            "Runtime ObservedDeltaContextEstimator over the exact rendered manifest."
        ),
    }
    assert report["ceilings"] == _RUNNER["_CEILINGS"]
    assert report["passed"] is True
    assert report["findings"] == []
    assert _RUNNER["_findings"](report["results"]) == []

    assert [result["backend"] for result in report["results"]] == ["memory", "sqlite"]
    for result in report["results"]:
        assert result["sample_count"] == 20
        assert result["case_count"] == 120
        assert result["useful_reanchor_precision"] == 1.0
        assert result["useful_reanchor_recall"] == 1.0
        assert result["false_injection_count"] == 0
        assert result["stale_injection_count"] == 0
        assert result["exact_revision_mismatch_count"] == 0
        assert result["duplicate_projection_count"] == 0
        assert result["reanchor_bytes"]["maximum"] > 0
        assert result["reanchor_bytes"]["maximum"] <= report["ceilings"]["max_reanchor_bytes"]
        assert result["estimated_reanchor_tokens"]["maximum"] > 0
        assert (
            result["estimated_reanchor_tokens"]["maximum"]
            <= report["ceilings"]["max_reanchor_estimated_tokens"]
        )
        for lane in (
            "zero_trigger_latency",
            "triggered_latency",
            "incremental_trigger_latency",
        ):
            assert result[lane]["p50_ms"] <= result[lane]["p95_ms"]
        assert result["case_outcomes"]["current_relevant"]["reanchor_count"] == 20
        assert result["case_outcomes"]["superseded_revision"]["reanchor_count"] == 0
        assert result["case_outcomes"]["same_entity_unrelated_aspect"]["reanchor_count"] == 0


def test_memory_reanchor_evaluation_exercises_current_runtime(monkeypatch) -> None:
    peaks: dict[str, int] = {}

    def observe_reads(store_type: type[InMemorySessionStore] | type[SQLiteSessionStore]) -> None:
        original = store_type.load_recall_item_exposures
        active = 0
        peaks[store_type.__name__] = 0

        async def load(store, session_id, exposure_id):
            nonlocal active
            active += 1
            peaks[store_type.__name__] = max(peaks[store_type.__name__], active)
            try:
                await asyncio.sleep(0)
                return await original(store, session_id, exposure_id)
            finally:
                active -= 1

        monkeypatch.setattr(store_type, "load_recall_item_exposures", load)

    observe_reads(InMemorySessionStore)
    observe_reads(SQLiteSessionStore)
    report = asyncio.run(_RUNNER["_run"](1))
    # Timing ceilings belong to the explicit benchmark command, not a loaded
    # test runner. Every semantic decision is asserted inside the live matrix.
    assert report["methodology"]["provider_calls"] == 90
    assert peaks == {"InMemorySessionStore": 8, "SQLiteSessionStore": 8}
    for result in report["results"]:
        assert result["useful_reanchor_precision"] == 1.0
        assert result["useful_reanchor_recall"] == 1.0
        assert result["false_injection_count"] == 0
        assert result["case_outcomes"]["same_entity_unrelated_aspect"]["reanchor_count"] == 0
        assert result["history_limit_probe"]["reanchor_count"] == 1
        assert result["history_limit_probe"]["false_injection_count"] == 0


def test_memory_reanchor_evaluation_detects_correctness_and_performance_regressions() -> None:
    report = json.loads(_BASELINE.read_text(encoding="utf-8"))
    regressed = deepcopy(report["results"])
    regressed[0].update(
        useful_reanchor_precision=0.5,
        useful_reanchor_recall=0.5,
        false_injection_count=1,
        stale_injection_count=1,
        exact_revision_mismatch_count=1,
        duplicate_projection_count=1,
    )
    regressed[0]["zero_trigger_latency"]["p95_ms"] = 10_000
    regressed[0]["triggered_latency"]["p95_ms"] = 10_000
    regressed[0]["incremental_trigger_latency"]["p95_ms"] = 10_000
    regressed[0]["reanchor_bytes"]["maximum"] = 100_000
    regressed[0]["estimated_reanchor_tokens"]["maximum"] = 100_000
    regressed[1]["zero_trigger_latency"].update(p50_ms=10_000, p95_ms=10_000)
    regressed[1]["triggered_latency"].update(p50_ms=10_000, p95_ms=10_000)
    regressed[1]["incremental_trigger_latency"].update(p50_ms=10_000, p95_ms=10_000)

    findings = _RUNNER["_findings"](regressed)
    assert {finding["metric"] for finding in findings} == {
        "useful_reanchor_precision",
        "useful_reanchor_recall",
        "false_injection_count",
        "stale_injection_count",
        "exact_revision_mismatch_count",
        "duplicate_projection_count",
        "memory_zero_trigger_p95_ms",
        "memory_triggered_p95_ms",
        "memory_incremental_p95_ms",
        "sqlite_zero_trigger_p50_ms",
        "sqlite_zero_trigger_p95_ms",
        "sqlite_triggered_p50_ms",
        "sqlite_triggered_p95_ms",
        "sqlite_incremental_p50_ms",
        "sqlite_incremental_p95_ms",
        "max_reanchor_bytes",
        "max_reanchor_estimated_tokens",
    }
