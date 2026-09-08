"""Qualify the generated policy, and verify the gate cannot pass vacuously."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import runpy
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "_shipped_memory_quality", _ROOT / "scripts/_shipped_memory_quality.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PROBE
_SPEC.loader.exec_module(_PROBE)


@pytest.fixture(scope="module")
def report():
    return asyncio.run(_PROBE.run_shipped_default_quality())


def test_generated_default_runtime_matrix(report):
    assert report["passed"], report["findings"]
    assert report["policy_kind"] == "generated_standard_default"
    assert len(report["corpus_sha256"]) == 64
    assert "memory/context.py" in report["generated_source_sha256"]
    assert json.loads(json.dumps(report)) == report
    assert len(report["results"]) == 30
    for summary in report["summary"]:
        assert summary["case_count"] == 15
        assert summary["observed_turn_count"] == 17
        assert summary["incomplete_case_count"] == 0
        for metric in (
            "false_focused_items",
            "missing_focused_items",
            "false_offered_items",
            "missing_offered_items",
        ):
            assert summary[metric] == 0
        assert summary["maximum_projection_bytes"] > 0
        assert summary["runtime_turn_p95_ms"] >= summary["runtime_turn_p50_ms"] >= 0
    for case in report["results"]:
        configuration = case["configuration"]
        assert configuration["sources"]["include_knowledge"] is True
        assert configuration["sources"]["include_transcript"] is False
        assert configuration["delta_policy"] is None
        assert configuration["admission_policy"]["relevance_policy"] == "cayu.query_concepts.v4"
        assert case["configuration_sha256"] == _PROBE._digest(configuration)
    by_case = {(case["backend"], case["case_id"]): case for case in report["results"]}
    for backend in ("memory", "sqlite"):
        continuity = by_case[backend, "follow-up-and-topic-switch"]
        assert [row["focused"] for row in continuity["turns"]] == [
            ["rollback:1"],
            ["rollback:1"],
            ["weather:1"],
        ]
        assert continuity["turns"][1]["query_resolution"]["decision"] != "independent_query"
        assert by_case[backend, "empty-store"]["turns"][0]["projection_bytes"] == 0


@pytest.mark.parametrize(
    "mutation",
    ["silent", "false", "duplicate", "missing", "evidence", "coverage", "configuration", "failure"],
)
def test_gate_rejects_regressions_and_incomplete_evidence(report, mutation):
    results = deepcopy(report["results"])
    positive = next(case for case in results if case["case_id"] == "useful-with-distractors-1")
    row = positive["turns"][0]
    if mutation == "silent":
        row["focused"] = []
        row["admitted_count"] = 0
        # Changing the reported expectation must not change the frozen oracle.
        row["expected_focused"] = []
    elif mutation == "false":
        row["focused"] = ["picnic-0:1"]
    elif mutation == "duplicate":
        results.append(deepcopy(results[0]))
    elif mutation == "missing":
        results.pop()
    elif mutation == "evidence":
        row["exact_item_linkage"] = False
    elif mutation == "coverage":
        row["sources"][0]["state"] = "complete"
    elif mutation == "configuration":
        positive["configuration"]["sources"]["include_transcript"] = True
    else:
        positive["error"] = "RuntimeError"
        positive["error_stage"] = "turn-0:runtime"
    assert _PROBE.findings(results)


def test_shipped_mode_rejects_external_corpus():
    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts/run_recall_baseline.py"),
            "--shipped-default",
            "--corpus",
            "private.json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "cannot accept --corpus" in result.stderr


def test_fixed_records_fit_complete_preview_budget():
    for case in _PROBE._cases():
        for entry in case.entries:
            text = case.correction if case.correction is not None else entry.text
            assert 0 < len(text.encode("utf-8")) <= 240


def test_gate_rejects_corrupted_provider_content(monkeypatch):
    import cayu.runtime.memory_context as context

    original = context._provider_projection

    def corrupt(projection):
        payload = original(projection)
        for item in payload.get("focus", {}).get("items", []):
            read = item["read"]
            if read["entry_id"] == "rollback" and read["revision"] == 2:
                item["text"] = "Deployment rollback safeguards use image OLD."
            elif read["entry_id"] == "atlas":
                item["text"] = ""
            elif read["entry_id"] == "cache":
                item["text_complete"] = False
        for item in payload.get("offer", {}).get("items", []):
            item["preview"] = "Unrelated but nonempty preview."
        return payload

    monkeypatch.setattr(context, "_provider_projection", corrupt)
    report = asyncio.run(_PROBE.run_shipped_default_quality())
    assert not report["passed"]
    affected = {
        "current-revision",
        "exact-identifier",
        "title-supported-contract",
        "bounded-offers",
    }
    for case in report["results"]:
        assert case["error"] is None
        for row in case["turns"]:
            # References and durable evidence still agree; only the actual
            # provider-facing content (or its completeness claim) is corrupted.
            assert row["exact_current_read_references"]
            assert row["exact_item_linkage"]
            assert row["focused"] == row["expected_focused"]
            assert row["offered"] == row["expected_offered"]
            assert row["delivered_content_matches_fixture"] is (case["case_id"] not in affected)
        if case["case_id"] in affected:
            assert any(
                finding.startswith(f"{case['backend']}:{case['case_id']}:")
                for finding in report["findings"]
            )


def test_environment_configuration_is_restored(monkeypatch):
    monkeypatch.setenv("CAYU_MODEL", "operator-model")
    with _PROBE._default_environment():
        assert "CAYU_MODEL" not in _PROBE.os.environ
    assert _PROBE.os.environ["CAYU_MODEL"] == "operator-model"


def test_failed_gate_writes_report_and_exits_nonzero(monkeypatch, tmp_path):
    async def failed_report():
        return {"passed": False, "findings": ["injected regression"]}

    monkeypatch.setattr(_PROBE, "run_shipped_default_quality", failed_report)
    output = tmp_path / "failed.json"
    monkeypatch.setattr(
        sys, "argv", ["run_recall_baseline.py", "--shipped-default", "--output", str(output)]
    )
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(_ROOT / "scripts/run_recall_baseline.py"), run_name="__main__")
    assert raised.value.code == 1
    assert json.loads(output.read_text()) == {"passed": False, "findings": ["injected regression"]}
