from __future__ import annotations

import asyncio

from examples.tool_exposure_economics.scenario import run_scenario


def test_tool_exposure_economics_reports_the_complete_paired_fixture() -> None:
    report = asyncio.run(run_scenario())
    stable = report.stable_broad
    narrow = report.changing_narrow

    assert report.evidence_scope == "deterministic_fixture"
    assert report.universal_savings_claimed is False
    assert stable.requests == narrow.requests == 2
    assert stable.retries == narrow.retries == 0
    assert stable.quality_passed is narrow.quality_passed is True
    assert stable.exposure_profiles == ("stable-broad", "stable-broad")
    assert narrow.exposure_profiles == ("narrow-phase-one", "narrow-phase-two")
    assert stable.profile_changes == 0
    assert narrow.profile_changes == 1
    assert len(set(stable.tool_manifest_fingerprints)) == 1
    assert len(set(narrow.tool_manifest_fingerprints)) == 2
    assert stable.cache_prefix_fingerprints[0] is None
    assert narrow.cache_prefix_fingerprints[0] is None
    assert stable.cache_prefix_fingerprints[1] is not None
    assert narrow.cache_prefix_fingerprints[1] is not None
    assert stable.cache_prefix_fingerprints[1] != narrow.cache_prefix_fingerprints[1]
    assert stable.cache_read_input_tokens == 800
    assert stable.cache_write_input_tokens == 800
    assert narrow.cache_read_input_tokens == narrow.cache_write_input_tokens == 0
    assert stable.input_tokens == 2_000
    assert narrow.input_tokens == 1_350
    assert stable.currency == narrow.currency == "USD"
    assert stable.estimated_cost > narrow.estimated_cost > 0
