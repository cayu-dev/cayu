from __future__ import annotations

import asyncio
from decimal import Decimal

from examples.tool_discovery_validation.scenario import run_scenario


def test_tool_discovery_validation_proves_lifecycle_and_bounded_evaluation() -> None:
    report = asyncio.run(run_scenario())
    lifecycle = report.lifecycle
    ranking = report.ranking
    direct = report.direct_catalogue
    discovery = report.search_tools

    assert report.evidence_scope == "deterministic_fixture"
    assert report.universal_savings_claimed is False

    assert lifecycle.provider_requests == lifecycle.model_steps == 9
    assert lifecycle.stable_core_tool_names == ("search_tools", "call_tool")
    assert lifecycle.distinct_tool_manifest_fingerprints == 1
    assert lifecycle.view_grant_counts_by_request == (0, 1, 1, 1, 1, 0, 0, 1, 1)
    assert lifecycle.parent_view_revision_before_resume == 1
    assert lifecycle.parent_view_revision_after_resume == 1
    assert lifecycle.child_view_revision_before_search == 0
    assert lifecycle.child_view_revision_after_search == 1
    assert lifecycle.parent_grant_count == 1
    assert lifecycle.child_grant_count_before_search == 0
    assert lifecycle.child_grant_count_after_search == 1
    assert lifecycle.generation_changed_on_fork is True
    assert lifecycle.parent_reference_survived_resume is True
    assert lifecycle.copied_parent_reference_rejections == 1
    assert lifecycle.child_reference_was_fresh is True
    assert lifecycle.references_omitted_from_public_evidence is True
    assert lifecycle.target_effects == 3
    assert lifecycle.typed_event_counts == {
        "session.completed": 3,
        "session.forked": 1,
        "tool.call.completed": 5,
        "tool.reference.rejected": 1,
        "request.footprint.recorded": 9,
    }

    assert ranking.case_count == ranking.top_1_hits == 6
    assert ranking.mean_reciprocal_rank == Decimal("1")
    assert all(case.observed_rank == 1 for case in ranking.cases)
    assert ranking.directly_exposed_tool_excluded is True

    assert direct.quality_passed is discovery.quality_passed is True
    assert direct.provider_requests == direct.model_steps == 3
    assert discovery.provider_requests == discovery.model_steps == 4
    assert direct.search_calls == 0
    assert discovery.search_calls == 1
    assert direct.unnecessary_searches == discovery.unnecessary_searches == 0
    assert direct.invalid_argument_attempts == discovery.invalid_argument_attempts == 1
    assert direct.invalid_argument_rejections == discovery.invalid_argument_rejections == 1
    assert direct.target_invocations_started == 2
    assert discovery.target_invocations_started == 1
    assert direct.target_effects == discovery.target_effects == 1
    assert direct.approval_requests == discovery.approval_requests == 0
    assert set(direct.provider_tool_counts) == {36}
    assert set(discovery.provider_tool_counts) == {2}
    assert direct.distinct_tool_manifest_fingerprints == 1
    assert discovery.distinct_tool_manifest_fingerprints == 1
    assert direct.input_tokens == 5_160
    assert discovery.input_tokens == 2_260
    assert direct.cache_read_input_tokens == 2_400
    assert discovery.cache_read_input_tokens == 900
    assert direct.cache_write_input_tokens == 1_200
    assert discovery.cache_write_input_tokens == 300
    assert direct.uncached_input_tokens == 1_560
    assert discovery.uncached_input_tokens == 1_060
    assert direct.observed_latency_ms >= 0
    assert discovery.observed_latency_ms >= 0
    assert direct.currency == discovery.currency == "USD"
    assert direct.estimated_cost == Decimal("0.003384")
    assert discovery.estimated_cost == Decimal("0.001625")

    serialized = report.model_dump_json()
    assert "cayu-tool-ref" not in serialized
    assert "tool_ref" not in serialized
