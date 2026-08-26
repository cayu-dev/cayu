from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import pytest
from pydantic import ValidationError
from tests._session_provenance import session_fixture

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.core.events import Event, EventType
from cayu.evals._memory_attribution import (
    eval_memory_attribution_evidence_from_runtime_source,
    eval_memory_attribution_evidence_from_trajectory,
)
from cayu.evals.memory_attribution import (
    EVAL_MEMORY_ATTRIBUTION_MAX_RETAINED_SOURCES_PER_RUN,
    EVAL_MEMORY_ATTRIBUTION_PROJECTION_BUDGET_BYTES,
    EVAL_MEMORY_ATTRIBUTION_RESULT_BUDGET_BYTES,
    EVAL_MEMORY_ATTRIBUTION_SCHEMA_VERSION,
    EVAL_MEMORY_ATTRIBUTION_SOURCE_BUDGET_BYTES,
    EvalMemoryAttributionEvidenceV1,
    EvalMemoryAttributionSourceV1,
    EvalMemoryEvidenceCompleteness,
    EvalMemoryEvidenceLimitation,
    EvalMemorySourceReferenceV1,
    eval_memory_attribution_bounds_for_trial_count,
    eval_memory_attribution_fingerprint,
    eval_memory_attribution_max_bytes_for_trial_count,
    eval_memory_attribution_source_limit_for_trial_count,
    eval_memory_attribution_summary,
    eval_memory_source_alias,
    standard_eval_memory_attribution_bounds,
)
from cayu.evals.models import Trajectory
from cayu.memory_attribution import (
    MemoryAttribution,
    MemoryAttributionBounds,
    MemoryAttributionStatus,
    MemoryAttributionUnavailableReason,
    MemoryContextExposureAttribution,
    MemoryEvidenceAlias,
    MemoryExposureTransitionAttribution,
)
from cayu.memory_evidence import ContextExposureEvidenceKind, ContextExposureState
from cayu.runtime.sessions import SessionStatus
from cayu.runtime.usage import SessionUsageSummary

_STATE_PATHS = {
    ContextExposureState.PLANNED: (ContextExposureState.PLANNED,),
    ContextExposureState.PREPARED: (
        ContextExposureState.PLANNED,
        ContextExposureState.PREPARED,
    ),
    ContextExposureState.DISPATCH_STARTED: (
        ContextExposureState.PLANNED,
        ContextExposureState.PREPARED,
        ContextExposureState.DISPATCH_STARTED,
    ),
    ContextExposureState.ACKNOWLEDGED: (
        ContextExposureState.PLANNED,
        ContextExposureState.PREPARED,
        ContextExposureState.DISPATCH_STARTED,
        ContextExposureState.ACKNOWLEDGED,
    ),
    ContextExposureState.COMPLETED: (
        ContextExposureState.PLANNED,
        ContextExposureState.PREPARED,
        ContextExposureState.DISPATCH_STARTED,
        ContextExposureState.ACKNOWLEDGED,
        ContextExposureState.COMPLETED,
    ),
    ContextExposureState.INDETERMINATE: (
        ContextExposureState.PLANNED,
        ContextExposureState.PREPARED,
        ContextExposureState.DISPATCH_STARTED,
        ContextExposureState.INDETERMINATE,
    ),
}
_EVIDENCE_KIND = {
    ContextExposureState.PLANNED: ContextExposureEvidenceKind.COMPOSITION_PLANNED,
    ContextExposureState.PREPARED: ContextExposureEvidenceKind.REQUEST_PREPARED,
    ContextExposureState.DISPATCH_STARTED: (ContextExposureEvidenceKind.DISPATCH_INTENT_COMMITTED),
    ContextExposureState.ACKNOWLEDGED: (ContextExposureEvidenceKind.PROVIDER_ACKNOWLEDGEMENT),
    ContextExposureState.COMPLETED: ContextExposureEvidenceKind.PROVIDER_COMPLETION,
    ContextExposureState.INDETERMINATE: ContextExposureEvidenceKind.AMBIGUOUS_TRANSPORT,
}


def _alias(
    kind: Literal["receipt", "exposure", "item", "interaction"],
    digit: str,
) -> MemoryEvidenceAlias:
    return MemoryEvidenceAlias(
        key_id="eval-memory-key",
        kind=kind,
        digest=digit * 64,
    )


def _attribution(state: ContextExposureState) -> MemoryAttribution:
    started = datetime(2026, 8, 25, tzinfo=UTC)
    states = _STATE_PATHS[state]
    transitions = tuple(
        MemoryExposureTransitionAttribution(
            revision=index,
            state=item,
            occurred_at=started + timedelta(seconds=index),
            evidence_kind=_EVIDENCE_KIND[item],
        )
        for index, item in enumerate(states)
    )
    exposure = MemoryContextExposureAttribution(
        exposure_alias=_alias("exposure", "1"),
        interaction_alias=_alias("interaction", "2"),
        projection_ordinal=0,
        model_step_id="mstep_" + "3" * 32,
        model_attempt_id="matt_" + "4" * 32,
        provider_attempt_id="patt_" + "5" * 32,
        created_at=transitions[0].occurred_at,
        updated_at=transitions[-1].occurred_at,
        state=state,
        state_revision=len(transitions) - 1,
        provider_exposure_proven=state
        in {ContextExposureState.ACKNOWLEDGED, ContextExposureState.COMPLETED},
        contributor_count=0,
        transitions=transitions,
        omitted_item_count_at_least=0,
    )
    return MemoryAttribution(
        status=MemoryAttributionStatus.COMPLETE,
        truncated=False,
        observed_receipt_count=0,
        observed_exposure_count=1,
        observed_item_count=0,
        omitted_receipt_count_at_least=0,
        omitted_exposure_count_at_least=0,
        omitted_item_count_at_least=0,
        exposures=(exposure,),
    )


def _empty_attribution() -> MemoryAttribution:
    return MemoryAttribution(
        status=MemoryAttributionStatus.COMPLETE,
        truncated=False,
        observed_receipt_count=0,
        observed_exposure_count=0,
        observed_item_count=0,
        omitted_receipt_count_at_least=0,
        omitted_exposure_count_at_least=0,
        omitted_item_count_at_least=0,
    )


def _runtime_source_evidence(
    *,
    terminal_status: Literal["completed", "failed", "interrupted"],
    attribution: MemoryAttribution,
    **kwargs: Any,
) -> EvalMemoryAttributionEvidenceV1:
    return eval_memory_attribution_evidence_from_runtime_source(
        terminal_status=terminal_status,
        attribution=attribution,
        terminal_evidence_available=True,
        terminal_evidence_limitation=None,
        expected_receipt_count=attribution.observed_receipt_count,
        expected_exposure_count=attribution.observed_exposure_count,
        effective_bounds=standard_eval_memory_attribution_bounds(),
        source_alias=None,
        **kwargs,
    )


@pytest.mark.parametrize(
    "state",
    [
        ContextExposureState.PLANNED,
        ContextExposureState.PREPARED,
        ContextExposureState.DISPATCH_STARTED,
        ContextExposureState.ACKNOWLEDGED,
        ContextExposureState.COMPLETED,
        ContextExposureState.INDETERMINATE,
    ],
)
def test_eval_memory_evidence_preserves_every_exposure_lifecycle_state(state) -> None:
    attribution = _attribution(state)
    evidence = _runtime_source_evidence(
        terminal_status="completed",
        attribution=attribution,
    )

    assert evidence.completeness is EvalMemoryEvidenceCompleteness.COMPLETE
    assert evidence.sources[0].attribution_fingerprint == (
        eval_memory_attribution_fingerprint(attribution)
    )
    assert evidence.sources[0].attribution is not None
    assert evidence.sources[0].attribution.exposures[0].state is state
    assert evidence.has_indeterminate_exposure is (state is ContextExposureState.INDETERMINATE)
    encoded = evidence.model_dump_json()
    assert EvalMemoryAttributionEvidenceV1.model_validate_json(encoded) == evidence


def test_eval_memory_attribution_summary_retains_bounded_lifecycle_evidence() -> None:
    evidence = _runtime_source_evidence(
        terminal_status="completed",
        attribution=_attribution(ContextExposureState.INDETERMINATE),
    )

    summary = eval_memory_attribution_summary(evidence)

    assert summary.startswith("complete · 1/1 source(s) retained")
    assert "indeterminate exposure" in summary
    assert "limitations none" in summary
    assert "lifecycle indeterminate" in summary


def test_eval_memory_evidence_distinguishes_proven_empty_from_unavailable_states() -> None:
    empty = _runtime_source_evidence(
        terminal_status="completed",
        attribution=_empty_attribution(),
    )
    assert empty.completeness is EvalMemoryEvidenceCompleteness.COMPLETE
    assert empty.proves_empty is True

    for limitation in (
        EvalMemoryEvidenceLimitation.MISSING,
        EvalMemoryEvidenceLimitation.LEGACY,
        EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED,
    ):
        unavailable = EvalMemoryAttributionEvidenceV1.unavailable(limitation)
        assert unavailable.completeness is EvalMemoryEvidenceCompleteness.UNAVAILABLE
        assert unavailable.proves_empty is False
        assert unavailable.sources == ()

    with pytest.raises(ValueError, match="positive retained source evidence"):
        EvalMemoryAttributionEvidenceV1.unavailable(EvalMemoryEvidenceLimitation.DELETED)

    with pytest.raises(ValidationError, match="requires one exact retained source tree"):
        EvalMemoryAttributionEvidenceV1.create(
            effective_bounds=standard_eval_memory_attribution_bounds(),
            completeness=EvalMemoryEvidenceCompleteness.COMPLETE,
            limitations=(),
            total_source_count=0,
            sources=(),
        )

    deleted_without_source = EvalMemoryAttributionEvidenceV1.unavailable().model_dump(mode="json")
    deleted_without_source["limitations"] = [EvalMemoryEvidenceLimitation.DELETED.value]
    deleted_without_source["revision"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="positive retained source evidence"):
        EvalMemoryAttributionEvidenceV1.model_validate(deleted_without_source)

    with pytest.raises(ValidationError, match="positive retained source evidence"):
        EvalMemoryAttributionSourceV1(
            source=EvalMemorySourceReferenceV1(role="root", tree_path=()),
            terminal_status="completed",
            limitations=(EvalMemoryEvidenceLimitation.DELETED,),
        )

    with pytest.raises(ValueError, match="Forced unavailable memory evidence"):
        eval_memory_attribution_evidence_from_trajectory(
            Trajectory(),
            unavailable_reason=EvalMemoryEvidenceLimitation.DELETED,
        )

    for status, reason, expected_limitation in (
        (
            MemoryAttributionStatus.UNAVAILABLE,
            MemoryAttributionUnavailableReason.EVIDENCE_READ_FAILED,
            EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED,
        ),
        (
            MemoryAttributionStatus.REDACTED,
            MemoryAttributionUnavailableReason.ALIAS_KEY_UNAVAILABLE,
            EvalMemoryEvidenceLimitation.ALIAS_KEY_UNAVAILABLE,
        ),
        (
            MemoryAttributionStatus.CONTRADICTORY,
            MemoryAttributionUnavailableReason.CONTRADICTORY_EVIDENCE,
            EvalMemoryEvidenceLimitation.CONTRADICTORY_LINEAGE,
        ),
    ):
        attribution = MemoryAttribution(
            status=status,
            truncated=False,
            reason=reason,
            observed_receipt_count=0,
            observed_exposure_count=0,
            observed_item_count=0,
            omitted_receipt_count_at_least=0,
            omitted_exposure_count_at_least=0,
            omitted_item_count_at_least=0,
        )
        evidence = _runtime_source_evidence(
            terminal_status="failed",
            attribution=attribution,
        )
        assert evidence.completeness is EvalMemoryEvidenceCompleteness.UNAVAILABLE
        assert evidence.proves_empty is False
        assert evidence.sources[0].limitations == (expected_limitation,)


def test_runtime_source_requires_positive_terminal_evidence_before_proving_empty() -> None:
    evidence = eval_memory_attribution_evidence_from_runtime_source(
        terminal_status="interrupted",
        attribution=_empty_attribution(),
        terminal_evidence_available=False,
        terminal_evidence_limitation=EvalMemoryEvidenceLimitation.MISSING,
        expected_receipt_count=None,
        expected_exposure_count=None,
        effective_bounds=standard_eval_memory_attribution_bounds(),
        source_alias=None,
    )

    assert evidence.completeness is EvalMemoryEvidenceCompleteness.UNAVAILABLE
    assert evidence.proves_empty is False
    assert evidence.sources[0].attribution is None
    assert evidence.sources[0].limitations == (EvalMemoryEvidenceLimitation.MISSING,)


def test_runtime_source_marks_terminally_proven_missing_records_as_deleted() -> None:
    bounds = MemoryAttributionBounds(
        max_receipts=1,
        max_exposures=2,
        max_items=3,
        max_source_bytes=1_024,
        max_projection_bytes=2_048,
    )
    evidence = eval_memory_attribution_evidence_from_runtime_source(
        terminal_status="completed",
        attribution=_empty_attribution(),
        terminal_evidence_available=True,
        terminal_evidence_limitation=None,
        expected_receipt_count=1,
        expected_exposure_count=0,
        effective_bounds=bounds,
        source_alias=None,
    )

    assert evidence.effective_bounds == bounds
    assert evidence.completeness is EvalMemoryEvidenceCompleteness.UNAVAILABLE
    assert evidence.proves_empty is False
    assert evidence.sources[0].expected_receipt_count == 1
    assert evidence.sources[0].limitations == (EvalMemoryEvidenceLimitation.DELETED,)


def test_eval_memory_evidence_marks_positive_missing_receipt_proof_as_deleted() -> None:
    attribution = _empty_attribution()
    source = EvalMemoryAttributionSourceV1(
        source=EvalMemorySourceReferenceV1(role="root", tree_path=()),
        terminal_status="completed",
        expected_receipt_count=1,
        attribution=attribution,
        attribution_fingerprint=eval_memory_attribution_fingerprint(attribution),
        limitations=(EvalMemoryEvidenceLimitation.DELETED,),
    )
    evidence = EvalMemoryAttributionEvidenceV1.create(
        effective_bounds=standard_eval_memory_attribution_bounds(),
        completeness=EvalMemoryEvidenceCompleteness.UNAVAILABLE,
        limitations=(),
        total_source_count=1,
        sources=(source,),
    )

    assert evidence.proves_empty is False
    assert evidence.sources[0].limitations == (EvalMemoryEvidenceLimitation.DELETED,)


def test_eval_memory_source_rejects_limitations_that_do_not_match_runtime_evidence() -> None:
    attribution = _empty_attribution()
    fingerprint = eval_memory_attribution_fingerprint(attribution)

    with pytest.raises(ValidationError, match="contradict.*runtime evidence"):
        EvalMemoryAttributionSourceV1(
            source=EvalMemorySourceReferenceV1(role="root", tree_path=()),
            terminal_status="completed",
            expected_receipt_count=0,
            attribution=attribution,
            attribution_fingerprint=fingerprint,
            limitations=(EvalMemoryEvidenceLimitation.DELETED,),
        )

    with pytest.raises(ValidationError, match="one unavailable limitation"):
        EvalMemoryAttributionSourceV1(
            source=EvalMemorySourceReferenceV1(role="root", tree_path=()),
            terminal_status="completed",
            limitations=(EvalMemoryEvidenceLimitation.RUNTIME_ATTRIBUTION_TRUNCATED,),
        )


def test_eval_memory_evidence_preserves_runtime_truncation_without_guessing_cause() -> None:
    attribution = MemoryAttribution(
        status=MemoryAttributionStatus.TRUNCATED,
        truncated=True,
        observed_receipt_count=0,
        observed_exposure_count=0,
        observed_item_count=0,
        omitted_receipt_count_at_least=1,
        omitted_exposure_count_at_least=0,
        omitted_item_count_at_least=0,
    )

    evidence = _runtime_source_evidence(
        terminal_status="completed",
        attribution=attribution,
    )

    assert evidence.completeness is EvalMemoryEvidenceCompleteness.TRUNCATED
    assert evidence.proves_empty is False
    assert evidence.sources[0].limitations == (
        EvalMemoryEvidenceLimitation.RUNTIME_ATTRIBUTION_TRUNCATED,
    )

    source = EvalMemoryAttributionSourceV1(
        source=EvalMemorySourceReferenceV1(role="root", tree_path=()),
        terminal_status="completed",
        expected_receipt_count=1,
        attribution=attribution,
        attribution_fingerprint=eval_memory_attribution_fingerprint(attribution),
        limitations=(EvalMemoryEvidenceLimitation.RUNTIME_ATTRIBUTION_TRUNCATED,),
    )
    bounded = EvalMemoryAttributionEvidenceV1.create(
        effective_bounds=standard_eval_memory_attribution_bounds(),
        completeness=EvalMemoryEvidenceCompleteness.TRUNCATED,
        limitations=(),
        total_source_count=1,
        sources=(source,),
    )
    assert EvalMemoryEvidenceLimitation.DELETED not in bounded.sources[0].limitations


def test_eval_memory_evidence_retains_exact_source_limit_boundary() -> None:
    attribution = _empty_attribution()
    fingerprint = eval_memory_attribution_fingerprint(attribution)
    paths = ((), *tuple(sorted((index,) for index in range(99))))
    sources = tuple(
        EvalMemoryAttributionSourceV1(
            source=EvalMemorySourceReferenceV1(
                role="root" if not path else "descendant",
                tree_path=path,
            ),
            terminal_status="completed",
            attribution=attribution,
            attribution_fingerprint=fingerprint,
        )
        for path in paths
    )

    evidence = EvalMemoryAttributionEvidenceV1.create(
        effective_bounds=standard_eval_memory_attribution_bounds(),
        completeness=EvalMemoryEvidenceCompleteness.TRUNCATED,
        limitations=(EvalMemoryEvidenceLimitation.SOURCE_LIMIT,),
        total_source_count=101,
        sources=sources,
        omitted_source_count_at_least=1,
    )

    assert evidence.retained_source_count == 100
    assert evidence.total_source_count == 101
    assert evidence.omitted_source_count_at_least == 1
    assert evidence.proves_empty is False
    assert (
        EvalMemoryAttributionEvidenceV1.model_validate_json(evidence.model_dump_json()) == evidence
    )

    false_limit = evidence.model_dump(mode="json")
    false_limit["total_source_count"] = 100
    with pytest.raises(ValidationError, match="source-limit classification"):
        EvalMemoryAttributionEvidenceV1.model_validate(false_limit)

    unsupported_omission = evidence.model_dump(mode="json")
    unsupported_omission["omitted_source_count_at_least"] = 2
    with pytest.raises(ValidationError, match="requires an incomplete source tree"):
        EvalMemoryAttributionEvidenceV1.model_validate(unsupported_omission)


def test_complete_eval_memory_evidence_rejects_sparse_source_trees() -> None:
    attribution = _empty_attribution()
    fingerprint = eval_memory_attribution_fingerprint(attribution)

    def source(path: tuple[int, ...]) -> EvalMemoryAttributionSourceV1:
        return EvalMemoryAttributionSourceV1(
            source=EvalMemorySourceReferenceV1(
                role="root" if not path else "descendant",
                tree_path=path,
            ),
            terminal_status="completed",
            attribution=attribution,
            attribution_fingerprint=fingerprint,
        )

    with pytest.raises(ValidationError, match="contiguous source child indexes"):
        EvalMemoryAttributionEvidenceV1.create(
            effective_bounds=standard_eval_memory_attribution_bounds(),
            completeness=EvalMemoryEvidenceCompleteness.COMPLETE,
            limitations=(),
            total_source_count=2,
            sources=(source(()), source((1,))),
        )

    with pytest.raises(ValidationError, match="requires every source parent"):
        EvalMemoryAttributionEvidenceV1.create(
            effective_bounds=standard_eval_memory_attribution_bounds(),
            completeness=EvalMemoryEvidenceCompleteness.COMPLETE,
            limitations=(),
            total_source_count=2,
            sources=(source(()), source((0, 0))),
        )


def test_eval_memory_source_alias_is_stable_secret_safe_and_round_trips() -> None:
    alias = eval_memory_source_alias(
        session_id="private-session-canary",
        key_id="eval-key-v1",
        key=b"k" * 32,
    )
    repeated = eval_memory_source_alias(
        session_id="private-session-canary",
        key_id="eval-key-v1",
        key=b"k" * 32,
    )
    other = eval_memory_source_alias(
        session_id="private-session-other",
        key_id="eval-key-v1",
        key=b"k" * 32,
    )

    assert alias == repeated
    assert alias != other
    assert "private-session" not in alias.model_dump_json()
    assert type(alias).model_validate_json(alias.model_dump_json()) == alias


def test_eval_memory_evidence_rejects_malformed_future_and_tampered_documents() -> None:
    evidence = _runtime_source_evidence(
        terminal_status="completed",
        attribution=_empty_attribution(),
    )
    future = evidence.model_dump(mode="json")
    future["schema_version"] = EVAL_MEMORY_ATTRIBUTION_SCHEMA_VERSION + 1
    with pytest.raises(ValidationError):
        EvalMemoryAttributionEvidenceV1.model_validate(future)

    for wrong_version in (True, 1.0, "1"):
        wrong_version_document = evidence.model_dump(mode="json")
        wrong_version_document["schema_version"] = wrong_version
        with pytest.raises(ValidationError, match="integer 1"):
            EvalMemoryAttributionEvidenceV1.model_validate(wrong_version_document)

    for wrong_limit in (True, 100.0, "100"):
        wrong_policy = evidence.model_dump(mode="json")
        wrong_policy["policy"]["max_sources"] = wrong_limit
        with pytest.raises(ValidationError, match="max_sources must be an integer"):
            EvalMemoryAttributionEvidenceV1.model_validate(wrong_policy)

    wrong_type = evidence.model_dump(mode="json")
    wrong_type["total_source_count"] = True
    with pytest.raises(ValidationError, match="valid integer"):
        EvalMemoryAttributionEvidenceV1.model_validate(wrong_type)

    oversized_count = evidence.model_dump(mode="json")
    oversized_count["total_source_count"] = MAX_DURABLE_JSON_INTEGER + 1
    with pytest.raises(ValidationError, match="less than or equal"):
        EvalMemoryAttributionEvidenceV1.model_validate(oversized_count)

    with pytest.raises(ValidationError, match="portable non-negative integers"):
        EvalMemorySourceReferenceV1(
            role="descendant",
            tree_path=(MAX_DURABLE_JSON_INTEGER + 1,),
        )

    with pytest.raises(ValidationError):
        eval_memory_source_alias(
            session_id="session",
            key_id="invalid\ud800key",
            key=b"k" * 32,
        )

    stale = evidence.model_dump(mode="json")
    stale["proves_empty"] = False
    with pytest.raises(ValidationError, match="empty proof"):
        EvalMemoryAttributionEvidenceV1.model_validate(stale)

    assert json.loads(evidence.model_dump_json())["revision"] == evidence.revision


def test_eval_memory_evidence_enforces_global_effective_bounds() -> None:
    attribution = _attribution(ContextExposureState.COMPLETED)
    fingerprint = eval_memory_attribution_fingerprint(attribution)
    sources = tuple(
        EvalMemoryAttributionSourceV1(
            source=EvalMemorySourceReferenceV1(
                role="root" if not path else "descendant",
                tree_path=path,
            ),
            terminal_status="completed",
            attribution=attribution,
            attribution_fingerprint=fingerprint,
        )
        for path in ((), (0,))
    )

    with pytest.raises(ValidationError, match="effective exposure bound"):
        bounded = standard_eval_memory_attribution_bounds()
        EvalMemoryAttributionEvidenceV1.create(
            effective_bounds=MemoryAttributionBounds.model_validate(
                {**bounded.model_dump(mode="python"), "max_exposures": 1}
            ),
            completeness=EvalMemoryEvidenceCompleteness.COMPLETE,
            limitations=(),
            total_source_count=2,
            sources=sources,
        )


def test_eval_memory_bounds_cover_both_fresh_closure_reads() -> None:
    trial_count = 16
    bounds = eval_memory_attribution_bounds_for_trial_count(trial_count)

    assert bounds.max_source_bytes * trial_count * 2 <= (
        EVAL_MEMORY_ATTRIBUTION_SOURCE_BUDGET_BYTES
    )
    assert bounds.max_projection_bytes * trial_count * 2 <= (
        EVAL_MEMORY_ATTRIBUTION_PROJECTION_BUDGET_BYTES
    )


@pytest.mark.parametrize("trial_count", [1, 16, 1_000, 10_000])
def test_eval_memory_bounds_cover_complete_published_run(
    trial_count: int,
) -> None:
    bounds = eval_memory_attribution_bounds_for_trial_count(trial_count)
    source_limit = eval_memory_attribution_source_limit_for_trial_count(trial_count)
    max_bytes = eval_memory_attribution_max_bytes_for_trial_count(trial_count)
    unavailable = EvalMemoryAttributionEvidenceV1.unavailable(
        effective_bounds=bounds,
        effective_source_limit=source_limit,
        effective_max_bytes=max_bytes,
    )

    assert max_bytes * trial_count <= EVAL_MEMORY_ATTRIBUTION_RESULT_BUDGET_BYTES
    assert source_limit * trial_count <= EVAL_MEMORY_ATTRIBUTION_MAX_RETAINED_SOURCES_PER_RUN
    assert len(unavailable.model_dump_json().encode("utf-8")) <= max_bytes


def test_eval_memory_projection_truncates_sources_to_preselected_result_budget() -> None:
    attribution = _empty_attribution()
    evidence = _runtime_source_evidence(
        terminal_status="completed",
        attribution=attribution,
        effective_source_limit=1,
        effective_max_bytes=943,
    )

    assert evidence.completeness is EvalMemoryEvidenceCompleteness.TRUNCATED
    assert evidence.limitations == (EvalMemoryEvidenceLimitation.PROJECTION_BYTES_LIMIT,)
    assert evidence.total_source_count == 1
    assert evidence.retained_source_count == 0
    assert evidence.omitted_source_count_at_least == 1
    assert len(evidence.model_dump_json().encode("utf-8")) <= 943


def test_eval_memory_source_cap_preserves_omitted_unavailable_classification() -> None:
    session = session_fixture(
        id="memory-root",
        agent_name="agent",
        provider_name="provider",
        model="model",
        causal_budget_id="budget",
        status=SessionStatus.COMPLETED,
    )
    trajectory = Trajectory(
        session=session,
        events=(Event(type=EventType.SESSION_COMPLETED, session_id=session.id),),
        usage_summary=SessionUsageSummary(session_id=session.id),
        memory_attribution=_empty_attribution(),
        children=(Trajectory(),),
    )

    evidence = eval_memory_attribution_evidence_from_trajectory(
        trajectory,
        effective_source_limit=1,
    )

    assert evidence.completeness is EvalMemoryEvidenceCompleteness.UNAVAILABLE
    assert evidence.limitations == (
        EvalMemoryEvidenceLimitation.MISSING,
        EvalMemoryEvidenceLimitation.SOURCE_LIMIT,
    )
    assert evidence.total_source_count == 2
    assert evidence.retained_source_count == 1
    assert evidence.omitted_source_count_at_least == 1
