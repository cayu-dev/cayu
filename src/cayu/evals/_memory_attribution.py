"""Eval-owned projection of runtime memory attribution through a session tree."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from pydantic import ValidationError

from cayu.evals.memory_attribution import (
    EVAL_MEMORY_ATTRIBUTION_MAX_BYTES,
    EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES,
    EvalMemoryAttributionEvidenceV1,
    EvalMemoryAttributionSourceV1,
    EvalMemoryEvidenceCompleteness,
    EvalMemoryEvidenceLimitation,
    EvalMemorySourceAliasV1,
    EvalMemorySourceReferenceV1,
    eval_memory_attribution_fingerprint,
    eval_memory_source_alias,
    standard_eval_memory_attribution_bounds,
)
from cayu.evals.models import (
    Trajectory,
    _memory_source_expected_counts,
    _model_instance_python_input,
    _validate_trajectory_record_contract,
)
from cayu.memory_attribution import (
    MemoryAttribution,
    MemoryAttributionBounds,
    MemoryAttributionStatus,
    MemoryAttributionUnavailableReason,
)
from cayu.runtime.sessions import SessionStatus


def _trajectory_sources(
    trajectory: Trajectory,
    *,
    path: tuple[int, ...] = (),
) -> Iterator[tuple[tuple[int, ...], Trajectory]]:
    yield path, trajectory
    for index, child in enumerate(trajectory.children):
        yield from _trajectory_sources(child, path=(*path, index))


def _source_limitations(
    attribution: MemoryAttribution | None,
    *,
    expected_receipt_count: int,
    expected_exposure_count: int,
) -> tuple[EvalMemoryEvidenceLimitation, ...]:
    limitations: set[EvalMemoryEvidenceLimitation] = set()
    if attribution is None:
        limitations.add(EvalMemoryEvidenceLimitation.MISSING)
    elif attribution.status is MemoryAttributionStatus.TRUNCATED:
        # The runtime intentionally does not expose which of its independently
        # bounded collection limits truncated the projection.  Preserve that
        # uncertainty instead of falsely attributing it to one eval limit.
        limitations.add(EvalMemoryEvidenceLimitation.RUNTIME_ATTRIBUTION_TRUNCATED)
    elif attribution.status is MemoryAttributionStatus.REDACTED:
        limitations.add(EvalMemoryEvidenceLimitation.ALIAS_KEY_UNAVAILABLE)
    elif attribution.status is MemoryAttributionStatus.CONTRADICTORY:
        limitations.add(EvalMemoryEvidenceLimitation.CONTRADICTORY_LINEAGE)
    elif attribution.status is MemoryAttributionStatus.UNAVAILABLE:
        limitations.add(
            EvalMemoryEvidenceLimitation.STORE_UNSUPPORTED
            if attribution.reason is MemoryAttributionUnavailableReason.STORE_UNSUPPORTED
            else EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED
        )
    if (
        attribution is not None
        and attribution.status
        in {MemoryAttributionStatus.COMPLETE, MemoryAttributionStatus.REDACTED}
        and not attribution.truncated
        and (
            attribution.observed_receipt_count < expected_receipt_count
            or attribution.observed_exposure_count < expected_exposure_count
        )
    ):
        limitations.add(EvalMemoryEvidenceLimitation.DELETED)
    return tuple(sorted(limitations, key=str))


def _terminal_status(
    trajectory: Trajectory,
) -> Literal["completed", "failed", "interrupted"] | None:
    if trajectory.session is None:
        return None
    status = trajectory.session.status
    if status is SessionStatus.COMPLETED:
        return "completed"
    if status is SessionStatus.FAILED:
        return "failed"
    if status is SessionStatus.INTERRUPTED:
        return "interrupted"
    return None


def eval_memory_attribution_evidence_from_trajectory(
    trajectory: Trajectory,
    *,
    effective_bounds: MemoryAttributionBounds | None = None,
    effective_source_limit: int = EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES,
    effective_max_bytes: int = EVAL_MEMORY_ATTRIBUTION_MAX_BYTES,
    source_alias_key_id: str | None = None,
    source_alias_key: bytes | None = None,
    unavailable_reason: EvalMemoryEvidenceLimitation | None = None,
) -> EvalMemoryAttributionEvidenceV1:
    """Create one portable tree projection without reading unrestricted events.

    The trajectory must already contain runtime-projected ``MemoryAttribution``
    records.  This function only binds them to deterministic tree references,
    checks positive terminal evidence for missing receipts, and applies the eval
    publication contract.
    """

    if type(trajectory) is not Trajectory:
        raise TypeError("trajectory must be an exact Trajectory.")
    if (source_alias_key_id is None) != (source_alias_key is None):
        raise ValueError("Memory source alias identity and key must be supplied together.")
    if unavailable_reason is not None and unavailable_reason not in {
        EvalMemoryEvidenceLimitation.STORE_UNSUPPORTED,
        EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED,
        EvalMemoryEvidenceLimitation.ALIAS_KEY_UNAVAILABLE,
        EvalMemoryEvidenceLimitation.MISSING,
        EvalMemoryEvidenceLimitation.LEGACY,
        EvalMemoryEvidenceLimitation.CONTRADICTORY_LINEAGE,
        EvalMemoryEvidenceLimitation.SOURCE_TREE_INCOMPLETE,
        EvalMemoryEvidenceLimitation.CLOSURE_CHANGED,
        EvalMemoryEvidenceLimitation.DEADLINE_EXPIRED,
    }:
        raise ValueError("Forced unavailable memory evidence requires an unavailable reason.")
    validated = Trajectory.model_validate(_model_instance_python_input(trajectory))
    _validate_trajectory_record_contract(validated)
    bounds = effective_bounds or standard_eval_memory_attribution_bounds()
    bounds = MemoryAttributionBounds.model_validate(bounds.model_dump(mode="python"))
    if type(effective_source_limit) is not int:
        raise TypeError("effective_source_limit must be an integer.")
    if not 0 <= effective_source_limit <= EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES:
        raise ValueError(
            f"effective_source_limit must be between 0 and {EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES}."
        )
    if type(effective_max_bytes) is not int:
        raise TypeError("effective_max_bytes must be an integer.")
    if not 1 <= effective_max_bytes <= EVAL_MEMORY_ATTRIBUTION_MAX_BYTES:
        raise ValueError(
            f"effective_max_bytes must be between 1 and {EVAL_MEMORY_ATTRIBUTION_MAX_BYTES}."
        )

    flattened = tuple(_trajectory_sources(validated))
    retained = flattened[:effective_source_limit]
    global_limitations: set[EvalMemoryEvidenceLimitation] = set()
    omitted = max(len(flattened) - len(retained), 0)
    if omitted:
        global_limitations.add(EvalMemoryEvidenceLimitation.SOURCE_LIMIT)
        # A source-count cap limits retained material, not the truth state of
        # the already bounded trajectory. Preserve unavailable classifications
        # from omitted sources so a missing or contradictory descendant cannot
        # be downgraded to ordinary truncation. ``deleted`` still requires the
        # positive source record and therefore cannot be promoted globally.
        for _, node in flattened[effective_source_limit:]:
            terminal_status = _terminal_status(node)
            if node.session is None or terminal_status is None:
                global_limitations.add(EvalMemoryEvidenceLimitation.MISSING)
                continue
            expected_receipts, expected_exposures = _memory_source_expected_counts(node.events)
            global_limitations.update(
                limitation
                for limitation in _source_limitations(
                    node.memory_attribution,
                    expected_receipt_count=expected_receipts,
                    expected_exposure_count=expected_exposures,
                )
                if limitation is not EvalMemoryEvidenceLimitation.DELETED
            )
    if any(node.children_incomplete for _, node in flattened):
        global_limitations.add(EvalMemoryEvidenceLimitation.SOURCE_TREE_INCOMPLETE)
        omitted = max(omitted, 1)

    sources: list[EvalMemoryAttributionSourceV1] = []
    for path, node in retained:
        terminal_status = _terminal_status(node)
        if node.session is None or terminal_status is None:
            global_limitations.add(EvalMemoryEvidenceLimitation.MISSING)
            omitted += 1
            continue
        expected_receipts, expected_exposures = _memory_source_expected_counts(node.events)
        limitations = (
            (unavailable_reason,)
            if unavailable_reason is not None
            else _source_limitations(
                node.memory_attribution,
                expected_receipt_count=expected_receipts,
                expected_exposure_count=expected_exposures,
            )
        )
        source_alias = (
            None
            if source_alias_key_id is None or source_alias_key is None
            else eval_memory_source_alias(
                session_id=node.session.id,
                key_id=source_alias_key_id,
                key=source_alias_key,
            )
        )
        sources.append(
            EvalMemoryAttributionSourceV1(
                source=EvalMemorySourceReferenceV1(
                    role="root" if not path else "descendant",
                    tree_path=path,
                    session_alias=source_alias,
                ),
                terminal_status=terminal_status,
                expected_receipt_count=expected_receipts,
                expected_exposure_count=expected_exposures,
                attribution=(None if unavailable_reason is not None else node.memory_attribution),
                attribution_fingerprint=(
                    None
                    if unavailable_reason is not None or node.memory_attribution is None
                    else eval_memory_attribution_fingerprint(node.memory_attribution)
                ),
                limitations=limitations,
            )
        )

    if unavailable_reason is not None:
        global_limitations.add(unavailable_reason)
    all_limitations = global_limitations | {
        limitation for source in sources for limitation in source.limitations
    }
    unavailable = {
        EvalMemoryEvidenceLimitation.STORE_UNSUPPORTED,
        EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED,
        EvalMemoryEvidenceLimitation.ALIAS_KEY_UNAVAILABLE,
        EvalMemoryEvidenceLimitation.MISSING,
        EvalMemoryEvidenceLimitation.DELETED,
        EvalMemoryEvidenceLimitation.LEGACY,
        EvalMemoryEvidenceLimitation.CONTRADICTORY_LINEAGE,
        EvalMemoryEvidenceLimitation.SOURCE_TREE_INCOMPLETE,
        EvalMemoryEvidenceLimitation.CLOSURE_CHANGED,
        EvalMemoryEvidenceLimitation.DEADLINE_EXPIRED,
    }
    completeness = (
        EvalMemoryEvidenceCompleteness.UNAVAILABLE
        if all_limitations.intersection(unavailable)
        else (
            EvalMemoryEvidenceCompleteness.TRUNCATED
            if all_limitations or omitted
            else EvalMemoryEvidenceCompleteness.COMPLETE
        )
    )
    retained_sources = tuple(sources)
    while True:
        try:
            return EvalMemoryAttributionEvidenceV1.create(
                effective_bounds=bounds,
                effective_source_limit=effective_source_limit,
                effective_max_bytes=effective_max_bytes,
                completeness=completeness,
                limitations=tuple(sorted(global_limitations, key=str)),
                total_source_count=len(flattened),
                sources=retained_sources,
                omitted_source_count_at_least=max(
                    omitted,
                    len(flattened) - len(retained_sources),
                ),
            )
        except ValidationError as exc:
            if "exceeds its effective byte limit" not in str(exc) or not retained_sources:
                raise
            dropped_source = retained_sources[-1]
            retained_sources = retained_sources[:-1]
            global_limitations.add(EvalMemoryEvidenceLimitation.PROJECTION_BYTES_LIMIT)
            global_limitations.update(
                limitation
                for limitation in dropped_source.limitations
                if limitation is not EvalMemoryEvidenceLimitation.DELETED
            )
            all_limitations = global_limitations | {
                limitation
                for retained_source in retained_sources
                for limitation in retained_source.limitations
            }
            completeness = (
                EvalMemoryEvidenceCompleteness.UNAVAILABLE
                if all_limitations.intersection(unavailable)
                else EvalMemoryEvidenceCompleteness.TRUNCATED
            )


def eval_memory_attribution_evidence_from_runtime_source(
    *,
    terminal_status: Literal["completed", "failed", "interrupted"],
    attribution: MemoryAttribution,
    terminal_evidence_available: bool,
    terminal_evidence_limitation: EvalMemoryEvidenceLimitation | None,
    expected_receipt_count: int | None,
    expected_exposure_count: int | None,
    effective_bounds: MemoryAttributionBounds,
    source_alias: EvalMemorySourceAliasV1 | None,
    effective_source_limit: int = EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES,
    effective_max_bytes: int = EVAL_MEMORY_ATTRIBUTION_MAX_BYTES,
) -> EvalMemoryAttributionEvidenceV1:
    """Bind one runtime-owned attribution to an eval result without private IDs."""

    if terminal_status not in {"completed", "failed", "interrupted"}:
        raise ValueError("terminal_status must be completed, failed, or interrupted.")
    if type(attribution) is not MemoryAttribution:
        raise TypeError("attribution must be an exact MemoryAttribution.")
    if type(terminal_evidence_available) is not bool:
        raise TypeError("terminal_evidence_available must be a bool.")
    if terminal_evidence_limitation is not None and not isinstance(
        terminal_evidence_limitation,
        EvalMemoryEvidenceLimitation,
    ):
        raise TypeError("terminal_evidence_limitation must be an evidence limitation or None.")
    if terminal_evidence_available:
        if terminal_evidence_limitation is not None:
            raise ValueError("Available terminal evidence cannot carry an unavailable reason.")
        if type(expected_receipt_count) is not int or expected_receipt_count < 0:
            raise ValueError("Available terminal evidence requires a non-negative receipt count.")
        if type(expected_exposure_count) is not int or expected_exposure_count < 0:
            raise ValueError("Available terminal evidence requires a non-negative exposure count.")
    else:
        if expected_receipt_count is not None or expected_exposure_count is not None:
            raise ValueError("Unavailable terminal evidence cannot carry expected memory counts.")
        if terminal_evidence_limitation not in {
            EvalMemoryEvidenceLimitation.STORE_UNSUPPORTED,
            EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED,
            EvalMemoryEvidenceLimitation.MISSING,
            EvalMemoryEvidenceLimitation.CONTRADICTORY_LINEAGE,
            EvalMemoryEvidenceLimitation.DEADLINE_EXPIRED,
        }:
            raise ValueError("Unavailable terminal evidence requires a terminal limitation.")
    if source_alias is not None:
        if type(source_alias) is not EvalMemorySourceAliasV1:
            raise TypeError("source_alias must be an exact EvalMemorySourceAliasV1 or None.")
        source_alias = EvalMemorySourceAliasV1.model_validate(
            source_alias.model_dump(mode="python")
        )
    bounds = MemoryAttributionBounds.model_validate(effective_bounds.model_dump(mode="python"))
    validated = MemoryAttribution.model_validate(attribution.model_dump(mode="python"))
    limitations: tuple[EvalMemoryEvidenceLimitation, ...]
    if terminal_evidence_available:
        if expected_receipt_count is None or expected_exposure_count is None:
            raise AssertionError("Validated terminal memory counts are unavailable.")
        limitations = _source_limitations(
            validated,
            expected_receipt_count=expected_receipt_count,
            expected_exposure_count=expected_exposure_count,
        )
    else:
        if terminal_evidence_limitation is None:
            raise AssertionError("Validated terminal evidence limitation is unavailable.")
        limitations = (terminal_evidence_limitation,)
    unavailable = any(
        limitation
        in {
            EvalMemoryEvidenceLimitation.STORE_UNSUPPORTED,
            EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED,
            EvalMemoryEvidenceLimitation.ALIAS_KEY_UNAVAILABLE,
            EvalMemoryEvidenceLimitation.MISSING,
            EvalMemoryEvidenceLimitation.DELETED,
            EvalMemoryEvidenceLimitation.CONTRADICTORY_LINEAGE,
            EvalMemoryEvidenceLimitation.DEADLINE_EXPIRED,
        }
        for limitation in limitations
    )
    completeness = (
        EvalMemoryEvidenceCompleteness.UNAVAILABLE
        if unavailable
        else (
            EvalMemoryEvidenceCompleteness.TRUNCATED
            if limitations
            else EvalMemoryEvidenceCompleteness.COMPLETE
        )
    )
    source = EvalMemoryAttributionSourceV1(
        source=EvalMemorySourceReferenceV1(
            role="root",
            tree_path=(),
            session_alias=source_alias,
        ),
        terminal_status=terminal_status,
        expected_receipt_count=expected_receipt_count,
        expected_exposure_count=expected_exposure_count,
        attribution=validated if terminal_evidence_available else None,
        attribution_fingerprint=(
            eval_memory_attribution_fingerprint(validated) if terminal_evidence_available else None
        ),
        limitations=limitations,
    )
    global_limitations: set[EvalMemoryEvidenceLimitation] = set()
    retained_sources = (source,) if effective_source_limit else ()
    if not retained_sources:
        global_limitations.add(EvalMemoryEvidenceLimitation.SOURCE_LIMIT)
        global_limitations.update(limitations)
        completeness = (
            EvalMemoryEvidenceCompleteness.UNAVAILABLE
            if unavailable
            else EvalMemoryEvidenceCompleteness.TRUNCATED
        )
    while True:
        try:
            return EvalMemoryAttributionEvidenceV1.create(
                effective_bounds=bounds,
                effective_source_limit=effective_source_limit,
                effective_max_bytes=effective_max_bytes,
                completeness=completeness,
                limitations=tuple(sorted(global_limitations, key=str)),
                total_source_count=1,
                sources=retained_sources,
                omitted_source_count_at_least=1 - len(retained_sources),
            )
        except ValidationError as exc:
            if "exceeds its effective byte limit" not in str(exc) or not retained_sources:
                raise
            retained_sources = ()
            global_limitations.add(EvalMemoryEvidenceLimitation.PROJECTION_BYTES_LIMIT)
            global_limitations.update(limitations)
            completeness = (
                EvalMemoryEvidenceCompleteness.UNAVAILABLE
                if unavailable
                else EvalMemoryEvidenceCompleteness.TRUNCATED
            )


__all__ = [
    "eval_memory_attribution_evidence_from_runtime_source",
    "eval_memory_attribution_evidence_from_trajectory",
]
