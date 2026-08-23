from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from cayu._validation import require_clean_nonblank
from cayu.core.events import Event, EventType, event_payload_authority_is_runtime_generated
from cayu.core.messages import Message, MessageRole, TextPart
from cayu.evals.models import (
    Trajectory,
    TrajectoryProbes,
    _trajectory_promotion_capture_sha256,
    _validate_trajectory_record_contract,
)
from cayu.runtime.app import CayuApp
from cayu.runtime.sessions import (
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS,
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_RECORD_BYTES,
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TOTAL_BYTES,
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TRANSCRIPT_RECORDS,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_EVENTS,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_TOTAL_BYTES,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_TRANSCRIPT_RECORDS,
    Session,
    SessionInputContractEvidence,
    SessionLineageNode,
    SessionLineageOrigin,
    SessionLineageQuery,
    SessionOrder,
    SessionQuery,
    TerminalSessionEvidence,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
    TerminalSessionEvidenceLimits,
    parse_session_input_contract_evidence,
)
from cayu.runtime.usage import SessionUsageSummary, session_usage_summary

# Fresh evals retain descendant evidence for assertions and replay. Page the durable
# parent index instead of assuming the first page is complete, while retaining a hard
# stop so a corrupt or adversarial tree cannot keep a trial alive indefinitely.
_CHILD_TRAJECTORY_PAGE_SIZE = 1000
_CHILD_TRAJECTORY_MAX_PAGES = 100
_STRICT_CHILD_PAGE_SIZE = 100
_SESSION_TRAJECTORY_DEFAULT_MAX_SESSIONS = 100
_SESSION_TRAJECTORY_HARD_MAX_SESSIONS = 500
_SESSION_TRAJECTORY_DEFAULT_MAX_DEPTH = 32
_SESSION_TRAJECTORY_HARD_MAX_DEPTH = 32
_SESSION_TRAJECTORY_HARD_MAX_LINEAGE_CANDIDATES = 500
_ORIGIN_EVENT_TYPES = (EventType.SESSION_STARTED, EventType.SESSION_FORKED)


class SessionTrajectoryBounds(BaseModel):
    """Global retained-evidence limits for one production-session trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_sessions: StrictInt = Field(
        default=_SESSION_TRAJECTORY_DEFAULT_MAX_SESSIONS,
        ge=1,
        le=_SESSION_TRAJECTORY_HARD_MAX_SESSIONS,
    )
    max_depth: StrictInt = Field(
        default=_SESSION_TRAJECTORY_DEFAULT_MAX_DEPTH,
        ge=1,
        le=_SESSION_TRAJECTORY_HARD_MAX_DEPTH,
    )
    max_events: StrictInt = Field(
        default=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS,
        ge=1,
        le=TERMINAL_SESSION_EVIDENCE_HARD_MAX_EVENTS,
    )
    max_transcript_records: StrictInt = Field(
        default=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TRANSCRIPT_RECORDS,
        ge=0,
        le=TERMINAL_SESSION_EVIDENCE_HARD_MAX_TRANSCRIPT_RECORDS,
    )
    max_record_bytes: StrictInt = Field(
        default=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_RECORD_BYTES,
        ge=1,
        le=TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES,
    )
    max_total_bytes: StrictInt = Field(
        default=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TOTAL_BYTES,
        ge=1,
        le=TERMINAL_SESSION_EVIDENCE_HARD_MAX_TOTAL_BYTES,
    )


class SessionTrajectoryErrorCode(StrEnum):
    """Stable reason a durable session tree cannot become exact eval evidence."""

    STORE_UNSUPPORTED = "store_unsupported"
    EVIDENCE_READ_FAILED = "evidence_read_failed"
    TERMINAL_EVIDENCE_REJECTED = "terminal_evidence_rejected"
    DESCENDANT_ENUMERATION_FAILED = "descendant_enumeration_failed"
    ORIGIN_EVIDENCE_REJECTED = "origin_evidence_rejected"
    PARENT_CONTRADICTION = "parent_contradiction"
    CYCLE_DETECTED = "cycle_detected"
    SESSION_LIMIT_EXCEEDED = "session_limit_exceeded"
    DEPTH_LIMIT_EXCEEDED = "depth_limit_exceeded"
    CLOSURE_CHANGED = "closure_changed"
    EVIDENCE_INCONSISTENT = "evidence_inconsistent"


_SESSION_TRAJECTORY_ERROR_MESSAGES = {
    SessionTrajectoryErrorCode.STORE_UNSUPPORTED: (
        "The configured session store does not support the required bounded reads."
    ),
    SessionTrajectoryErrorCode.EVIDENCE_READ_FAILED: (
        "The session store failed while reading trajectory evidence."
    ),
    SessionTrajectoryErrorCode.TERMINAL_EVIDENCE_REJECTED: (
        "A session in the trajectory was not eligible exact terminal evidence."
    ),
    SessionTrajectoryErrorCode.DESCENDANT_ENUMERATION_FAILED: (
        "The durable descendant set could not be enumerated completely."
    ),
    SessionTrajectoryErrorCode.ORIGIN_EVIDENCE_REJECTED: (
        "A child session has missing, contradictory, or oversized origin evidence."
    ),
    SessionTrajectoryErrorCode.PARENT_CONTRADICTION: (
        "A child session contradicts its durable parent relationship."
    ),
    SessionTrajectoryErrorCode.CYCLE_DETECTED: (
        "The durable session lineage contains a cycle or repeated identity."
    ),
    SessionTrajectoryErrorCode.SESSION_LIMIT_EXCEEDED: (
        "The capture exceeds a retained-session or lineage-candidate limit."
    ),
    SessionTrajectoryErrorCode.DEPTH_LIMIT_EXCEEDED: (
        "The capture exceeds the configured session-tree depth limit."
    ),
    SessionTrajectoryErrorCode.CLOSURE_CHANGED: (
        "The admitted durable descendant closure changed during capture."
    ),
    SessionTrajectoryErrorCode.EVIDENCE_INCONSISTENT: (
        "The reconstructed trajectory violates its durable evidence contract."
    ),
}


class SessionTrajectoryError(RuntimeError):
    """Typed fail-closed error from :func:`trajectory_from_session`."""

    def __init__(
        self,
        code: SessionTrajectoryErrorCode,
        *,
        session_id: str,
        parent_session_id: str | None = None,
        terminal_code: TerminalSessionEvidenceErrorCode | None = None,
        limit: int | None = None,
        observed: int | None = None,
    ) -> None:
        if not isinstance(code, SessionTrajectoryErrorCode):
            raise TypeError("code must be a SessionTrajectoryErrorCode.")
        if terminal_code is not None and not isinstance(
            terminal_code, TerminalSessionEvidenceErrorCode
        ):
            raise TypeError("terminal_code must be a TerminalSessionEvidenceErrorCode or None.")
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("limit must be a non-negative integer or None.")
        if observed is not None and (type(observed) is not int or observed < 0):
            raise ValueError("observed must be a non-negative integer or None.")
        self.code = code
        self.session_id = require_clean_nonblank(session_id, "session_id")
        self.parent_session_id = (
            None
            if parent_session_id is None
            else require_clean_nonblank(parent_session_id, "parent_session_id")
        )
        self.terminal_code = terminal_code
        self.limit = limit
        self.observed = observed
        detail = _SESSION_TRAJECTORY_ERROR_MESSAGES[code]
        if terminal_code is not None:
            detail = f"{detail} Terminal reason: {terminal_code.value}."
        if limit is not None:
            detail = f"{detail} Limit: {limit}."
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class _ChildOrigin:
    node: SessionLineageNode
    origin: SessionLineageOrigin

    @property
    def fingerprint(self) -> tuple[str, str, datetime, int, str, str]:
        return (
            self.node.id,
            self.node.parent_session_id,
            self.node.created_at,
            self.origin.sequence,
            self.origin.event_id,
            str(self.origin.event_type),
        )


@dataclass(frozen=True, slots=True)
class _ChildCandidates:
    """One parent-local child enumeration and its completeness state."""

    values: tuple[Session | SessionLineageNode, ...]
    incomplete: bool = False


@dataclass(frozen=True, slots=True)
class _ChildTrajectoryLoad:
    """Result of resolving one best-effort child candidate."""

    trajectory: Trajectory | None = None
    excluded_after_terminal: bool = False


@dataclass(slots=True)
class _CaptureState:
    bounds: SessionTrajectoryBounds
    strict: bool
    retained_session_ids: set[str] = field(default_factory=set)
    lineage_candidate_parents: dict[str, str] = field(default_factory=dict)
    evidence_by_session_id: dict[str, TerminalSessionEvidence] = field(default_factory=dict)
    evidence_limits_by_session_id: dict[str, TerminalSessionEvidenceLimits] = field(
        default_factory=dict
    )
    parent_closures: dict[
        str,
        tuple[int, tuple[tuple[str, str, datetime, int, str, str], ...]],
    ] = field(default_factory=dict)
    event_count: int = 0
    transcript_count: int = 0
    total_bytes: int = 0

    def ensure_retained_session_capacity(
        self,
        session_id: str,
        *,
        parent_session_id: str | None = None,
    ) -> None:
        if session_id in self.retained_session_ids:
            return
        observed = len(self.retained_session_ids) + 1
        if observed > self.bounds.max_sessions:
            raise SessionTrajectoryError(
                SessionTrajectoryErrorCode.SESSION_LIMIT_EXCEEDED,
                session_id=session_id,
                parent_session_id=parent_session_id,
                limit=self.bounds.max_sessions,
                observed=observed,
            )

    def observe_lineage_candidate(
        self,
        session_id: str,
        *,
        parent_session_id: str,
    ) -> None:
        prior_parent = self.lineage_candidate_parents.get(session_id)
        if prior_parent is not None:
            if prior_parent != parent_session_id:
                raise SessionTrajectoryError(
                    SessionTrajectoryErrorCode.PARENT_CONTRADICTION,
                    session_id=session_id,
                    parent_session_id=parent_session_id,
                )
            return
        observed = len(self.lineage_candidate_parents) + 1
        if observed > _SESSION_TRAJECTORY_HARD_MAX_LINEAGE_CANDIDATES:
            raise SessionTrajectoryError(
                SessionTrajectoryErrorCode.SESSION_LIMIT_EXCEEDED,
                session_id=session_id,
                parent_session_id=parent_session_id,
                limit=_SESSION_TRAJECTORY_HARD_MAX_LINEAGE_CANDIDATES,
                observed=observed,
            )
        self.lineage_candidate_parents[session_id] = parent_session_id

    def remaining_terminal_limits(
        self,
        session_id: str,
        *,
        enforce_session_capacity: bool = True,
    ) -> TerminalSessionEvidenceLimits:
        if enforce_session_capacity and session_id not in self.evidence_by_session_id:
            self.ensure_retained_session_capacity(session_id)
        remaining_events = self.bounds.max_events - self.event_count
        remaining_transcript = self.bounds.max_transcript_records - self.transcript_count
        remaining_bytes = self.bounds.max_total_bytes - self.total_bytes
        exhausted = next(
            (
                (code, limit, observed)
                for code, limit, observed in (
                    (
                        TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
                        self.bounds.max_events,
                        self.event_count,
                    ),
                    (
                        TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
                        self.bounds.max_total_bytes,
                        self.total_bytes,
                    ),
                )
                if observed >= limit
            ),
            None,
        )
        if exhausted is not None:
            code, limit, observed = exhausted
            raise SessionTrajectoryError(
                SessionTrajectoryErrorCode.TERMINAL_EVIDENCE_REJECTED,
                session_id=session_id,
                terminal_code=code,
                limit=limit,
                observed=observed,
            )
        return TerminalSessionEvidenceLimits(
            max_events=remaining_events,
            max_transcript_records=remaining_transcript,
            max_record_bytes=self.bounds.max_record_bytes,
            max_total_bytes=remaining_bytes,
        )

    def retain(
        self,
        evidence: TerminalSessionEvidence,
        *,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> None:
        session_id = evidence.session.id
        prior = self.evidence_by_session_id.get(session_id)
        if prior is not None:
            if prior != evidence:
                raise SessionTrajectoryError(
                    SessionTrajectoryErrorCode.EVIDENCE_INCONSISTENT,
                    session_id=session_id,
                )
            return
        self.ensure_retained_session_capacity(
            session_id,
            parent_session_id=evidence.session.parent_session_id,
        )
        next_events = self.event_count + evidence.boundary.event_count
        next_transcript = self.transcript_count + evidence.boundary.transcript_count
        next_bytes = self.total_bytes + evidence.boundary.total_bytes
        exceeded = next(
            (
                (code, limit, observed)
                for code, limit, observed in (
                    (
                        TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
                        self.bounds.max_events,
                        next_events,
                    ),
                    (
                        TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED,
                        self.bounds.max_transcript_records,
                        next_transcript,
                    ),
                    (
                        TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                        self.bounds.max_record_bytes,
                        evidence.boundary.largest_record_bytes,
                    ),
                    (
                        TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
                        self.bounds.max_total_bytes,
                        next_bytes,
                    ),
                )
                if observed > limit
            ),
            None,
        )
        if exceeded is not None:
            code, limit, observed = exceeded
            raise SessionTrajectoryError(
                SessionTrajectoryErrorCode.TERMINAL_EVIDENCE_REJECTED,
                session_id=session_id,
                parent_session_id=evidence.session.parent_session_id,
                terminal_code=code,
                limit=limit,
                observed=observed,
            )
        self.evidence_by_session_id[session_id] = evidence
        self.retained_session_ids.add(session_id)
        if limits is not None:
            self.evidence_limits_by_session_id[session_id] = limits.model_copy(deep=True)
        self.event_count = next_events
        self.transcript_count = next_transcript
        self.total_bytes = next_bytes


class _IncompleteFlag:
    """Node-local signal that a best-effort descendant walk was incomplete."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = False


async def _load_terminal_evidence(
    app: CayuApp,
    session_id: str,
    *,
    limits: TerminalSessionEvidenceLimits | None = None,
) -> TerminalSessionEvidence:
    if not app.session_store.supports_terminal_session_evidence:
        raise NotImplementedError
    return await app.session_store.load_terminal_session_evidence(session_id, limits=limits)


def _project_terminal_evidence(
    evidence: TerminalSessionEvidence,
) -> tuple[Session, tuple[Event, ...], tuple[Message, ...], SessionUsageSummary]:
    """Build one trajectory node's assertion substrate from exact terminal evidence."""

    events = tuple(_trajectory_event(record.event) for record in evidence.events)
    transcript = tuple(record.message for record in evidence.transcript)
    usage_summary = session_usage_summary(evidence.session.id, list(events))
    return evidence.session, events, transcript, usage_summary


def _trajectory_event(event: Event) -> Event:
    """Remove runtime-only promotion facts from the serializable trajectory."""

    if SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY not in event.payload:
        return event
    payload = {
        key: value
        for key, value in event.payload.items()
        if key != SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY
    }
    return event.model_copy(update={"payload": payload})


def _trajectory_from_terminal_evidence(
    evidence: TerminalSessionEvidence,
    *,
    children: tuple[Trajectory, ...] = (),
    children_incomplete: bool = False,
    probes: TrajectoryProbes | None = None,
    metadata: dict[str, Any] | None = None,
) -> Trajectory:
    trajectory = _trajectory_node_from_terminal_evidence(
        evidence,
        children=children,
        children_incomplete=children_incomplete,
        probes=probes,
        metadata=metadata,
    )
    try:
        _validate_trajectory_record_contract(trajectory)
    except (TypeError, ValueError) as exc:
        raise SessionTrajectoryError(
            SessionTrajectoryErrorCode.EVIDENCE_INCONSISTENT,
            session_id=evidence.session.id,
            parent_session_id=evidence.session.parent_session_id,
        ) from exc
    if trajectory.initial_input_message_count is not None:
        # Bind only the finalized root. Hashing each subtree while recursive
        # capture unwinds would revisit deep descendants once per ancestor.
        trajectory._promotion_capture_sha256 = _trajectory_promotion_capture_sha256(trajectory)
    return trajectory


def _trajectory_node_from_terminal_evidence(
    evidence: TerminalSessionEvidence,
    *,
    children: tuple[Trajectory, ...] = (),
    children_incomplete: bool = False,
    probes: TrajectoryProbes | None = None,
    metadata: dict[str, Any] | None = None,
) -> Trajectory:
    """Build one validated node and attach already-captured descendants.

    Child subtrees are validated once, from the completed root. Refeeding each
    existing child through Pydantic while recursive capture unwinds would copy and
    revalidate the same descendants once per ancestor, turning a deep but bounded
    session tree into quadratic work.
    """

    session, events, transcript, usage_summary = _project_terminal_evidence(evidence)
    try:
        node = Trajectory(
            session=session,
            events=events,
            transcript=transcript,
            usage_summary=usage_summary,
            final_output=final_output_text(transcript),
            probes=TrajectoryProbes() if probes is None else probes,
            children=(),
            children_incomplete=False,
            metadata={} if metadata is None else metadata,
        )
    except (TypeError, ValueError) as exc:
        raise SessionTrajectoryError(
            SessionTrajectoryErrorCode.EVIDENCE_INCONSISTENT,
            session_id=evidence.session.id,
            parent_session_id=evidence.session.parent_session_id,
        ) from exc
    input_contract = _attested_initial_input_contract(evidence)
    if input_contract is not None:
        node._initial_input_message_start_index = input_contract.message_start_index
        node._initial_input_message_count = input_contract.message_count
        node._initial_input_messages_sha256 = input_contract.messages_sha256
        node._input_redactions_applied = input_contract.redactions_applied
        node._structured_output_requested = input_contract.structured_output_requested
    # `node` and every child were independently validated from exact evidence.
    # The root-level record-contract pass below checks all cross-node invariants.
    return node.model_copy(
        update={
            "children": children,
            "children_incomplete": children_incomplete,
        }
    )


def _attested_initial_input_contract(
    evidence: TerminalSessionEvidence,
) -> SessionInputContractEvidence | None:
    """Return only runtime-owned fresh-input attribution from exact evidence."""

    started = tuple(
        record.event for record in evidence.events if record.event.type == EventType.SESSION_STARTED
    )
    if len(started) != 1:
        return None
    event = started[0]
    raw_contract = event.payload.get(SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY)
    if type(raw_contract) is not str or not event_payload_authority_is_runtime_generated(
        event,
        field_name=SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
        value=raw_contract,
    ):
        return None
    try:
        return parse_session_input_contract_evidence(raw_contract)
    except ValueError:
        return None


async def trajectory_from_session(
    app: CayuApp,
    session_id: str,
    *,
    bounds: SessionTrajectoryBounds | None = None,
) -> Trajectory:
    """Reconstruct one exact, bounded production session tree as a ``Trajectory``.

    The operation reads durable session state only. It never runs application code,
    providers, tools, environments, hooks, recovery, or writes. Children are admitted
    only when their first durable ``session.started`` or ``session.forked`` event is no
    later than their parent's terminal event. Every admitted node must independently
    satisfy the completed/failed terminal-evidence contract.
    """

    if not isinstance(app, CayuApp):
        raise TypeError("app must be a CayuApp.")
    session_id = require_clean_nonblank(session_id, "session_id")
    selected_bounds = _copy_session_trajectory_bounds(bounds)
    if (
        not app.session_store.supports_terminal_session_evidence
        or not app.session_store.supports_session_lineage
    ):
        raise SessionTrajectoryError(
            SessionTrajectoryErrorCode.STORE_UNSUPPORTED,
            session_id=session_id,
        )

    state = _CaptureState(bounds=selected_bounds, strict=True)
    root_limits = state.remaining_terminal_limits(session_id)
    root_evidence = await _load_strict_terminal_evidence(
        app,
        session_id,
        limits=root_limits,
    )
    state.retain(root_evidence, limits=root_limits)
    children = await _build_child_trajectories(
        app,
        session_id,
        visited={session_id},
        parent_terminal_sequence=root_evidence.boundary.terminal_event_sequence,
        state=state,
        depth=1,
    )
    trajectory = _trajectory_from_terminal_evidence(
        root_evidence,
        children=children,
    )
    await _revalidate_strict_capture(app, state)
    return trajectory


def _copy_session_trajectory_bounds(
    bounds: SessionTrajectoryBounds | None,
) -> SessionTrajectoryBounds:
    if bounds is None:
        return SessionTrajectoryBounds()
    if type(bounds) is not SessionTrajectoryBounds:
        raise TypeError("bounds must be a SessionTrajectoryBounds or None.")
    return SessionTrajectoryBounds.model_validate(bounds.model_dump(mode="python"))


def _terminal_trajectory_error(
    session_id: str,
    exc: TerminalSessionEvidenceError,
    *,
    parent_session_id: str | None = None,
) -> SessionTrajectoryError:
    return SessionTrajectoryError(
        SessionTrajectoryErrorCode.TERMINAL_EVIDENCE_REJECTED,
        session_id=session_id,
        parent_session_id=parent_session_id,
        terminal_code=exc.code,
        limit=exc.limit,
        observed=exc.observed,
    )


async def _load_strict_terminal_evidence(
    app: CayuApp,
    session_id: str,
    *,
    limits: TerminalSessionEvidenceLimits,
    parent_session_id: str | None = None,
) -> TerminalSessionEvidence:
    try:
        evidence = await _load_terminal_evidence(app, session_id, limits=limits)
    except TerminalSessionEvidenceError as exc:
        raise _terminal_trajectory_error(
            session_id,
            exc,
            parent_session_id=parent_session_id,
        ) from exc
    except Exception as exc:
        raise SessionTrajectoryError(
            SessionTrajectoryErrorCode.EVIDENCE_READ_FAILED,
            session_id=session_id,
            parent_session_id=parent_session_id,
        ) from exc
    _validate_terminal_origin_lineage(evidence)
    return evidence


def _validate_terminal_origin_lineage(evidence: TerminalSessionEvidence) -> None:
    """Require one origin event whose runtime-owned lineage matches the session."""

    session = evidence.session
    origins = tuple(
        record for record in evidence.events if record.event.type in _ORIGIN_EVENT_TYPES
    )
    if len(origins) != 1:
        raise SessionTrajectoryError(
            SessionTrajectoryErrorCode.ORIGIN_EVIDENCE_REJECTED,
            session_id=session.id,
            parent_session_id=session.parent_session_id,
        )
    origin = origins[0].event
    expected_parent = session.parent_session_id
    origin_parent = origin.payload.get("parent_session_id")
    if expected_parent is None:
        parent_matches = "parent_session_id" not in origin.payload
    else:
        parent_matches = (
            type(origin_parent) is str
            and origin_parent == expected_parent
            and event_payload_authority_is_runtime_generated(
                origin,
                field_name="parent_session_id",
                value=origin_parent,
            )
        )
    source_matches = True
    if origin.type == EventType.SESSION_FORKED:
        origin_source = origin.payload.get("source_session_id")
        source_matches = (
            expected_parent is not None
            and type(origin_source) is str
            and origin_source == expected_parent
            and event_payload_authority_is_runtime_generated(
                origin,
                field_name="source_session_id",
                value=origin_source,
            )
        )
    if not parent_matches or not source_matches:
        raise SessionTrajectoryError(
            SessionTrajectoryErrorCode.ORIGIN_EVIDENCE_REJECTED,
            session_id=session.id,
            parent_session_id=expected_parent,
        )


async def _build_child_trajectories(
    app: CayuApp,
    parent_session_id: str | None,
    *,
    visited: set[str],
    incomplete: _IncompleteFlag | None = None,
    parent_terminal_sequence: int | None = None,
    state: _CaptureState | None = None,
    depth: int = 1,
) -> tuple[Trajectory, ...]:
    """Construct descendants under either best-effort or strict admission policy."""

    if parent_session_id is None:
        return ()
    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be a positive integer.")
    capture = state or _CaptureState(bounds=SessionTrajectoryBounds(), strict=False)
    try:
        candidate_result = await _child_candidates(
            app,
            parent_session_id,
            state=capture,
            parent_terminal_sequence=parent_terminal_sequence,
        )
        candidates = candidate_result.values
        admitted: tuple[Session | _ChildOrigin, ...]
        if capture.strict or (candidates and isinstance(candidates[0], SessionLineageNode)):
            if parent_terminal_sequence is None:
                raise SessionTrajectoryError(
                    SessionTrajectoryErrorCode.EVIDENCE_INCONSISTENT,
                    session_id=parent_session_id,
                )
            origins = tuple(
                _child_origin(candidate)
                for candidate in candidates
                if isinstance(candidate, SessionLineageNode)
            )
            admitted_origins = tuple(
                sorted(
                    (
                        origin
                        for origin in origins
                        if origin.origin.sequence <= parent_terminal_sequence
                    ),
                    key=lambda origin: (origin.origin.sequence, origin.node.id),
                )
            )
            if capture.strict:
                capture.parent_closures[parent_session_id] = (
                    parent_terminal_sequence,
                    tuple(origin.fingerprint for origin in admitted_origins),
                )
            admitted = admitted_origins
        else:
            admitted = tuple(
                candidate for candidate in candidates if isinstance(candidate, Session)
            )
    except SessionTrajectoryError:
        if capture.strict:
            raise
        if incomplete is not None:
            incomplete.value = True
        return ()

    if not capture.strict and candidate_result.incomplete and incomplete is not None:
        incomplete.value = True

    children: list[Trajectory] = []
    for candidate in admitted:
        child_id = candidate.node.id if isinstance(candidate, _ChildOrigin) else candidate.id
        child_depth = depth + 1
        if child_depth > capture.bounds.max_depth:
            error = SessionTrajectoryError(
                SessionTrajectoryErrorCode.DEPTH_LIMIT_EXCEEDED,
                session_id=child_id,
                parent_session_id=parent_session_id,
                limit=capture.bounds.max_depth,
                observed=child_depth,
            )
            if capture.strict:
                raise error
            if incomplete is not None:
                incomplete.value = True
            continue
        if child_id in visited:
            error = SessionTrajectoryError(
                SessionTrajectoryErrorCode.CYCLE_DETECTED,
                session_id=child_id,
                parent_session_id=parent_session_id,
            )
            if capture.strict:
                raise error
            if incomplete is not None:
                incomplete.value = True
            continue
        visited.add(child_id)
        try:
            loaded = await _load_child_trajectory(
                app,
                candidate,
                expected_parent_session_id=parent_session_id,
                parent_terminal_sequence=parent_terminal_sequence,
                visited=visited,
                state=capture,
                depth=child_depth,
            )
        except SessionTrajectoryError:
            if capture.strict:
                raise
            if incomplete is not None:
                incomplete.value = True
            continue
        if loaded.excluded_after_terminal:
            continue
        child = loaded.trajectory
        if child is not None:
            children.append(child)
            if child.children_incomplete and incomplete is not None:
                incomplete.value = True
        elif incomplete is not None:
            incomplete.value = True
    return tuple(children)


async def _child_candidates(
    app: CayuApp,
    parent_session_id: str,
    *,
    state: _CaptureState,
    parent_terminal_sequence: int | None,
) -> _ChildCandidates:
    if state.strict or (
        parent_terminal_sequence is not None and app.session_store.supports_session_lineage
    ):
        return _ChildCandidates(
            values=await _strict_child_nodes(
                app,
                parent_session_id,
                state=state,
            )
        )
    return await _best_effort_child_sessions(app, parent_session_id, state=state)


async def _best_effort_child_sessions(
    app: CayuApp,
    parent_session_id: str,
    *,
    state: _CaptureState,
) -> _ChildCandidates:
    sessions: list[Session] = []
    seen_ids: set[str] = set()
    cursor: str | None = None
    for _ in range(_CHILD_TRAJECTORY_MAX_PAGES):
        try:
            result = await app.session_store.list_sessions(
                SessionQuery(
                    parent_session_id=parent_session_id,
                    limit=_CHILD_TRAJECTORY_PAGE_SIZE,
                    cursor=cursor,
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
        except Exception:
            return _ChildCandidates(values=tuple(sessions), incomplete=True)
        for session in result.sessions:
            if session.parent_session_id != parent_session_id or session.id in seen_ids:
                raise SessionTrajectoryError(
                    SessionTrajectoryErrorCode.PARENT_CONTRADICTION,
                    session_id=session.id,
                    parent_session_id=parent_session_id,
                )
            try:
                state.observe_lineage_candidate(
                    session.id,
                    parent_session_id=parent_session_id,
                )
            except SessionTrajectoryError:
                return _ChildCandidates(values=tuple(sessions), incomplete=True)
            seen_ids.add(session.id)
            sessions.append(session)
        if result.next_cursor is None:
            return _ChildCandidates(values=tuple(sessions))
        if result.next_cursor == cursor:
            break
        cursor = result.next_cursor
    return _ChildCandidates(values=tuple(sessions), incomplete=True)


async def _strict_child_nodes(
    app: CayuApp,
    parent_session_id: str,
    *,
    state: _CaptureState,
) -> tuple[SessionLineageNode, ...]:
    nodes: list[SessionLineageNode] = []
    seen_ids: set[str] = set()
    cursor: str | None = None
    while True:
        try:
            result = await app.session_store.query_session_lineage(
                SessionLineageQuery(
                    parent_session_id=parent_session_id,
                    cursor=cursor,
                    limit=_STRICT_CHILD_PAGE_SIZE,
                )
            )
        except Exception as exc:
            raise SessionTrajectoryError(
                SessionTrajectoryErrorCode.DESCENDANT_ENUMERATION_FAILED,
                session_id=parent_session_id,
            ) from exc
        if result.parent_session_id != parent_session_id:
            raise SessionTrajectoryError(
                SessionTrajectoryErrorCode.DESCENDANT_ENUMERATION_FAILED,
                session_id=parent_session_id,
            )
        for node in result.children:
            if node.parent_session_id != parent_session_id or node.id in seen_ids:
                raise SessionTrajectoryError(
                    SessionTrajectoryErrorCode.PARENT_CONTRADICTION,
                    session_id=node.id,
                    parent_session_id=parent_session_id,
                )
            state.observe_lineage_candidate(
                node.id,
                parent_session_id=parent_session_id,
            )
            seen_ids.add(node.id)
            nodes.append(node)
        if not result.has_more:
            return tuple(nodes)
        if result.next_cursor is None or result.next_cursor == cursor:
            raise SessionTrajectoryError(
                SessionTrajectoryErrorCode.DESCENDANT_ENUMERATION_FAILED,
                session_id=parent_session_id,
            )
        cursor = result.next_cursor


def _child_origin(node: SessionLineageNode) -> _ChildOrigin:
    origins = node.origin_events
    if len(origins) != 1:
        raise SessionTrajectoryError(
            SessionTrajectoryErrorCode.ORIGIN_EVIDENCE_REJECTED,
            session_id=node.id,
            parent_session_id=node.parent_session_id,
        )
    return _ChildOrigin(node=node, origin=origins[0])


async def _load_child_trajectory(
    app: CayuApp,
    candidate: Session | _ChildOrigin,
    *,
    expected_parent_session_id: str,
    parent_terminal_sequence: int | None,
    visited: set[str],
    state: _CaptureState,
    depth: int,
) -> _ChildTrajectoryLoad:
    session_id = candidate.node.id if isinstance(candidate, _ChildOrigin) else candidate.id
    # A custom store without the minimal lineage capability can reveal the
    # origin boundary only through terminal evidence. Do not charge a candidate
    # against the retained-session ceiling until that evidence proves it began
    # before the parent terminated.
    enforce_session_capacity = not (
        not state.strict and isinstance(candidate, Session) and parent_terminal_sequence is not None
    )
    limits = state.remaining_terminal_limits(
        session_id,
        enforce_session_capacity=enforce_session_capacity,
    )
    if state.strict:
        evidence = await _load_strict_terminal_evidence(
            app,
            session_id,
            limits=limits,
            parent_session_id=expected_parent_session_id,
        )
    else:
        try:
            evidence = await _load_terminal_evidence(
                app,
                session_id,
                limits=limits,
            )
        except TerminalSessionEvidenceError as exc:
            if (
                exc.code != TerminalSessionEvidenceErrorCode.SESSION_INTERRUPTED
                or not app.session_store.supports_runner_owned_interrupted_evidence
            ):
                return _ChildTrajectoryLoad()
            try:
                evidence = await app.session_store.load_runner_owned_interrupted_evidence(
                    session_id,
                    expected_parent_session_id=expected_parent_session_id,
                    limits=limits,
                )
            except Exception:
                return _ChildTrajectoryLoad()
        except Exception:
            return _ChildTrajectoryLoad()
    if evidence.session.parent_session_id != expected_parent_session_id:
        if state.strict:
            raise SessionTrajectoryError(
                SessionTrajectoryErrorCode.PARENT_CONTRADICTION,
                session_id=session_id,
                parent_session_id=expected_parent_session_id,
            )
        return _ChildTrajectoryLoad()
    if not state.strict:
        try:
            _validate_terminal_origin_lineage(evidence)
        except SessionTrajectoryError:
            return _ChildTrajectoryLoad()
    origins = tuple(
        record for record in evidence.events if record.event.type in _ORIGIN_EVENT_TYPES
    )
    if isinstance(candidate, _ChildOrigin) and (
        evidence.session.id != candidate.node.id
        or evidence.session.parent_session_id != candidate.node.parent_session_id
        or evidence.session.created_at != candidate.node.created_at
        or origins[0].sequence != candidate.origin.sequence
        or origins[0].event.id != candidate.origin.event_id
        or origins[0].event.type != candidate.origin.event_type
    ):
        raise SessionTrajectoryError(
            SessionTrajectoryErrorCode.CLOSURE_CHANGED,
            session_id=session_id,
            parent_session_id=expected_parent_session_id,
        )
    if parent_terminal_sequence is not None and origins[0].sequence > parent_terminal_sequence:
        return _ChildTrajectoryLoad(excluded_after_terminal=True)
    state.retain(evidence, limits=limits)
    grandchildren_incomplete = _IncompleteFlag()
    grandchildren = await _build_child_trajectories(
        app,
        session_id,
        visited=visited,
        incomplete=grandchildren_incomplete,
        parent_terminal_sequence=evidence.boundary.terminal_event_sequence,
        state=state,
        depth=depth,
    )
    return _ChildTrajectoryLoad(
        trajectory=_trajectory_node_from_terminal_evidence(
            evidence,
            children=grandchildren,
            children_incomplete=grandchildren_incomplete.value,
        )
    )


async def _revalidate_strict_capture(app: CayuApp, state: _CaptureState) -> None:
    revalidation_state = _CaptureState(bounds=state.bounds, strict=True)
    for evidence in state.evidence_by_session_id.values():
        revalidation_state.ensure_retained_session_capacity(
            evidence.session.id,
            parent_session_id=evidence.session.parent_session_id,
        )
        revalidation_state.retained_session_ids.add(evidence.session.id)

    for parent_session_id, (terminal_sequence, expected) in state.parent_closures.items():
        candidates = await _strict_child_nodes(
            app,
            parent_session_id,
            state=revalidation_state,
        )
        origins = tuple(_child_origin(node) for node in candidates)
        observed = tuple(
            origin.fingerprint
            for origin in sorted(origins, key=lambda item: (item.origin.sequence, item.node.id))
            if origin.origin.sequence <= terminal_sequence
        )
        if observed != expected:
            raise SessionTrajectoryError(
                SessionTrajectoryErrorCode.CLOSURE_CHANGED,
                session_id=parent_session_id,
            )

    for session_id, expected_evidence in state.evidence_by_session_id.items():
        original_limits = state.evidence_limits_by_session_id.get(session_id)
        if original_limits is None:
            raise SessionTrajectoryError(
                SessionTrajectoryErrorCode.EVIDENCE_INCONSISTENT,
                session_id=session_id,
                parent_session_id=expected_evidence.session.parent_session_id,
            )
        limits = TerminalSessionEvidenceLimits(
            max_events=expected_evidence.boundary.event_count,
            max_transcript_records=expected_evidence.boundary.transcript_count,
            max_record_bytes=original_limits.max_record_bytes,
            max_total_bytes=original_limits.max_total_bytes,
        )
        observed_evidence = await _load_strict_terminal_evidence(
            app,
            session_id,
            limits=limits,
            parent_session_id=expected_evidence.session.parent_session_id,
        )
        if observed_evidence != expected_evidence:
            raise SessionTrajectoryError(
                SessionTrajectoryErrorCode.CLOSURE_CHANGED,
                session_id=session_id,
                parent_session_id=expected_evidence.session.parent_session_id,
            )


def final_output_text(transcript: Iterable[Message]) -> str:
    """Return the text of the last assistant message in ``transcript``."""

    for message in reversed(tuple(transcript)):
        if message.role != MessageRole.ASSISTANT:
            continue
        text = "".join(part.text for part in message.content if type(part) is TextPart)
        if text:
            return text
    return ""
