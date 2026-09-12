"""Bounded failed-workflow observations using existing store and lineage contracts."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Awaitable, Callable

from cayu._validation import canonical_durable_json_bytes, compact_json_utf8_size
from cayu.core.events import Event, EventType, event_payload_authority_is_runtime_generated
from cayu.evals.capture_policy import (
    SessionTrajectoryBounds,
    SessionTrajectoryErrorCode,
    WorkflowCaptureDiagnostic,
    WorkflowFailureCapture,
    WorkflowFailureRecordReference,
)
from cayu.evals.trajectory import (
    SessionTrajectoryError,
    _CaptureState,
    _child_origin,
    _strict_child_nodes,
)
from cayu.runtime.app import CayuApp
from cayu.runtime.sessions import (
    EventQueryResultTooLarge,
    EventRecord,
    SessionInspectionIdentity,
    TerminalSessionEvidenceErrorCode,
)
from cayu.runtime.usage import (
    SessionUsageSummary,
    combine_session_usage_summaries,
    count_model_steps_with_usage,
    session_usage_summary,
)
from cayu.workflows import WORKFLOW_ATTEMPT_EVENT_TYPE, WORKFLOW_JOURNAL_PROVIDER


async def capture_failed_workflow(
    app: CayuApp,
    *,
    session_id: str,
    workflow_name: str,
    started: Event | None,
    bounds: SessionTrajectoryBounds,
    load_records: Callable[..., Awaitable[tuple[EventRecord, ...]]],
) -> tuple[WorkflowFailureCapture, SessionUsageSummary | None]:
    """Observe exact record ranges without constructing a completed trajectory.

    One bad/missing child cannot erase already validated sibling activity. The
    current root attempt is checked again before publication. No provider, tool,
    projector, judge, recovery, or checkpoint operation is invoked.
    """
    state = _CaptureState(bounds=bounds, strict=False, fail_closed=True)
    diagnostics: list[WorkflowCaptureDiagnostic] = []
    references: list[WorkflowFailureRecordReference] = []
    summaries: list[SessionUsageSummary] = []
    diagnostics_truncated = False
    event_reads_failed = False
    model_calls = tool_calls = model_calls_with_usage = 0

    def reject(code: SessionTrajectoryErrorCode, rejected_id: str, error=None) -> None:
        nonlocal diagnostics_truncated
        if len(diagnostics) == 16:
            diagnostics_truncated = True
            return
        diagnostics.append(
            WorkflowCaptureDiagnostic(
                stage="execution",
                code=code,
                session_id=rejected_id,
                bounds=bounds,
                terminal_code=getattr(error, "terminal_code", None),
                limit=getattr(error, "limit", None),
                observed_lower_bound=getattr(error, "observed", None),
                consumed_events=state.event_count,
                consumed_bytes=state.total_bytes,
            )
        )

    def unavailable() -> tuple[WorkflowFailureCapture, None]:
        return WorkflowFailureCapture(
            session_id=session_id,
            diagnostics=tuple(diagnostics),
            diagnostics_truncated=diagnostics_truncated,
        ), None

    async def read(current_id: str, workflow: str | None = None):
        nonlocal event_reads_failed
        # Reuse the live workflow record reader, with finite aggregate allowances.
        limits = state.remaining_terminal_limits(current_id)
        try:
            loaded = await load_records(
                app,
                session_id=current_id,
                workflow_name=workflow,
                max_events=limits.max_events,
                max_bytes=limits.max_total_bytes,
                max_record_bytes=limits.max_record_bytes,
            )
            # Charge reads before identity validation, so rejected records cannot
            # give the next child a fresh aggregate allowance.
            state.event_count += len(loaded)
            state.total_bytes += sum(
                compact_json_utf8_size(record.model_dump(mode="json")) for record in loaded
            )
            return loaded
        except Exception as error:
            # A rejected read may already have consumed the remaining allowance.
            # Preserve earlier references, but do not start more descendant reads.
            event_reads_failed = True
            terminal_code = getattr(error, "terminal_code", None)
            if isinstance(error, EventQueryResultTooLarge):
                terminal_code = (
                    TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED
                    if limits.max_record_bytes <= limits.max_total_bytes
                    else TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED
                )
            if terminal_code is not None:
                raise SessionTrajectoryError(
                    SessionTrajectoryErrorCode.TERMINAL_EVIDENCE_REJECTED,
                    session_id=current_id,
                    terminal_code=terminal_code,
                    limit=getattr(error, "limit", None),
                    observed=getattr(error, "observed", None),
                ) from None
            raise

    def retain(session: SessionInspectionIdentity, records: tuple[EventRecord, ...]) -> None:
        nonlocal model_calls, tool_calls, model_calls_with_usage
        if not records:
            raise SessionTrajectoryError(
                SessionTrajectoryErrorCode.EVIDENCE_INCONSISTENT, session_id=session.id
            )
        state.ensure_retained_session_capacity(session.id)
        state.retained_session_ids.add(session.id)
        events = [record.event for record in records]
        model_calls += sum(event.type == EventType.MODEL_STARTED for event in events)
        tool_calls += sum(event.type == EventType.TOOL_CALL_STARTED for event in events)
        model_calls_with_usage += count_model_steps_with_usage(events)
        summaries.append(session_usage_summary(session.id, events))
        references.append(
            WorkflowFailureRecordReference(
                session_id=session.id,
                run_epoch=session.run_epoch,
                first_sequence=records[0].sequence,
                last_sequence=records[-1].sequence,
                first_event_id=records[0].event.id,
                last_event_id=records[-1].event.id,
                record_count=len(records),
                records_sha256=hashlib.sha256(
                    canonical_durable_json_bytes(
                        [record.model_dump(mode="json") for record in records], "failure records"
                    )
                ).hexdigest(),
            )
        )

    if (
        started is None
        or started.session_id != session_id
        or started.workflow_name != workflow_name
    ):
        reject(SessionTrajectoryErrorCode.ORIGIN_EVIDENCE_REJECTED, session_id)
        return unavailable()
    attempt_id = started.payload.get("attempt_id")
    if type(attempt_id) is not str or not attempt_id:
        reject(SessionTrajectoryErrorCode.ORIGIN_EVIDENCE_REJECTED, session_id)
        return unavailable()
    try:
        root = await app.session_store.inspect_identity(session_id)
        if (
            root is None
            or root.id != session_id
            or root.provider_name != WORKFLOW_JOURNAL_PROVIDER
            or root.agent_name != workflow_name
        ):
            raise SessionTrajectoryError(
                SessionTrajectoryErrorCode.ORIGIN_EVIDENCE_REJECTED, session_id=session_id
            )
        records = await read(session_id, workflow_name)
        markers = [record for record in records if record.event.type == WORKFLOW_ATTEMPT_EVENT_TYPE]
        starts = [record for record in records if record.event.id == started.id]
        if (
            not markers
            or markers[-1].event.payload.get("attempt_id") != attempt_id
            or len(starts) != 1
            or starts[0].event != started
            or markers[-1].sequence >= starts[0].sequence
        ):
            raise SessionTrajectoryError(
                SessionTrajectoryErrorCode.CLOSURE_CHANGED, session_id=session_id
            )
        current = tuple(record for record in records if record.sequence >= markers[-1].sequence)
        if any(record.event.payload.get("attempt_id") != attempt_id for record in current):
            raise SessionTrajectoryError(
                SessionTrajectoryErrorCode.ORIGIN_EVIDENCE_REJECTED, session_id=session_id
            )
        retain(root, current)
    except SessionTrajectoryError as exc:
        reject(exc.code, session_id, exc)
        return unavailable()
    except Exception:
        reject(SessionTrajectoryErrorCode.EVIDENCE_READ_FAILED, session_id)
        return unavailable()

    # Only registrations made in this attempt are attributed to this failure.
    registered = {
        record.event.payload["child_session_id"]
        for record in current
        if record.event.type in {EventType.WORKFLOW_STEP_STARTED, EventType.WORKFLOW_STEP_COMPLETED}
        and type(record.event.payload.get("child_session_id")) is str
    }
    pending: deque[tuple[str, int, set[str] | None]] = deque([(session_id, 1, registered)])
    seen = {session_id}
    while pending:
        parent_id, depth, selected = pending.popleft()
        try:
            nodes = await _strict_child_nodes(app, parent_id, state=state)
        except SessionTrajectoryError as exc:
            reject(exc.code, parent_id, exc)
            continue
        if selected is not None:
            missing = selected - {node.id for node in nodes}
            for missing_id in sorted(missing):
                reject(SessionTrajectoryErrorCode.ORIGIN_EVIDENCE_REJECTED, missing_id)
            nodes = tuple(node for node in nodes if node.id in selected)
        for node in nodes:
            if depth >= bounds.max_depth:
                reject(SessionTrajectoryErrorCode.DEPTH_LIMIT_EXCEEDED, node.id)
                continue
            if node.id in seen:
                reject(SessionTrajectoryErrorCode.CYCLE_DETECTED, node.id)
                continue
            seen.add(node.id)
            try:
                origin = _child_origin(node)
                child = await app.session_store.inspect_identity(node.id)
                if (
                    child is None
                    or child.id != node.id
                    or child.parent_session_id != parent_id
                    or child.created_at != node.created_at
                ):
                    raise SessionTrajectoryError(
                        SessionTrajectoryErrorCode.PARENT_CONTRADICTION, session_id=node.id
                    )
                child_records = await read(node.id)
                origins = [
                    record
                    for record in child_records
                    if record.event.type in {EventType.SESSION_STARTED, EventType.SESSION_FORKED}
                ]
                if (
                    len(origins) != 1
                    or origins[0].sequence != origin.origin.sequence
                    or origins[0].event.id != origin.origin.event_id
                    or origins[0].event.payload.get("parent_session_id") != parent_id
                    or not event_payload_authority_is_runtime_generated(
                        origins[0].event, field_name="parent_session_id", value=parent_id
                    )
                ):
                    raise SessionTrajectoryError(
                        SessionTrajectoryErrorCode.ORIGIN_EVIDENCE_REJECTED, session_id=node.id
                    )
                latest = await app.session_store.inspect_identity(node.id)
                if (
                    latest is None
                    or latest.run_epoch != child.run_epoch
                    or latest.parent_session_id != parent_id
                ):
                    raise SessionTrajectoryError(
                        SessionTrajectoryErrorCode.CLOSURE_CHANGED, session_id=node.id
                    )
                retain(child, child_records)
                pending.append((node.id, depth + 1, None))
            except SessionTrajectoryError as exc:
                reject(exc.code, node.id, exc)
            except Exception:
                reject(SessionTrajectoryErrorCode.EVIDENCE_READ_FAILED, node.id)
            if event_reads_failed:
                pending.clear()
                break

    try:
        latest_records = await load_records(
            app,
            session_id=session_id,
            workflow_name=workflow_name,
            max_events=bounds.max_events,
            max_bytes=bounds.max_total_bytes,
            max_record_bytes=bounds.max_record_bytes,
        )
        if latest_records != records:
            reject(SessionTrajectoryErrorCode.CLOSURE_CHANGED, session_id)
            return unavailable()
    except Exception:
        reject(SessionTrajectoryErrorCode.EVIDENCE_READ_FAILED, session_id)
        return unavailable()
    return WorkflowFailureCapture(
        session_id=session_id,
        attempt_id=attempt_id,
        started_event_id=started.id,
        state="partial",
        records=tuple(references),
        model_calls=model_calls,
        tool_calls=tool_calls,
        model_calls_with_usage=model_calls_with_usage,
        usage_basis="observed_usage_records" if model_calls_with_usage else "unavailable",
        diagnostics=tuple(diagnostics),
        diagnostics_truncated=diagnostics_truncated,
    ), combine_session_usage_summaries(session_id, summaries) if model_calls_with_usage else None
