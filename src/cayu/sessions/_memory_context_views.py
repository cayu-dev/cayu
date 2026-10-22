"""In-memory context-view operations over explicit locked state.

Factories must bind the current indexes after acquiring their lock. Source
validation callbacks run synchronously under that same lock.
"""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any, NamedTuple

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import EventType
from cayu.messages import Message, detach_message
from cayu.sessions import _closure_records as closure_records
from cayu.sessions import _runtime_publication as publication_contracts
from cayu.sessions import _runtime_publication_rules as publication_rules
from cayu.sessions._model_stage import MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY
from cayu.sessions._runtime_publication import RuntimePublicationReceipt
from cayu.sessions.messaging import SessionMessageDeliveryMode, SessionQueuedMessage
from cayu.sessions.records import EventRecord, Session, TranscriptRecord, copy_session

if TYPE_CHECKING:
    from cayu.collaboration.participants import ParticipantRef as _ParticipantRef
    from cayu.sessions._context_selection_fence import ContextViewSelectionDecision
    from cayu.sessions._context_view_source import CompletedTurnSnapshot
    from cayu.sessions.context_views import ContextViewLifecycleEvent as _ContextViewLifecycleEvent
    from cayu.sessions.context_views import ContextViewManifest as _ContextViewManifest
    from cayu.sessions.context_views import ContextViewReadback as _ContextViewReadback
    from cayu.sessions.context_views import (
        ContextViewSelectionReceipt as _ContextViewSelectionReceipt,
    )
    from cayu.sessions.context_views import ParticipantSessionBinding


class MemoryContextViewSourceState(NamedTuple):
    """Current indexes resolved under the caller's lock for one operation."""

    checkpoints: dict[str, dict[str, Any]]
    event_records_by_id: dict[tuple[str, str], EventRecord]
    participant_session_bindings: dict[str, ParticipantSessionBinding]
    pending_session_messages: dict[
        tuple[str, SessionMessageDeliveryMode], deque[SessionQueuedMessage]
    ]
    session_closure_progress: closure_records.ClosureProgressIndex
    session_event_records: dict[str, list[EventRecord]]
    session_operation_records: dict[str, dict[str, dict[str, Any]]]
    sessions: dict[str, Session]
    transcript_interaction_ids: dict[str, list[str | None]]
    transcripts: dict[str, list[Message]]
    validate_publication_material: Callable[[RuntimePublicationReceipt], None]


LockedContextViewSource = Callable[[], AbstractAsyncContextManager[MemoryContextViewSourceState]]


class MemoryContextSelectionSnapshot(NamedTuple):
    """Selection evidence to restore together on rejected admission."""

    receipts: dict[str, _ContextViewSelectionReceipt]
    commitments: dict[str, str]
    events: dict[str, _ContextViewLifecycleEvent]


class MemoryContextViewState(NamedTuple):
    """Current indexes resolved under the caller's lock for one operation."""

    context_lifecycle_events: dict[str, _ContextViewLifecycleEvent]
    context_ownership_operation_commitments: dict[str, str]
    context_ownership_operation_receipts: dict[str, _ContextViewSelectionReceipt]
    context_selection_exclusions: dict[str, ContextViewSelectionDecision]
    context_selection_receipts: dict[str, _ContextViewSelectionReceipt]
    context_selection_request_commitments: dict[str, str]
    ownership_clock: Callable[[], datetime]
    participant_session_bindings: dict[str, ParticipantSessionBinding]
    restore_selection: Callable[[MemoryContextSelectionSnapshot], None]
    context_publication_keys: dict[str, str]
    context_views: dict[str, _ContextViewManifest]
    sessions: dict[str, Session]


LockedContextViews = Callable[[], AbstractAsyncContextManager[MemoryContextViewState]]


async def capture_completed_turn_snapshot(
    locked: LockedContextViewSource, session_id: str
) -> CompletedTurnSnapshot:
    from cayu.sessions._context_view_source import (
        CompletedTurnSnapshot,
        capture_source,
        closed_round_publication_id,
        completed_boundary,
        publication_frontier,
    )

    async with locked() as state:
        session = state.sessions.get(session_id)
        binding = state.participant_session_bindings.get(session_id)
        checkpoint = state.checkpoints.get(session_id)
        pointer = completed_boundary(session, binding, checkpoint)
        assert session is not None and binding is not None
        tool_receipt = None
        if pointer.tool_round_id is not None:
            publication_id = f"tool-round:{pointer.tool_round_id}"
            key = publication_contracts._runtime_publication_storage_key(publication_id)
            record = state.session_operation_records.get(session_id, {}).get(key)
            if record is None:
                from itertools import islice

                closures = tuple(
                    islice(
                        (
                            item.event
                            for item in reversed(state.session_event_records.get(session_id, ()))
                            if item.event.type is EventType.SESSION_CHECKPOINTED
                            and item.event.payload.get("tool_round_id") == pointer.tool_round_id
                            and (
                                item.event.payload.get("cleared") is True
                                or item.event.payload.get("transition") == "answered"
                            )
                        ),
                        2,
                    )
                )
                publication_id = closed_round_publication_id(pointer, closures)
                key = publication_contracts._runtime_publication_storage_key(publication_id)
                record = state.session_operation_records.get(session_id, {}).get(key)
            if record is not None:
                tool_receipt = publication_rules._reconstruct_runtime_publication_receipt(
                    record,
                    storage_key=key,
                    session_id=session_id,
                    publication_id=publication_id,
                )
            publication_frontier(pointer, tool_receipt)
            assert tool_receipt is not None
            state.validate_publication_material(tool_receipt)
        end = publication_frontier(pointer, tool_receipt)
        event = state.event_records_by_id.get((session_id, pointer.completion_event_id))
        messages = state.transcripts.get(session_id, [])
        interactions = state.transcript_interaction_ids.get(session_id, [])
        records = tuple(
            TranscriptRecord(
                index=index,
                interaction_id=interactions[index],
                message=detach_message(messages[index]),
            )
            for index in range(
                pointer.source_transcript_cursor,
                min(end, len(messages)),
            )
        )
        publication = capture_source(
            session,
            binding,
            checkpoint,
            pointer,
            None if event is None else event.event,
            records,
            tool_receipt,
        )
        return CompletedTurnSnapshot(
            publication=publication,
            current_session=copy_session(session),
            checkpoint=deepcopy(checkpoint),
            has_queued_input=any(
                (session_id, mode) in state.pending_session_messages
                for mode in SessionMessageDeliveryMode
            ),
            has_closure_owner=state.session_closure_progress.owns(session_id),
            has_active_model_stage=MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY
            in state.session_operation_records.get(session_id, {}),
            current_transcript_cursor=len(messages),
        )


async def publish_context_view(
    locked: LockedContextViews, manifest: _ContextViewManifest, *, publication_key: str
) -> _ContextViewManifest:
    from cayu.sessions.context_views import (
        CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER,
    )
    from cayu.sessions.context_views import (
        ContextViewManifest as ContextViewManifestType,
    )

    if type(manifest) is not ContextViewManifestType or type(publication_key) is not str:
        raise TypeError("Context-view publication requires typed manifest and key.")
    if len(publication_key.encode("utf-8")) > 512:
        raise ValueError("publication_key must be at most 512 UTF-8 bytes.")
    publication_key = require_clean_nonblank(publication_key, "publication_key")
    async with locked() as state:
        existing_id = state.context_publication_keys.get(publication_key)
        if existing_id is not None:
            existing = state.context_views[existing_id]
            if existing != manifest:
                raise ValueError("Context-view publication key conflicts with its manifest.")
            return existing.model_copy(deep=True)
        if manifest.view_id in state.context_views:
            raise ValueError("Context-view ID is already bound to another manifest.")
        source = state.sessions.get(manifest.source_session_id)
        if source is None or source.instance_id != manifest.source_session_instance_id:
            raise LookupError("The source session incarnation is unavailable.")
        publication_count = sum(
            view.source_owner == manifest.source_owner for view in state.context_views.values()
        )
        if publication_count >= CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER:
            raise OverflowError("Context-view publication quota exceeded for the owner.")
        state.context_views[manifest.view_id] = manifest.model_copy(deep=True)
        state.context_publication_keys[publication_key] = manifest.view_id
        return manifest.model_copy(deep=True)


async def lookup_context_view_publication(
    locked: LockedContextViews, publication_key: str
) -> _ContextViewManifest | None:
    from cayu.sessions.context_views import ContextViewManifest as ContextViewManifestType

    if type(publication_key) is not str or not publication_key.strip():
        raise ValueError("Context-view publication keys must be non-blank strings.")
    if len(publication_key.encode("utf-8")) > 512:
        raise ValueError("publication_key must be at most 512 UTF-8 bytes.")
    async with locked() as state:
        view_id = state.context_publication_keys.get(publication_key)
        if view_id is None:
            return None
        manifest = state.context_views.get(view_id)
        if manifest is None or type(manifest) is not ContextViewManifestType:
            raise RuntimeError("Context-view publication index is inconsistent.")
        return manifest.model_copy(deep=True)


async def read_context_view(
    locked: LockedContextViews, view_id: str, *, source_session_id: str
) -> _ContextViewReadback:
    from cayu.sessions.context_views import ContextViewReadback as ContextViewReadbackType

    async with locked() as state:
        view = state.context_views.get(view_id)
        if view is None or view.source_session_id != source_session_id:
            raise LookupError("Context view is unavailable.")
        return ContextViewReadbackType(view=view.model_copy(deep=True))


@asynccontextmanager
async def _selection_transaction(
    locked: LockedContextViews,
) -> AsyncIterator[MemoryContextViewState]:
    """Roll back selection-side expiry when admission fails, including cancellation.

    Receipt values are replaced, never mutated in place. The restore capability
    must synchronously replace all three current indexes under the same lock.
    Ownership transitions deliberately use the ordinary lock: rejected expired
    ownership requests retain their expiry evidence.
    """
    async with locked() as state:
        snapshot = MemoryContextSelectionSnapshot(
            state.context_selection_receipts.copy(),
            state.context_selection_request_commitments.copy(),
            state.context_lifecycle_events.copy(),
        )
        try:
            yield state
        except BaseException:
            state.restore_selection(snapshot)
            raise


def _require_lifecycle_capacity(
    state: MemoryContextViewState, view_id: str, *, additional_slots: int = 0
) -> None:
    from cayu.sessions.context_views import validate_context_view_lifecycle_capacity

    events = sum(event.view_id == view_id for event in state.context_lifecycle_events.values())
    unsettled = sum(
        receipt.view.view_id == view_id and receipt.state in {"selected", "adopted", "transferred"}
        for receipt in state.context_selection_receipts.values()
    )
    validate_context_view_lifecycle_capacity(events, unsettled, additional_slots=additional_slots)


async def select_context_view(locked: LockedContextViews, request, *, target=None):
    from cayu.sessions.context_views import (
        CONTEXT_VIEW_EXPIRY_BATCH_SIZE,
        CONTEXT_VIEW_MAX_SELECTIONS_PER_OWNER,
        context_view_manifest_bytes,
    )
    from cayu.sessions.context_views import (
        ContextViewSelectionReceipt as ContextViewSelectionReceiptType,
    )
    from cayu.sessions.context_views import (
        ContextViewSelectionRequest as ContextViewSelectionRequestType,
    )

    if type(request) is not ContextViewSelectionRequestType:
        raise TypeError("Context-view selection requires a typed request.")
    request = ContextViewSelectionRequestType.model_validate(request)
    from cayu.sessions.context_views import json_commitment

    request_commitment = json_commitment(
        canonical_bounded_durable_json_bytes(
            request.model_dump(mode="json"),
            "context view selection request",
            max_bytes=256 * 1024,
            max_nodes=8192,
            max_nesting=64,
        ).decode("utf-8"),
        "context view selection request",
    )
    async with _selection_transaction(locked) as state:
        from cayu.sessions._context_selection_fence import require_not_excluded

        require_not_excluded(
            request,
            state.context_selection_exclusions.get(request.selection_key),
            target=target,
        )
        now_ms = int(state.ownership_clock().timestamp() * 1000)
        expired_count = 0
        for selection_key, receipt in sorted(
            state.context_selection_receipts.items(),
            key=lambda item: item[0] != request.selection_key,
        ):
            if expired_count >= CONTEXT_VIEW_EXPIRY_BATCH_SIZE:
                break
            if (
                receipt.owner == request.source_owner
                and receipt.state == "selected"
                and receipt.expires_at_ms <= now_ms
            ):
                _require_lifecycle_capacity(state, receipt.view.view_id)
                expired = receipt.model_copy(update={"state": "expired"}, deep=True)
                state.context_selection_receipts[selection_key] = expired
                from cayu.sessions.context_views import ContextViewLifecycleEvent

                operation_key = f"expiry:{selection_key}:{receipt.ownership_revision}"
                event = ContextViewLifecycleEvent(
                    event_id="sha256:"
                    + sha256(f"context-view-event:{operation_key}".encode()).hexdigest(),
                    operation_key=operation_key,
                    selection_key=selection_key,
                    view_id=receipt.view.view_id,
                    state="expired",
                    owner=expired.owner,
                    owner_participant=expired.owner_participant,
                    pin_commitment=expired.pin_commitment,
                    ownership_revision=expired.ownership_revision,
                )
                state.context_lifecycle_events[event.event_id] = event
                expired_count += 1
        existing = state.context_selection_receipts.get(request.selection_key)
        if existing is not None:
            if state.context_selection_request_commitments.get(request.selection_key) != (
                request_commitment
            ):
                raise ValueError("Context-view selection key conflicts with its request.")
            if existing.state == "selected" and existing.expires_at_ms <= now_ms:
                _require_lifecycle_capacity(state, existing.view.view_id)
                existing = existing.model_copy(update={"state": "expired"}, deep=True)
                state.context_selection_receipts[request.selection_key] = existing
                from cayu.sessions.context_views import ContextViewLifecycleEvent

                operation_key = f"expiry:{request.selection_key}:{existing.ownership_revision}"
                event = ContextViewLifecycleEvent(
                    event_id="sha256:"
                    + sha256(f"context-view-event:{operation_key}".encode()).hexdigest(),
                    operation_key=operation_key,
                    selection_key=request.selection_key,
                    view_id=existing.view.view_id,
                    state="expired",
                    owner=existing.owner,
                    owner_participant=existing.owner_participant,
                    pin_commitment=existing.pin_commitment,
                    ownership_revision=existing.ownership_revision,
                )
                state.context_lifecycle_events[event.event_id] = event
            return existing.model_copy(deep=True)
        source = state.sessions.get(request.source_session_id)
        if source is None or source.instance_id != request.source_session_instance_id:
            raise LookupError("Context-view source session incarnation is unavailable.")
        from cayu.sessions._context_selection_fence import require_selection_source

        require_selection_source(
            target,
            state.participant_session_bindings.get(request.source_session_id),
            now_ms=now_ms,
        )
        candidates = [
            view
            for view in state.context_views.values()
            if view.source_owner == request.source_owner
            and view.source_session_id == request.source_session_id
            and view.source_session_instance_id == request.source_session_instance_id
            and view.projection_schema == request.projection_schema
            and view.extension_set_commitment == request.extension_set_commitment
            and (request.selector != "exact" or view.view_id == request.exact_view_id)
            and (
                request.minimum_transcript_cursor is None
                or view.transcript_cursor >= request.minimum_transcript_cursor
            )
        ]
        if not candidates:
            raise LookupError("No eligible context view is available.")
        if len(candidates) > request.limits.max_views:
            raise OverflowError("Context-view count exceeds the requested limit.")
        selection_count = sum(
            receipt.owner == request.source_owner
            for receipt in state.context_selection_receipts.values()
        )
        if selection_count >= CONTEXT_VIEW_MAX_SELECTIONS_PER_OWNER:
            raise OverflowError("Context-view selection quota exceeded for the owner.")
        active_pins = sum(
            1
            for receipt in state.context_selection_receipts.values()
            if receipt.owner == request.source_owner
            and receipt.state in {"selected", "adopted", "transferred"}
            and (receipt.state != "selected" or receipt.expires_at_ms > now_ms)
        )
        if active_pins >= request.limits.max_pins:
            raise OverflowError("Context-view pin count exceeds the requested limit.")
        selected = max(candidates, key=lambda view: (view.transcript_cursor, view.view_id))
        from cayu.sessions._context_selection_fence import require_selected_participant

        require_selected_participant(target, selected)
        _require_lifecycle_capacity(state, selected.view_id, additional_slots=1)
        selected_bytes = context_view_manifest_bytes(selected)
        if selected_bytes > request.limits.max_view_bytes:
            raise OverflowError("Context-view manifest exceeds the requested byte limit.")
        retained_views = {
            receipt.view.view_id: receipt.view
            for receipt in state.context_selection_receipts.values()
            if receipt.owner == request.source_owner
            and receipt.state in {"selected", "adopted", "transferred"}
            and (receipt.state != "selected" or receipt.expires_at_ms > now_ms)
        }
        retained_views[selected.view_id] = selected
        retained_bytes = sum(context_view_manifest_bytes(view) for view in retained_views.values())
        if retained_bytes > request.limits.max_retained_bytes:
            raise OverflowError("Context-view retained bytes exceed the requested limit.")
        pin_commitment = (
            "sha256:"
            + sha256(f"context-pin:{request.selection_key}:{selected.view_id}".encode()).hexdigest()
        )
        receipt = ContextViewSelectionReceiptType(
            selection_key=request.selection_key,
            view=selected,
            owner=request.source_owner,
            state="selected",
            pin_commitment=pin_commitment,
            owner_participant=selected.participant,
            expires_at_ms=int(
                state.ownership_clock().timestamp() * 1000
                + request.limits.max_lifetime_seconds * 1000
            ),
        )
        state.context_selection_receipts[request.selection_key] = receipt
        state.context_selection_request_commitments[request.selection_key] = request_commitment
        return receipt.model_copy(deep=True)


async def control_selection(
    locked: LockedContextViews, request, *, authority, exclude, target=None, register=False
):
    from cayu.sessions._context_selection_fence import (
        CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER,
        ContextViewSelectionDecision,
        control_transition,
        require_authority,
        snapshot_request,
    )

    require_authority(authority)
    request = snapshot_request(request)
    async with locked() as state:
        existing = _selection_decision(state, request)
        decision = control_transition(
            request, existing, exclude=exclude, target=target, register=register
        )
        if decision == existing:
            return decision
        if (
            existing is None
            and sum(
                value.request.source_owner == request.source_owner
                for value in state.context_selection_exclusions.values()
            )
            >= CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER
        ):
            raise OverflowError("Context-view selection control quota exceeded.")
        state.context_selection_exclusions[request.selection_key] = decision
        return ContextViewSelectionDecision.model_validate(decision)


def _selection_decision(state: MemoryContextViewState, request):
    from cayu.sessions._context_selection_fence import (
        ContextViewSelectionConflict,
        ContextViewSelectionDecision,
        selected_decision,
    )

    excluded = state.context_selection_exclusions.get(request.selection_key)
    receipt = state.context_selection_receipts.get(request.selection_key)
    if excluded is not None:
        excluded = ContextViewSelectionDecision.model_validate(excluded)
        if (
            excluded.request.selection_key != request.selection_key
            or (receipt is not None and excluded.state != "reserved")
            or excluded.state not in {"reserved", "excluded"}
        ):
            raise ValueError("Context-view exclusion indexes conflict.")
        if excluded.request != request:
            raise ContextViewSelectionConflict("Context-view exclusion evidence conflicts.")
        if receipt is None:
            return excluded
    if receipt is None:
        return None
    return selected_decision(
        request,
        receipt,
        state.context_selection_request_commitments.get(request.selection_key),
        excluded,
    )


async def read_retention(locked: LockedContextViews, target):
    from cayu.collaboration._contracts import (
        ExactConflict,
        ExactMatch,
        ExactNotFound,
        ExactUnavailable,
    )
    from cayu.sessions._context_selection_fence import (
        ContextViewRetentionEvidence,
        ContextViewSelectionConflict,
        ContextViewSelectionTarget,
        retention_evidence,
    )

    target = ContextViewSelectionTarget.model_validate(target)
    async with locked() as state:
        try:
            decision = _selection_decision(state, target.request)
            if decision is not None and decision.target != target:
                return ExactConflict()
            result = retention_evidence(
                target,
                decision,
                state.context_selection_receipts.get(target.request.selection_key),
            )
        except ContextViewSelectionConflict:
            return ExactConflict()
        except (TypeError, ValueError):
            return ExactUnavailable()
        return (
            ExactNotFound()
            if result is None
            else ExactMatch[ContextViewRetentionEvidence](receipt=result)
        )


async def read_context_view_selection_decision(locked: LockedContextViews, request):
    from cayu.collaboration._contracts import (
        ExactConflict,
        ExactMatch,
        ExactNotFound,
        ExactUnavailable,
    )
    from cayu.sessions._context_selection_fence import (
        ContextViewSelectionConflict,
        ContextViewSelectionDecision,
        snapshot_request,
    )

    request = snapshot_request(request)
    async with locked() as state:
        try:
            decision = _selection_decision(state, request)
        except ContextViewSelectionConflict:
            return ExactConflict()
        except (TypeError, ValueError):
            return ExactUnavailable()
        return (
            ExactNotFound()
            if decision is None
            else ExactMatch[ContextViewSelectionDecision](receipt=decision)
        )


async def lookup_context_view_selection(locked: LockedContextViews, selection_key: str):
    from cayu.sessions.context_views import ContextViewSelectionReceipt

    async with locked() as state:
        receipt = state.context_selection_receipts.get(selection_key)
        if receipt is None:
            return None
        return ContextViewSelectionReceipt.model_validate(receipt.model_dump(mode="json"))


async def transition_ownership(locked: LockedContextViews, request, *, target=None):
    from cayu.sessions.context_views import (
        ContextViewOwnershipRequest as ContextViewOwnershipRequestType,
    )
    from cayu.sessions.context_views import (
        json_commitment,
    )

    if type(request) is not ContextViewOwnershipRequestType:
        raise TypeError("Context-view ownership requires a typed request.")
    request = ContextViewOwnershipRequestType.model_validate(request)
    request_commitment = json_commitment(
        canonical_bounded_durable_json_bytes(
            request.model_dump(mode="json"),
            "context view ownership request",
            max_bytes=256 * 1024,
            max_nodes=8192,
            max_nesting=64,
        ).decode("utf-8"),
        "context view ownership request",
    )
    async with locked() as state:
        from cayu.sessions._context_selection_fence import require_selection_adoption

        control = state.context_selection_exclusions.get(request.selection_key)
        require_selection_adoption(
            request,
            _selection_decision(state, control.request) if control is not None else None,
            target=target,
        )
        prior_commitment = state.context_ownership_operation_commitments.get(request.operation_key)
        if prior_commitment is not None:
            if prior_commitment != request_commitment:
                raise ValueError("Ownership operation key conflicts with its request.")
            receipt = state.context_ownership_operation_receipts[request.operation_key]
            return receipt.model_copy(deep=True)
        receipt = state.context_selection_receipts.get(request.selection_key)
        if receipt is None or receipt.view.view_id != request.view_id:
            raise LookupError("Context-view selection is unavailable.")
        if receipt.pin_commitment != request.pin_commitment:
            raise ValueError("Context-view pin commitment conflicts with the request.")
        if receipt.state != request.expected_state:
            raise ValueError("Context-view ownership state is stale or already settled.")
        if receipt.ownership_revision != request.expected_revision:
            raise ValueError("Context-view ownership revision is stale.")
        if receipt.owner != request.current_owner:
            raise PermissionError("The current owner does not control this context-view pin.")
        if (
            request.current_participant is not None
            and receipt.owner_participant is not None
            and receipt.owner_participant != request.current_participant
        ):
            raise PermissionError("The current participant does not control this context-view pin.")
        now_ms = int(state.ownership_clock().timestamp() * 1000)
        from cayu.sessions._context_selection_fence import require_adoption_deadline

        require_adoption_deadline(target, now_ms)
        _require_lifecycle_capacity(state, receipt.view.view_id)
        if receipt.state == "selected" and receipt.expires_at_ms <= now_ms:
            expired = receipt.model_copy(update={"state": "expired"}, deep=True)
            state.context_selection_receipts[request.selection_key] = expired
            from cayu.sessions.context_views import ContextViewLifecycleEvent

            operation_key = f"expiry:{request.selection_key}:{receipt.ownership_revision}"
            state.context_lifecycle_events[
                "sha256:" + sha256(f"context-view-event:{operation_key}".encode()).hexdigest()
            ] = ContextViewLifecycleEvent(
                event_id="sha256:"
                + sha256(f"context-view-event:{operation_key}".encode()).hexdigest(),
                operation_key=operation_key,
                selection_key=request.selection_key,
                view_id=receipt.view.view_id,
                state="expired",
                owner=expired.owner,
                owner_participant=expired.owner_participant,
                pin_commitment=expired.pin_commitment,
                ownership_revision=expired.ownership_revision,
            )
            raise LookupError("Context-view selection has expired.")
        if request.operation == "adopt":
            assert request.destination_owner is not None
            next_state = "adopted"
            next_owner = request.destination_owner
        elif request.operation == "transfer":
            assert request.destination_owner is not None
            next_state = "transferred"
            next_owner = request.destination_owner
        else:
            next_state = "released"
            next_owner = receipt.owner
        next_participant = receipt.owner_participant
        if request.operation != "release":
            if request.destination_participant is None:
                if request.destination_owner != receipt.owner:
                    raise PermissionError(
                        "A transfer to another owner requires destination participant evidence."
                    )
            else:
                next_participant = request.destination_participant
        updated = receipt.model_copy(
            update={
                "owner": next_owner,
                "state": next_state,
                "ownership_revision": receipt.ownership_revision + 1,
                "owner_participant": next_participant,
            },
            deep=True,
        )
        _require_lifecycle_capacity(
            state, updated.view.view_id, additional_slots=int(next_state != "released")
        )
        state.context_selection_receipts[request.selection_key] = updated
        state.context_ownership_operation_commitments[request.operation_key] = request_commitment
        state.context_ownership_operation_receipts[request.operation_key] = updated
        from cayu.sessions.context_views import ContextViewLifecycleEvent

        event = ContextViewLifecycleEvent(
            event_id="sha256:"
            + sha256(f"context-view-event:{request.operation_key}".encode()).hexdigest(),
            operation_key=request.operation_key,
            selection_key=request.selection_key,
            view_id=request.view_id,
            state=next_state,
            owner=updated.owner,
            owner_participant=updated.owner_participant,
            pin_commitment=updated.pin_commitment,
            ownership_revision=updated.ownership_revision,
        )
        state.context_lifecycle_events[event.event_id] = event
        return updated.model_copy(deep=True)


async def validate_context_view_source_closure(locked: LockedContextViews, session_id: str) -> None:
    async with locked() as state:
        now_ms = int(state.ownership_clock().timestamp() * 1000)
        for receipt in state.context_selection_receipts.values():
            if (
                receipt.view.source_session_id == session_id
                and receipt.state in {"selected", "adopted", "transferred"}
                and (receipt.state != "selected" or receipt.expires_at_ms > now_ms)
            ):
                raise ValueError("Session has an active context-view retention pin.")


async def validate_context_view_compaction(
    locked: LockedContextViews, session_id: str, expected_transcript_cursor: int
) -> None:
    if type(expected_transcript_cursor) is not int or expected_transcript_cursor < 0:
        raise ValueError("Context-view compaction requires a non-negative transcript cursor.")
    async with locked() as state:
        validate_compaction(
            session_id,
            expected_transcript_cursor,
            ownership_clock=state.ownership_clock,
            receipts=state.context_selection_receipts,
            views=state.context_views,
        )


def validate_compaction(
    session_id: str,
    expected_transcript_cursor: int,
    *,
    ownership_clock: Callable[[], datetime],
    receipts: Mapping[str, _ContextViewSelectionReceipt],
    views: Mapping[str, _ContextViewManifest],
) -> None:
    """Validate retained material while the caller holds the publication lock."""
    from cayu.sessions.context_views import require_independent_context_view_material

    now_ms = int(ownership_clock().timestamp() * 1000)
    for receipt in receipts.values():
        if (
            receipt.view.source_session_id == session_id
            and receipt.state in {"selected", "adopted", "transferred"}
            and (receipt.state != "selected" or receipt.expires_at_ms > now_ms)
            and receipt.view.transcript_cursor <= expected_transcript_cursor
        ):
            manifest = views.get(receipt.view.view_id)
            if manifest is None or manifest != receipt.view:
                raise ValueError("Pinned context-view material is unavailable for compaction.")
            require_independent_context_view_material(manifest)


async def read_context_view_lifecycle_events(
    locked: LockedContextViews,
    view_id: str,
    *,
    limit: int = 256,
    owner_participant: _ParticipantRef | None = None,
) -> tuple[_ContextViewLifecycleEvent, ...]:
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError("Context-view lifecycle event limits must be between 1 and 256.")
    if owner_participant is not None:
        from cayu.collaboration.participants import ParticipantRef

        owner_participant = ParticipantRef.model_validate(owner_participant)
    async with locked() as state:
        events = [
            event
            for event in state.context_lifecycle_events.values()
            if event.view_id == view_id
            and (owner_participant is None or event.owner_participant == owner_participant)
        ]
        events.sort(key=lambda event: (event.ownership_revision, event.event_id))
        return tuple(event.model_copy(deep=True) for event in events[:limit])
