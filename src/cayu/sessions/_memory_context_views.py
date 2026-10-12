"""In-memory context-view operations over explicit locked state.

Factories must bind the current indexes after acquiring their lock. Source
validation callbacks run synchronously under that same lock.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, NamedTuple

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
    from cayu.sessions._context_view_source import CompletedTurnSnapshot
    from cayu.sessions.context_views import ContextViewManifest as _ContextViewManifest
    from cayu.sessions.context_views import ContextViewReadback as _ContextViewReadback
    from cayu.sessions.context_views import ParticipantSessionBinding


class MemoryContextViewSourceState(NamedTuple):
    """Current indexes resolved under the caller's lock for one operation."""

    checkpoints: dict[str, dict[str, Any]]
    event_records_by_id: dict[tuple[str, str], EventRecord]
    participant_session_bindings: dict[str, ParticipantSessionBinding]
    pending_session_messages: dict[
        tuple[str, SessionMessageDeliveryMode], deque[SessionQueuedMessage]
    ]
    session_closure_progress: dict[tuple[str, str], dict[str, Any]]
    session_event_records: dict[str, list[EventRecord]]
    session_operation_records: dict[str, dict[str, dict[str, Any]]]
    sessions: dict[str, Session]
    transcript_interaction_ids: dict[str, list[str | None]]
    transcripts: dict[str, list[Message]]
    validate_publication_material: Callable[[RuntimePublicationReceipt], None]


LockedContextViewSource = Callable[[], AbstractAsyncContextManager[MemoryContextViewSourceState]]


class MemoryContextViewState(NamedTuple):
    """Current indexes resolved under the caller's lock for one operation."""

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
            has_closure_owner=any(
                session_id in closure_records._closure_progress_targets(owner)
                for owner in state.session_closure_progress.values()
            ),
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
