"""In-memory transcript operations and indexes with explicit locked composition."""

from __future__ import annotations

import asyncio
import heapq
from bisect import bisect_left
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.messages import Message, MessageRole, detach_message
from cayu.sessions import _checkpoint_preservation as checkpoint_preservation
from cayu.sessions import _checkpoint_publication as checkpoint_publication
from cayu.sessions import _closure_records as closure_records
from cayu.sessions import _invocation_scope as invocation_scope
from cayu.sessions import transcript_input as session_transcript_input
from cayu.sessions import transcript_queries as session_transcript_queries
from cayu.sessions._invocation_scope import (
    INHERIT_INTERACTION,
    InteractionAttribution,
    resolve_interaction_attribution,
)
from cayu.sessions.records import (
    CheckpointTransform,
    Session,
    TranscriptRecord,
)
from cayu.sessions.transcript_input import (
    DeferredInteractionInput,
    copy_transcript_messages,
    require_deferred_initial_transcript_replacement,
)
from cayu.sessions.transcript_queries import (
    LATEST_TRANSCRIPT_TEXT_MAX_CHARS,
    TranscriptPage,
    TranscriptQuery,
    TranscriptSearchHit,
    TranscriptSearchQuery,
    TranscriptSearchResult,
    TranscriptSnapshot,
    copy_transcript_query,
    copy_transcript_search_query,
    decode_transcript_search_cursor,
    encode_transcript_search_cursor,
    filter_transcript_records,
    transcript_search_document,
    transcript_search_document_score,
    transcript_search_hit_from_message,
    transcript_search_position_after_cursor,
    transcript_search_query_document,
)

if TYPE_CHECKING:
    from cayu.sessions.context_views import (
        ParticipantSessionCreationReceipt,
    )


class MemoryTranscriptIndexes(NamedTuple):
    """Current transcript indexes borrowed under the caller's store lock.

    Index helpers neither acquire nor release that lock. Atomic session writers
    can use them inside their existing publication boundary. Reads may populate
    the interaction projection cache while holding the same lock.
    """

    messages: dict[str, list[Message]]
    interaction_ids: dict[str, list[str | None]]
    indices_by_interaction: dict[str, dict[str | None, list[int]]]
    latest_indexes_by_role: dict[str, dict[MessageRole, int]]
    search_documents: dict[str, list[tuple[MessageRole, str]]]
    search_postings: dict[tuple[str, MessageRole], dict[str, list[int]]]


class MemoryTranscriptReadState(NamedTuple):
    """Read state resolved after acquiring the store lock on every operation."""

    indexes: MemoryTranscriptIndexes
    sessions: dict[str, Session]
    deferred_interaction_inputs: dict[str, DeferredInteractionInput]


LockedTranscriptRead = Callable[[], AbstractAsyncContextManager[MemoryTranscriptReadState]]


async def load_deferred_interaction_input(
    read: LockedTranscriptRead,
    session_id: str,
) -> DeferredInteractionInput | None:
    session_id = require_clean_nonblank(session_id, "session_id")
    async with read() as state:
        if session_id not in state.sessions:
            raise KeyError(f"Session not found: {session_id}")
        deferred = state.deferred_interaction_inputs.get(session_id)
        if deferred is None:
            return None
        return DeferredInteractionInput.model_validate(
            deferred.model_dump(mode="python", warnings=False)
        )


async def load_transcript(read: LockedTranscriptRead, session_id: str) -> list[Message]:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")
    async with read() as state:
        if access_bounds is not None:
            access_bounds.require_read(state.sessions.get(session_id))
        if session_id not in state.sessions:
            raise KeyError(f"Session not found: {session_id}")
        return [detach_message(message) for message in state.indexes.messages.get(session_id, [])]


async def load_transcript_snapshot(
    read: LockedTranscriptRead, session_id: str
) -> TranscriptSnapshot:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")
    async with read() as state:
        if access_bounds is not None:
            access_bounds.require_read(state.sessions.get(session_id))
        if session_id not in state.sessions:
            raise KeyError(f"Session not found: {session_id}")
        messages = state.indexes.messages.get(session_id, [])
        interaction_ids = state.indexes.interaction_ids.get(session_id, [])
        return TranscriptSnapshot(
            records=[
                TranscriptRecord(
                    index=index,
                    interaction_id=interaction_ids[index],
                    message=detach_message(message),
                )
                for index, message in enumerate(messages)
            ],
            cursor=len(messages),
        )


async def load_transcript_cursor(read: LockedTranscriptRead, session_id: str) -> int:
    session_id = require_clean_nonblank(session_id, "session_id")
    async with read() as state:
        if session_id not in state.sessions:
            raise KeyError(f"Session not found: {session_id}")
        return len(state.indexes.messages.get(session_id, []))


async def load_latest_transcript_message(
    read: LockedTranscriptRead,
    session_id: str,
    *,
    role: MessageRole,
) -> TranscriptRecord | None:
    session_id = require_clean_nonblank(session_id, "session_id")
    if not isinstance(role, MessageRole):
        raise TypeError("role must be a MessageRole.")
    async with read() as state:
        if session_id not in state.sessions:
            raise KeyError(f"Session not found: {session_id}")
        messages = state.indexes.messages.get(session_id, [])
        interaction_ids = state.indexes.interaction_ids.get(session_id, [])
        index = state.indexes.latest_indexes_by_role.get(session_id, {}).get(role)
        if index is None:
            return None
        return TranscriptRecord(
            index=index,
            interaction_id=interaction_ids[index],
            message=detach_message(messages[index]),
        )


async def load_latest_transcript_text(
    read: LockedTranscriptRead,
    session_id: str,
    *,
    role: MessageRole,
    max_chars: int,
) -> tuple[str, bool] | None:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")
    if not isinstance(role, MessageRole):
        raise TypeError("role must be a MessageRole.")
    if type(max_chars) is not int:
        raise TypeError("max_chars must be an integer.")
    if not 1 <= max_chars <= LATEST_TRANSCRIPT_TEXT_MAX_CHARS:
        raise ValueError(f"max_chars must be between 1 and {LATEST_TRANSCRIPT_TEXT_MAX_CHARS}.")
    async with read() as state:
        if access_bounds is not None:
            access_bounds.require_read(state.sessions.get(session_id))
        if session_id not in state.sessions:
            raise KeyError(f"Session not found: {session_id}")
        index = state.indexes.latest_indexes_by_role.get(session_id, {}).get(role)
        if index is None:
            return None
        return session_transcript_queries._bounded_transcript_message_text(
            state.indexes.messages[session_id][index],
            max_chars=max_chars,
        )


async def query_transcript(read: LockedTranscriptRead, query: TranscriptQuery) -> TranscriptPage:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    query = copy_transcript_query(query)
    async with read() as state:
        if access_bounds is not None:
            access_bounds.require_read(state.sessions.get(query.session_id))
        if query.session_id not in state.sessions:
            raise KeyError(f"Session not found: {query.session_id}")

        messages = state.indexes.messages.get(query.session_id, [])
        interaction_ids = state.indexes.interaction_ids.get(query.session_id, [])
        if query.interaction_id is None:
            candidate_indices: Sequence[int] = range(len(messages))
        else:
            candidate_indices = interaction_projection(state.indexes, query.session_id).get(
                query.interaction_id, ()
            )
        if query.role is not None:
            candidate_indices = [
                index for index in candidate_indices if messages[index].role == query.role
            ]
        total_records = len(candidate_indices)
        page_indices = candidate_indices[query.offset : query.offset + query.limit]
        # Per filter_transcript_records' contract, the include_thinking=False
        # path already isolates — detach only pass-through records.
        records = [
            TranscriptRecord(
                index=index,
                interaction_id=interaction_ids[index],
                message=(
                    detach_message(messages[index]) if query.include_thinking else messages[index]
                ),
            )
            for index in page_indices
        ]
        return TranscriptPage(
            records=filter_transcript_records(records, include_thinking=query.include_thinking),
            total_records=total_records,
        )


async def search_transcript(
    read: LockedTranscriptRead,
    query: TranscriptSearchQuery,
) -> TranscriptSearchResult:
    query = copy_transcript_search_query(query)
    cursor = decode_transcript_search_cursor(query)
    query_document = transcript_search_query_document(query.text)
    query_tokens = tuple(query_document.split())
    role_set = set(query.roles)
    candidate_heap: list[tuple[int, str, int, Message, str | None]] = []
    overflow = False

    async with read() as state:
        selected = frozenset(
            session_id for session_id in query.session_ids if session_id in state.sessions
        )
        posting_sources: list[tuple[str, list[int]]] = []
        posting_heap: list[tuple[str, int, int, int]] = []
        for token in query_tokens:
            for role in query.roles:
                posting_sessions = state.indexes.search_postings.get((token, role), {})
                for session_id in sorted(selected):
                    indexes = posting_sessions.get(session_id)
                    if not indexes:
                        continue
                    source_index = len(posting_sources)
                    posting_sources.append((session_id, indexes))
                    before_index = query.before_transcript_indexes.get(session_id)
                    position = (
                        len(indexes) - 1
                        if before_index is None
                        else bisect_left(indexes, before_index) - 1
                    )
                    if position < 0:
                        continue
                    heapq.heappush(
                        posting_heap,
                        (session_id, -indexes[position], source_index, position),
                    )

        popped = 0
        next_yield = 256
        while posting_heap:
            session_id, negative_index, source_index, position = heapq.heappop(posting_heap)
            candidate_key = (session_id, negative_index)
            same_candidate = [(source_index, position)]
            while posting_heap and posting_heap[0][:2] == candidate_key:
                _, _, duplicate_source, duplicate_position = heapq.heappop(posting_heap)
                same_candidate.append((duplicate_source, duplicate_position))
            for matched_source, matched_position in same_candidate:
                source_session_id, indexes = posting_sources[matched_source]
                next_position = matched_position - 1
                if next_position >= 0:
                    heapq.heappush(
                        posting_heap,
                        (
                            source_session_id,
                            -indexes[next_position],
                            matched_source,
                            next_position,
                        ),
                    )
            popped += len(same_candidate)
            if popped >= next_yield:
                next_yield = popped + 256
                await asyncio.sleep(0)

            transcript_index = -negative_index
            messages = state.indexes.messages.get(session_id, ())
            interaction_ids = state.indexes.interaction_ids.get(session_id, ())
            documents = state.indexes.search_documents.get(session_id, ())
            if not (
                transcript_index < len(messages)
                and len(messages) == len(interaction_ids) == len(documents)
            ):
                raise RuntimeError("In-memory transcript search index is inconsistent.")
            message = messages[transcript_index]
            if message.role not in role_set:
                continue
            indexed_role, document = documents[transcript_index]
            if indexed_role != message.role:
                raise RuntimeError("In-memory transcript search index is inconsistent.")
            score = transcript_search_document_score(document, query_document)
            if score <= 0:  # pragma: no cover - postings establish a token match
                raise RuntimeError("In-memory transcript search posting is inconsistent.")
            if len(candidate_heap) >= query.max_records_scanned:
                overflow = True
                break
            heapq.heappush(
                candidate_heap,
                (
                    -score,
                    session_id,
                    -transcript_index,
                    message,
                    interaction_ids[transcript_index],
                ),
            )

        if overflow:
            return TranscriptSearchResult(
                query=query,
                matched_records_examined=query.max_records_scanned,
                truncated=True,
                coverage_complete=False,
            )

        matched_records_examined = len(candidate_heap)
        hits: list[TranscriptSearchHit] = []
        remaining_bytes = query.max_bytes
        truncated = False
        continuation_available = False
        ranked_examined = 0
        while candidate_heap:
            (
                negative_score,
                session_id,
                negative_transcript_index,
                message,
                interaction_id,
            ) = heapq.heappop(candidate_heap)
            score = -negative_score
            transcript_index = -negative_transcript_index
            ranked_examined += 1
            if ranked_examined % 256 == 0:
                await asyncio.sleep(0)
            if cursor is not None and not transcript_search_position_after_cursor(
                raw_score=score,
                session_id=session_id,
                transcript_index=transcript_index,
                cursor=cursor,
            ):
                continue
            if len(hits) >= query.limit:
                truncated = True
                continuation_available = True
                break
            hit = transcript_search_hit_from_message(
                session_id=session_id,
                transcript_index=transcript_index,
                interaction_id=interaction_id,
                message=message,
                max_text_bytes=remaining_bytes,
                raw_score=float(score),
            )
            if hit is None:
                truncated = True
                continuation_available = bool(hits)
                break
            hits.append(hit)
            remaining_bytes -= len(hit.text.encode("utf-8"))
            if not hit.text_complete:
                truncated = True
                continuation_available = bool(candidate_heap)
                break
            if remaining_bytes == 0:
                truncated = bool(candidate_heap)
                continuation_available = truncated
                break

    next_cursor = (
        encode_transcript_search_cursor(
            query,
            raw_score=int(hits[-1].raw_score or 0),
            session_id=hits[-1].session_id,
            transcript_index=hits[-1].transcript_index,
        )
        if continuation_available and hits
        else None
    )
    return TranscriptSearchResult(
        query=query,
        hits=tuple(hits),
        matched_records_examined=matched_records_examined,
        truncated=truncated,
        coverage_complete=True,
        next_cursor=next_cursor,
    )


def extend_attribution(
    indexes: MemoryTranscriptIndexes,
    session_id: str,
    interaction_ids: list[str | None],
) -> None:
    """Append attribution and maintain a lazily built absolute-index projection."""

    stored = indexes.interaction_ids[session_id]
    start = len(stored)
    stored.extend(interaction_ids)
    messages = indexes.messages[session_id]
    if len(messages) != len(stored):
        raise RuntimeError("In-memory transcript attribution is inconsistent.")
    latest_by_role = indexes.latest_indexes_by_role.setdefault(session_id, {})
    for index in range(start, len(messages)):
        latest_by_role[messages[index].role] = index
    projection = indexes.indices_by_interaction.get(session_id)
    if projection is None:
        return
    for offset, interaction_id in enumerate(interaction_ids):
        projection.setdefault(interaction_id, []).append(start + offset)


def replace_latest_role_indexes(indexes: MemoryTranscriptIndexes, session_id: str) -> None:
    latest_by_role: dict[MessageRole, int] = {}
    for index, message in enumerate(indexes.messages.get(session_id, ())):
        latest_by_role[message.role] = index
    indexes.latest_indexes_by_role[session_id] = latest_by_role


def interaction_projection(
    indexes: MemoryTranscriptIndexes,
    session_id: str,
) -> dict[str | None, list[int]]:
    projection = indexes.indices_by_interaction.get(session_id)
    if projection is not None:
        return projection
    projection = {}
    for index, interaction_id in enumerate(indexes.interaction_ids.get(session_id, ())):
        projection.setdefault(interaction_id, []).append(index)
    indexes.indices_by_interaction[session_id] = projection
    return projection


def remove_search_session(indexes: MemoryTranscriptIndexes, session_id: str) -> None:
    documents = indexes.search_documents.pop(session_id, ())
    posting_keys = {(token, role) for role, document in documents for token in document.split()}
    for posting_key in posting_keys:
        sessions = indexes.search_postings.get(posting_key)
        if sessions is None:
            raise RuntimeError("In-memory transcript search index is inconsistent.")
        sessions.pop(session_id, None)
        if not sessions:
            del indexes.search_postings[posting_key]


def extend_search(
    indexes: MemoryTranscriptIndexes,
    session_id: str,
    messages: Sequence[Message],
) -> None:
    documents = indexes.search_documents.setdefault(session_id, [])
    expected_start = len(indexes.messages.get(session_id, ())) - len(messages)
    if len(documents) != expected_start:
        raise RuntimeError("In-memory transcript search index is inconsistent.")
    for offset, message in enumerate(messages):
        document = (
            transcript_search_document(message)
            if message.role in {MessageRole.USER, MessageRole.ASSISTANT}
            else ""
        )
        transcript_index = expected_start + offset
        documents.append((message.role, document))
        for token in set(document.split()):
            indexes.search_postings.setdefault((token, message.role), {}).setdefault(
                session_id, []
            ).append(transcript_index)


def replace_search_session(
    indexes: MemoryTranscriptIndexes,
    session_id: str,
    messages: Sequence[Message],
) -> None:
    remove_search_session(indexes, session_id)
    indexes.search_documents[session_id] = []
    extend_search(indexes, session_id, messages)


class RegisterPrivateAlias(Protocol):
    """Register transcript interaction authority inside the current lock scope."""

    def __call__(
        self, private_value: str, *, field_name: str, scope_session_id: str | None = None
    ) -> None: ...


class MemoryTranscriptWriteState(NamedTuple):
    """Transcript publication state and capabilities sharing one store lock.

    Resolve every reference after locking. Checkpoint and alias capabilities
    must operate in this same scope without acquiring the lock again.
    """

    indexes: MemoryTranscriptIndexes
    sessions: dict[str, Session]
    deferred_interaction_inputs: dict[str, DeferredInteractionInput]
    checkpoints: dict[str, dict[str, Any]]
    session_closure_progress: dict[tuple[str, str], dict[str, Any]]
    participant_session_receipts: dict[str, ParticipantSessionCreationReceipt]
    ownership_clock: Callable[[], datetime]
    register_private_alias: RegisterPrivateAlias
    store_checkpoint: Callable[[str, dict[str, Any]], None]
    refresh_queued_receipts: Callable[[str, dict[str, Any] | None], None]


LockedTranscriptWrite = Callable[[], AbstractAsyncContextManager[MemoryTranscriptWriteState]]


async def append_transcript_messages(
    write: LockedTranscriptWrite,
    session_id: str,
    messages: list[Message],
    *,
    interaction_id: InteractionAttribution = INHERIT_INTERACTION,
) -> None:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = resolve_interaction_attribution(session_id, interaction_id)
    copied_messages = copy_transcript_messages(messages)
    async with write() as state:
        session = state.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        invocation_scope._assert_session_run_epoch(session_id, session)
        if not copied_messages:
            return
        for owner in state.session_closure_progress.values():
            closure_records._check_closure_lineage_owner(owner, (session_id,))
        if interaction_id is not None:
            state.register_private_alias(
                interaction_id,
                field_name="interaction_id",
                scope_session_id=session_id,
            )
        state.indexes.messages[session_id].extend(copied_messages)
        extend_search(state.indexes, session_id, copied_messages)
        extend_attribution(state.indexes, session_id, [interaction_id] * len(copied_messages))
        state.sessions[session_id] = session.model_copy(
            update={"last_activity_at": state.ownership_clock()}
        )


async def replace_initial_transcript_messages(
    write: LockedTranscriptWrite,
    session_id: str,
    expected_messages: list[Message],
    replacement_messages: list[Message],
    *,
    interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    checkpoint_transform: CheckpointTransform | None = None,
    runtime_suffix_count: int = 0,
) -> None:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = resolve_interaction_attribution(session_id, interaction_id)
    if interaction_id is None:
        raise ValueError("Initial transcript publication requires an interaction identity.")
    expected = copy_transcript_messages(expected_messages)
    replacement = copy_transcript_messages(replacement_messages)
    async with write() as state:
        session = state.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        invocation_scope._assert_session_run_epoch(session_id, session)
        deferred = state.deferred_interaction_inputs.get(session_id)
        if deferred is None or deferred.interaction_id != interaction_id:
            raise RuntimeError("Deferred interaction input changed before finalization.")
        for owner in state.session_closure_progress.values():
            closure_records._check_closure_lineage_owner(owner, (session_id,))
        require_deferred_initial_transcript_replacement(
            deferred,
            expected_messages=expected,
            replacement_messages=replacement,
        )
        state.register_private_alias(
            interaction_id,
            field_name="interaction_id",
            scope_session_id=session_id,
        )
        if state.indexes.messages.get(session_id):
            from cayu.sessions._participant_execution_identity import (
                require_initial_execution_input,
            )

            require_initial_execution_input(
                session,
                state.participant_session_receipts.get(session_id),
                state.indexes.messages[session_id],
                expected,
            )
        prefix_count = session_transcript_input._initial_transcript_prefix_count(
            expected,
            replacement,
            runtime_suffix_count=runtime_suffix_count,
        )
        current_checkpoint = state.checkpoints.get(session_id)
        if checkpoint_transform is not None:
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                checkpoint_preservation._copy_checkpoint_for_transform(
                    current_checkpoint,
                    session_id=session_id,
                ),
            )
            if transformed is not None:
                current_checkpoint = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                    current_checkpoint,
                    transformed,
                    session_id=session_id,
                )
        checkpoint = checkpoint_publication._checkpoint_after_initial_transcript_publication(
            current_checkpoint,
            interaction_id=interaction_id,
        )
        state.indexes.messages[session_id] = replacement
        replace_latest_role_indexes(state.indexes, session_id)
        replace_search_session(state.indexes, session_id, replacement)
        state.indexes.interaction_ids[session_id] = [None] * prefix_count + [interaction_id] * (
            len(expected) + runtime_suffix_count
        )
        state.indexes.indices_by_interaction.pop(session_id, None)
        state.deferred_interaction_inputs.pop(session_id, None)
        if checkpoint is None:
            state.refresh_queued_receipts(session_id, None)
            state.checkpoints.pop(session_id, None)
        else:
            state.store_checkpoint(session_id, checkpoint)
        state.sessions[session_id] = session.model_copy(
            update={"last_activity_at": state.ownership_clock()}
        )


async def materialize_deferred_interaction_input(
    write: LockedTranscriptWrite,
    session_id: str,
    *,
    interaction_id: InteractionAttribution = INHERIT_INTERACTION,
) -> bool:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = resolve_interaction_attribution(session_id, interaction_id)
    async with write() as state:
        session = state.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        invocation_scope._assert_session_run_epoch(session_id, session)
        deferred = state.deferred_interaction_inputs.get(session_id)
        if deferred is None:
            return False
        if deferred.interaction_id != interaction_id:
            raise RuntimeError("Deferred interaction input belongs to another interaction.")
        for owner in state.session_closure_progress.values():
            closure_records._check_closure_lineage_owner(owner, (session_id,))
        messages = deferred.source_messages
        state.indexes.messages[session_id].extend(messages)
        extend_search(state.indexes, session_id, messages)
        extend_attribution(
            state.indexes,
            session_id,
            [interaction_id] * len(messages),
        )
        state.deferred_interaction_inputs.pop(session_id, None)
        state.sessions[session_id] = session.model_copy(
            update={"last_activity_at": state.ownership_clock()}
        )
        return True


async def append_transcript_messages_and_transform_checkpoint(
    write: LockedTranscriptWrite,
    session_id: str,
    messages: list[Message],
    checkpoint_transform: CheckpointTransform,
    *,
    interaction_id: InteractionAttribution = INHERIT_INTERACTION,
) -> None:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = resolve_interaction_attribution(session_id, interaction_id)
    copied_messages = copy_transcript_messages(messages)
    if checkpoint_transform is None:
        raise TypeError("checkpoint_transform is required.")
    async with write() as state:
        session = state.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        invocation_scope._assert_session_run_epoch(session_id, session)
        for owner in state.session_closure_progress.values():
            closure_records._check_closure_lineage_owner(owner, (session_id,))
        current = state.checkpoints.get(session_id)
        transformed = checkpoint_transform(
            session.model_copy(deep=True),
            checkpoint_preservation._copy_checkpoint_for_transform(current, session_id=session_id),
        )
        if transformed is None:
            raise ValueError("Checkpoint transform must return a checkpoint.")
        copied_checkpoint = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
            current,
            transformed,
            session_id=session_id,
        )
        if copied_messages:
            if interaction_id is not None:
                state.register_private_alias(
                    interaction_id,
                    field_name="interaction_id",
                    scope_session_id=session_id,
                )
            state.indexes.messages[session_id].extend(copied_messages)
            extend_search(state.indexes, session_id, copied_messages)
            extend_attribution(state.indexes, session_id, [interaction_id] * len(copied_messages))
        state.store_checkpoint(session_id, copied_checkpoint)
        state.sessions[session_id] = session.model_copy(
            update={"last_activity_at": state.ownership_clock()}
        )
