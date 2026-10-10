"""Complete PostgreSQL transcript operations with explicit native capabilities."""

from __future__ import annotations

import asyncio
import heapq
from collections.abc import Awaitable, Callable, Iterable
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Any, Protocol

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.messages import Message, MessageRole
from cayu.sessions._checkpoint_preservation import (
    _checkpoint_transform_result_preserving_completion_result_event_publications,
    _copy_checkpoint_for_transform,
)
from cayu.sessions.base import (
    INHERIT_INTERACTION,
    InteractionAttribution,
    _assert_session_run_epoch,
    _check_closure_lineage_owner,
    _checkpoint_after_initial_transcript_publication,
    resolve_interaction_attribution,
)
from cayu.sessions.records import CheckpointTransform, Session, TranscriptRecord
from cayu.sessions.transcript_input import (
    DeferredInteractionInput,
    _initial_transcript_prefix_count,
    copy_transcript_messages,
    deferred_interaction_input_from_storage_payload,
    require_deferred_initial_transcript_replacement,
)
from cayu.sessions.transcript_queries import (
    LATEST_TRANSCRIPT_TEXT_MAX_CHARS,
    LATEST_TRANSCRIPT_TEXT_MAX_PARTS,
    LATEST_TRANSCRIPT_TEXT_MAX_SOURCE_BYTES,
    TranscriptPage,
    TranscriptQuery,
    TranscriptSearchHit,
    TranscriptSearchQuery,
    TranscriptSearchResult,
    TranscriptSnapshot,
    TranscriptTextReadLimitExceeded,
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
    transcript_search_session_token,
)
from cayu.storage import _postgres_support as pg_support
from cayu.storage._participant_bindings_schema import PARTICIPANT_BINDING_PROJECTION

PostgresConnection = Callable[[], AbstractAsyncContextManager[Any]]
SessionLoader = Callable[[Any, str], Awaitable[Session | None]]


class AuthorityRegistrar(Protocol):
    def __call__(
        self, cur: Any, session_id: str, *, interaction_ids: tuple[str, ...] = ()
    ) -> Awaitable[None]: ...


async def transcript_cursor(cur: Any, session_id: str) -> int:
    """Return the permanent next transcript position, independent of retention."""

    await cur.execute(
        "SELECT transcript_seq FROM cayu_sessions WHERE id = %s",
        (session_id,),
    )
    row = await cur.fetchone()
    if row is None:
        raise KeyError(f"Session not found: {session_id}")
    return int(row[0])


def index_document(session_id: str, message: Message) -> str:
    narrative_document = transcript_search_document(message)
    session_term = transcript_search_session_token(session_id)
    return session_term if not narrative_document else f"{session_term} {narrative_document}"


def _search_expression(query: TranscriptSearchQuery) -> str:
    session_terms = " | ".join(
        transcript_search_session_token(session_id) for session_id in query.session_ids
    )
    text_terms = " | ".join(transcript_search_query_document(query.text).split())
    return f"({session_terms}) & ({text_terms})"


async def append_transcript_messages(
    connect: PostgresConnection,
    session_id: str,
    messages: list[Message],
    *,
    interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    store_now: Callable[[Any], Awaitable[datetime]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    register_authorities: AuthorityRegistrar,
    touch_activity: Callable[[Any, str, datetime], Awaitable[None]],
) -> None:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = resolve_interaction_attribution(session_id, interaction_id)
    copied_messages = copy_transcript_messages(messages)
    async with connect() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM cayu_sessions WHERE id = %s FOR UPDATE",
                (session_id,),
            )
            if await cur.fetchone() is None:
                raise KeyError(f"Session not found: {session_id}")
            if copied_messages:
                for owner in await closure_owners(cur, (session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                await register_authorities(
                    cur,
                    session_id,
                    interaction_ids=(() if interaction_id is None else (interaction_id,)),
                )
                await touch_activity(
                    cur,
                    session_id,
                    await store_now(cur),
                )
                await cur.executemany(
                    """
                    INSERT INTO cayu_transcript_messages
                        (session_id, interaction_id, message,
                         transcript_search_document)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [
                        (
                            session_id,
                            interaction_id,
                            pg_support._dumps(message.model_dump(mode="json")),
                            index_document(session_id, message),
                        )
                        for message in copied_messages
                    ],
                )
        await conn.commit()


async def replace_initial_transcript_messages(
    connect: PostgresConnection,
    session_id: str,
    expected_messages: list[Message],
    replacement_messages: list[Message],
    *,
    interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    checkpoint_transform: CheckpointTransform | None = None,
    runtime_suffix_count: int = 0,
    load_for_update: SessionLoader,
    store_now: Callable[[Any], Awaitable[datetime]],
    load_checkpoint: Callable[[Any, str], Awaitable[dict[str, Any] | None]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    register_authorities: AuthorityRegistrar,
    upsert_checkpoint: Callable[[Any, str, dict[str, Any], datetime], Awaitable[None]],
    touch_activity: Callable[[Any, str, datetime], Awaitable[None]],
) -> None:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = resolve_interaction_attribution(session_id, interaction_id)
    if interaction_id is None:
        raise ValueError("Initial transcript publication requires an interaction identity.")
    expected = copy_transcript_messages(expected_messages)
    replacement = copy_transcript_messages(replacement_messages)
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                session = await load_for_update(cur, session_id)
                if session is None:
                    raise KeyError(f"Session not found: {session_id}")
                updated_at = await store_now(cur)
                _assert_session_run_epoch(session_id, session)
                await cur.execute(
                    "SELECT interaction_id, source_messages "
                    "FROM cayu_deferred_interaction_inputs "
                    "WHERE session_id = %s FOR UPDATE",
                    (session_id,),
                )
                row = await cur.fetchone()
                if row is None or row[0] != interaction_id:
                    raise RuntimeError("Deferred interaction input changed before finalization.")
                for owner in await closure_owners(cur, (session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                stored = deferred_interaction_input_from_storage_payload(
                    row[0],
                    pg_support._json_obj(row[1]),
                )
                require_deferred_initial_transcript_replacement(
                    stored,
                    expected_messages=expected,
                    replacement_messages=replacement,
                )
                await cur.execute(
                    "SELECT 1 FROM cayu_transcript_messages WHERE session_id = %s LIMIT 1",
                    (session_id,),
                )
                if await cur.fetchone() is not None:
                    from cayu.sessions._participant_execution_identity import (
                        require_initial_execution_input,
                    )
                    from cayu.storage._participant_session_records import reconstruct

                    await cur.execute(
                        f"SELECT {PARTICIPANT_BINDING_PROJECTION} FROM cayu_participant_session_bindings WHERE session_id = %s",
                        (session_id,),
                    )
                    binding_row = await cur.fetchone()
                    await cur.execute(
                        "SELECT message FROM cayu_transcript_messages WHERE session_id = %s ORDER BY session_order",
                        (session_id,),
                    )
                    current_rows = await cur.fetchall()
                    require_initial_execution_input(
                        session,
                        None if binding_row is None else reconstruct(binding_row, session),
                        [
                            Message.model_validate(pg_support._json_obj(row[0]))
                            for row in current_rows
                        ],
                        expected,
                    )
                    await cur.execute(
                        "DELETE FROM cayu_transcript_messages WHERE session_id = %s",
                        (session_id,),
                    )
                    await cur.execute(
                        "UPDATE cayu_sessions SET transcript_seq = 0 WHERE id = %s",
                        (session_id,),
                    )
                prefix_count = _initial_transcript_prefix_count(
                    expected,
                    replacement,
                    runtime_suffix_count=runtime_suffix_count,
                )
                current_checkpoint = await load_checkpoint(cur, session_id)
                if checkpoint_transform is not None:
                    transformed = checkpoint_transform(
                        session,
                        _copy_checkpoint_for_transform(
                            current_checkpoint,
                            session_id=session_id,
                        ),
                    )
                    if transformed is not None:
                        current_checkpoint = _checkpoint_transform_result_preserving_completion_result_event_publications(
                            current_checkpoint,
                            transformed,
                            session_id=session_id,
                        )
                checkpoint = _checkpoint_after_initial_transcript_publication(
                    current_checkpoint,
                    interaction_id=interaction_id,
                )
                await register_authorities(
                    cur,
                    session_id,
                    interaction_ids=(interaction_id,),
                )
                await cur.executemany(
                    "INSERT INTO cayu_transcript_messages "
                    "(session_id, interaction_id, message, "
                    "transcript_search_document) VALUES (%s, %s, %s, %s)",
                    [
                        (
                            session_id,
                            None if index < prefix_count else interaction_id,
                            pg_support._dumps(message.model_dump(mode="json")),
                            index_document(session_id, message),
                        )
                        for index, message in enumerate(replacement)
                    ],
                )
                await cur.execute(
                    "DELETE FROM cayu_deferred_interaction_inputs WHERE session_id = %s",
                    (session_id,),
                )
                if checkpoint is None:
                    await cur.execute(
                        "DELETE FROM cayu_checkpoints WHERE session_id = %s",
                        (session_id,),
                    )
                else:
                    await upsert_checkpoint(cur, session_id, checkpoint, updated_at)
                await touch_activity(cur, session_id, updated_at)
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise


async def materialize_deferred_interaction_input(
    connect: PostgresConnection,
    session_id: str,
    *,
    interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    load_for_update: SessionLoader,
    store_now: Callable[[Any], Awaitable[datetime]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    register_authorities: AuthorityRegistrar,
    touch_activity: Callable[[Any, str, datetime], Awaitable[None]],
) -> bool:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = resolve_interaction_attribution(session_id, interaction_id)
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                session = await load_for_update(cur, session_id)
                if session is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, session)
                await cur.execute(
                    "SELECT interaction_id, source_messages "
                    "FROM cayu_deferred_interaction_inputs "
                    "WHERE session_id = %s FOR UPDATE",
                    (session_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    await conn.commit()
                    return False
                if row[0] != interaction_id:
                    raise RuntimeError("Deferred interaction input belongs to another interaction.")
                for owner in await closure_owners(cur, (session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                deferred = deferred_interaction_input_from_storage_payload(
                    row[0],
                    pg_support._json_obj(row[1]),
                )
                messages = deferred.source_messages
                if interaction_id is not None:
                    await register_authorities(
                        cur,
                        session_id,
                        interaction_ids=(interaction_id,),
                    )
                await cur.executemany(
                    "INSERT INTO cayu_transcript_messages "
                    "(session_id, interaction_id, message, "
                    "transcript_search_document) VALUES (%s, %s, %s, %s)",
                    [
                        (
                            session_id,
                            interaction_id,
                            pg_support._dumps(message.model_dump(mode="json")),
                            index_document(session_id, message),
                        )
                        for message in messages
                    ],
                )
                await cur.execute(
                    "DELETE FROM cayu_deferred_interaction_inputs WHERE session_id = %s",
                    (session_id,),
                )
                await touch_activity(
                    cur,
                    session_id,
                    await store_now(cur),
                )
            await conn.commit()
            return True
        except Exception:
            await conn.rollback()
            raise


async def load_deferred_interaction_input(
    connect: PostgresConnection, session_id: str, *, load_session: SessionLoader
) -> DeferredInteractionInput | None:
    session_id = require_clean_nonblank(session_id, "session_id")
    async with connect() as conn, conn.cursor() as cur:
        if await load_session(cur, session_id) is None:
            raise KeyError(f"Session not found: {session_id}")
        await cur.execute(
            "SELECT interaction_id, source_messages "
            "FROM cayu_deferred_interaction_inputs WHERE session_id = %s",
            (session_id,),
        )
        row = await cur.fetchone()
    if row is None:
        return None
    return deferred_interaction_input_from_storage_payload(row[0], pg_support._json_obj(row[1]))


async def append_transcript_messages_and_transform_checkpoint(
    connect: PostgresConnection,
    session_id: str,
    messages: list[Message],
    checkpoint_transform: CheckpointTransform,
    *,
    interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    load_for_update: SessionLoader,
    store_now: Callable[[Any], Awaitable[datetime]],
    load_checkpoint: Callable[[Any, str], Awaitable[dict[str, Any] | None]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    register_authorities: AuthorityRegistrar,
    upsert_checkpoint: Callable[[Any, str, dict[str, Any], datetime], Awaitable[None]],
    touch_activity: Callable[[Any, str, datetime], Awaitable[None]],
) -> None:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = resolve_interaction_attribution(session_id, interaction_id)
    copied_messages = copy_transcript_messages(messages)
    if checkpoint_transform is None:
        raise TypeError("checkpoint_transform is required.")
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                session = await load_for_update(cur, session_id)
                if session is None:
                    raise KeyError(f"Session not found: {session_id}")
                updated_at = await store_now(cur)
                _assert_session_run_epoch(session_id, session)
                for owner in await closure_owners(cur, (session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                current_checkpoint = await load_checkpoint(cur, session_id)
                transformed = checkpoint_transform(
                    session,
                    _copy_checkpoint_for_transform(
                        current_checkpoint,
                        session_id=session_id,
                    ),
                )
                if transformed is None:
                    raise ValueError("Checkpoint transform must return a checkpoint.")
                transformed = (
                    _checkpoint_transform_result_preserving_completion_result_event_publications(
                        current_checkpoint,
                        transformed,
                        session_id=session_id,
                    )
                )
                await touch_activity(cur, session_id, updated_at)
                if copied_messages:
                    await register_authorities(
                        cur,
                        session_id,
                        interaction_ids=(() if interaction_id is None else (interaction_id,)),
                    )
                    await cur.executemany(
                        """
                        INSERT INTO cayu_transcript_messages
                            (session_id, interaction_id, message,
                             transcript_search_document)
                        VALUES (%s, %s, %s, %s)
                        """,
                        [
                            (
                                session_id,
                                interaction_id,
                                pg_support._dumps(message.model_dump(mode="json")),
                                index_document(session_id, message),
                            )
                            for message in copied_messages
                        ],
                    )
                await upsert_checkpoint(cur, session_id, transformed, updated_at)
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise


async def load_transcript(
    connect: PostgresConnection, session_id: str, *, load_session: SessionLoader
) -> list[Message]:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")
    async with connect() as conn, conn.cursor() as cur:
        if access_bounds is not None:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            access_bounds.require_read(await load_session(cur, session_id))
        await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")
        await cur.execute(
            """
            SELECT message
            FROM cayu_transcript_messages
            WHERE session_id = %s
            ORDER BY sequence ASC
            """,
            (session_id,),
        )
        rows = await cur.fetchall()
        return [Message(**pg_support._json_obj(row[0])) for row in rows]


async def load_transcript_snapshot(
    connect: PostgresConnection, session_id: str, *, load_session: SessionLoader
) -> TranscriptSnapshot:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")
    async with connect() as conn, conn.cursor() as cur:
        if access_bounds is not None:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            access_bounds.require_read(await load_session(cur, session_id))
        await cur.execute(
            """
            SELECT session.transcript_seq,
                   transcript.session_order - 1 AS transcript_index,
                   transcript.interaction_id,
                   transcript.message
            FROM cayu_sessions AS session
            LEFT JOIN cayu_transcript_messages AS transcript
              ON transcript.session_id = session.id
            WHERE session.id = %s
            ORDER BY transcript.session_order ASC
            """,
            (session_id,),
        )
        rows = await cur.fetchall()
        if not rows:
            raise KeyError(f"Session not found: {session_id}")
        return TranscriptSnapshot(
            records=[
                TranscriptRecord(
                    index=row[1],
                    interaction_id=row[2],
                    message=Message(**pg_support._json_obj(row[3])),
                )
                for row in rows
                if row[1] is not None
            ],
            cursor=int(rows[0][0]),
        )


async def load_transcript_cursor(connect: PostgresConnection, session_id: str) -> int:
    session_id = require_clean_nonblank(session_id, "session_id")
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT transcript_seq FROM cayu_sessions WHERE id = %s",
            (session_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Session not found: {session_id}")
        return int(row[0])


async def load_latest_transcript_message(
    connect: PostgresConnection, session_id: str, *, role: MessageRole
) -> TranscriptRecord | None:
    session_id = require_clean_nonblank(session_id, "session_id")
    if not isinstance(role, MessageRole):
        raise TypeError("role must be a MessageRole.")
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT session.id,
                   transcript.session_order - 1,
                   transcript.interaction_id,
                   transcript.message
            FROM cayu_sessions AS session
            LEFT JOIN LATERAL (
                SELECT session_order, interaction_id, message
                FROM cayu_transcript_messages
                WHERE session_id = session.id
                  AND message ->> 'role' = %s
                ORDER BY session_order DESC
                LIMIT 1
            ) AS transcript ON TRUE
            WHERE session.id = %s
            """,
            (str(role), session_id),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Session not found: {session_id}")
        if row[1] is None:
            return None
        return TranscriptRecord(
            index=row[1],
            interaction_id=row[2],
            message=Message(**pg_support._json_obj(row[3])),
        )


async def load_latest_transcript_text(
    connect: PostgresConnection,
    session_id: str,
    *,
    role: MessageRole,
    max_chars: int,
    load_session: SessionLoader,
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
    async with connect() as conn, conn.cursor() as cur:
        if access_bounds is not None:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            access_bounds.require_read(await load_session(cur, session_id))
        await cur.execute(
            """
            SELECT session.id,
                   transcript.sequence,
                   pg_column_size(transcript.message)
            FROM cayu_sessions AS session
            LEFT JOIN LATERAL (
                SELECT sequence, message
                FROM cayu_transcript_messages
                WHERE session_id = session.id
                  AND message ->> 'role' = %s
                ORDER BY session_order DESC
                LIMIT 1
            ) AS transcript ON TRUE
            WHERE session.id = %s
            """,
            (str(role), session_id),
        )
        row = await cur.fetchone()
        if row is None:
            raise KeyError(f"Session not found: {session_id}")
        sequence = row[1]
        if sequence is None:
            return None
        if int(row[2]) > LATEST_TRANSCRIPT_TEXT_MAX_SOURCE_BYTES:
            raise TranscriptTextReadLimitExceeded(
                "Transcript message exceeds the bounded serialized-source limit."
            )
        await cur.execute(
            """
            WITH RECURSIVE
            source(message, part_count) AS (
                SELECT message, jsonb_array_length(message -> 'content')
                FROM cayu_transcript_messages
                WHERE sequence = %s
            ),
            prefix(part_index, text_value, part_count) AS (
                SELECT 0, ''::text, part_count
                FROM source
                UNION ALL
                SELECT
                    prefix.part_index + 1,
                    left(
                        prefix.text_value ||
                        CASE
                            WHEN source.message -> 'content' -> prefix.part_index
                                 ->> 'type' = 'text'
                            THEN COALESCE(
                                source.message -> 'content' -> prefix.part_index ->> 'text',
                                ''
                            )
                            ELSE ''
                        END,
                        %s
                    ),
                    prefix.part_count
                FROM prefix
                CROSS JOIN source
                WHERE prefix.part_index < prefix.part_count
                  AND prefix.part_index < %s
                  AND length(prefix.text_value) <= %s
            )
            SELECT text_value, part_index, part_count
            FROM prefix
            ORDER BY part_index DESC
            LIMIT 1
            """,
            (
                int(sequence),
                max_chars + 1,
                LATEST_TRANSCRIPT_TEXT_MAX_PARTS,
                max_chars,
            ),
        )
        projection = await cur.fetchone()
        if projection is None:
            raise TranscriptTextReadLimitExceeded(
                "Transcript message changed during its bounded text projection."
            )
        text_value = str(projection[0])
        if int(projection[1]) < int(projection[2]) and len(text_value) <= max_chars:
            raise TranscriptTextReadLimitExceeded(
                "Transcript message exceeds the bounded content-part inspection limit."
            )
        return text_value[:max_chars], len(text_value) > max_chars


async def load_transcript_window(
    connect: PostgresConnection,
    session_id: str,
    *,
    start_index: int,
    limit: int,
    load_session: SessionLoader,
) -> TranscriptSnapshot:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(start_index) is not int:
        raise TypeError("start_index must be an integer.")
    if not 0 <= start_index <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError("start_index exceeds the durable integer limit.")
    if type(limit) is not int:
        raise TypeError("limit must be an integer.")
    if not 1 <= limit <= 5000:
        raise ValueError("limit must be between 1 and 5000.")

    async with connect() as conn, conn.cursor() as cur:
        if access_bounds is not None:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            access_bounds.require_read(await load_session(cur, session_id))
        await cur.execute(
            """
            SELECT session.transcript_seq,
                   transcript.session_order - 1 AS transcript_index,
                   transcript.interaction_id,
                   transcript.message
            FROM cayu_sessions AS session
            LEFT JOIN cayu_transcript_messages AS transcript
              ON transcript.session_id = session.id
             AND transcript.session_order > %s
            WHERE session.id = %s
            ORDER BY transcript.session_order ASC
            LIMIT %s
            """,
            (start_index, session_id, limit),
        )
        rows = await cur.fetchall()
        if not rows:
            raise KeyError(f"Session not found: {session_id}")
        return TranscriptSnapshot(
            records=[
                TranscriptRecord(
                    index=row[1],
                    interaction_id=row[2],
                    message=Message(**pg_support._json_obj(row[3])),
                )
                for row in rows
                if row[1] is not None
            ],
            cursor=int(rows[0][0]),
        )


async def query_transcript(
    connect: PostgresConnection, query: TranscriptQuery, *, load_session: SessionLoader
) -> TranscriptPage:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    query = copy_transcript_query(query)
    filters: list[str] = []
    filter_params: list[object] = []
    if query.role is not None:
        filters.append("message ->> 'role' = %s")
        filter_params.append(str(query.role))
    if query.interaction_id is not None:
        filters.append("interaction_id = %s")
        filter_params.append(query.interaction_id)
    filter_clause = " AND " + " AND ".join(filters) if filters else ""

    async with connect() as conn, conn.cursor() as cur:
        if access_bounds is not None:
            await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            access_bounds.require_read(await load_session(cur, query.session_id))
        await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (query.session_id,))
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {query.session_id}")

        await cur.execute(
            f"""
            SELECT COUNT(*)
            FROM cayu_transcript_messages
            WHERE session_id = %s
            {filter_clause}
            """,
            [query.session_id, *filter_params],
        )
        total_row = await cur.fetchone()
        total_records = int(total_row[0]) if total_row is not None else 0

        await cur.execute(
            f"""
            SELECT session_order - 1 AS transcript_index, interaction_id, message
            FROM cayu_transcript_messages
            WHERE session_id = %s
            {filter_clause}
            ORDER BY session_order ASC
            LIMIT %s OFFSET %s
            """,
            [query.session_id, *filter_params, query.limit, query.offset],
        )
        rows = await cur.fetchall()
        records = [
            TranscriptRecord(
                index=row[0],
                interaction_id=row[1],
                message=Message(**pg_support._json_obj(row[2])),
            )
            for row in rows
        ]
        return TranscriptPage(
            records=filter_transcript_records(records, include_thinking=query.include_thinking),
            total_records=total_records,
        )


async def search_transcript(
    connect: PostgresConnection, query: TranscriptSearchQuery
) -> TranscriptSearchResult:
    query = copy_transcript_search_query(query)
    cursor = decode_transcript_search_cursor(query)
    query_document = transcript_search_query_document(query.text)
    before_filter = "".join(
        " AND (transcript.session_id <> %s OR transcript.session_order <= %s)"
        for _ in query.before_transcript_indexes
    )
    before_params = [
        value
        for session_id, before_index in query.before_transcript_indexes.items()
        for value in (session_id, before_index)
    ]
    fetch_limit = query.max_records_scanned + 1

    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            f"""
            WITH search_query AS (
                SELECT to_tsquery('simple'::regconfig, %s) AS value
            )
            SELECT
                transcript.session_id,
                transcript.session_order - 1 AS transcript_index,
                transcript.interaction_id,
                transcript.message,
                transcript.transcript_search_document
            FROM cayu_transcript_messages AS transcript
            CROSS JOIN search_query
            WHERE transcript.session_id = ANY(%s)
              AND transcript.message ->> 'role' IN ('user', 'assistant')
              AND transcript.message ->> 'role' = ANY(%s)
              {before_filter}
              AND to_tsvector(
                    'simple'::regconfig,
                    transcript.transcript_search_document
                  ) @@ search_query.value
            LIMIT %s
            """,
            [
                _search_expression(query),
                list(query.session_ids),
                [str(role) for role in query.roles],
                *before_params,
                fetch_limit,
            ],
        )
        rows = await cur.fetchall()

    if len(rows) > query.max_records_scanned:
        return TranscriptSearchResult(
            query=query,
            matched_records_examined=query.max_records_scanned,
            truncated=True,
            coverage_complete=False,
        )

    candidate_heap: list[tuple[int, str, int, Any, Message]] = []
    for row_number, row in enumerate(rows, start=1):
        message = Message(**pg_support._json_obj(row[3]))
        document = transcript_search_document(message)
        if row[4] != index_document(str(row[0]), message):
            raise RuntimeError("Postgres transcript search document is inconsistent.")
        score = transcript_search_document_score(document, query_document)
        if score <= 0:
            raise RuntimeError("Postgres transcript search index is inconsistent.")
        heapq.heappush(
            candidate_heap,
            (-score, str(row[0]), -int(row[1]), row, message),
        )
        if row_number % 256 == 0:
            await asyncio.sleep(0)

    hits: list[TranscriptSearchHit] = []
    remaining_bytes = query.max_bytes
    truncated = False
    continuation_available = False
    ranked_examined = 0
    while candidate_heap:
        negative_score, session_id, negative_transcript_index, row, message = heapq.heappop(
            candidate_heap
        )
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
            interaction_id=row[2],
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
        matched_records_examined=len(rows),
        truncated=truncated,
        coverage_complete=True,
        next_cursor=next_cursor,
    )
