"""Native PostgreSQL context-view publication, selection and ownership.

Operations retain admission, bounded expiry, pin changes and lifecycle evidence
in one native transaction. Store composition supplies explicit capabilities.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from hashlib import sha256
from typing import Any

from cayu._validation import canonical_bounded_durable_json_bytes, require_nonblank
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.sessions.records import Session
from cayu.storage._participant_bindings_schema import PARTICIPANT_BINDING_PROJECTION
from cayu.storage._postgres_transcript import PostgresConnection


async def lock_admission(
    cur, *, owner=None, session_id: str | None = None, lifecycle: bool = False
) -> None:
    """Acquire context-view advisory locks in the canonical order.

    Selection and publication both cover the source session and owner, so
    they must never acquire those locks in opposite orders.  Keep the
    lifecycle lock between them for operations that publish lifecycle
    evidence.  Session closure/deletion only acquire the session lock and
    therefore cannot form a cycle with this order.
    """

    if owner is not None:
        await cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"context-view-owner:{owner.application_scope}:{owner.owner_id}:{owner.incarnation}",),
        )
    if lifecycle:
        await cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            ("context-view-lifecycle",),
        )
    if session_id is not None:
        await cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"context-view-session:{session_id}",),
        )


async def publish_context_view(
    connect: PostgresConnection,
    manifest,
    *,
    publication_key,
    ensure_ready: Callable[[], Awaitable[None]],
    load_session: Callable[[Any, str], Awaitable[Session | None]],
    jsonb: Callable[[Any], Any],
    unique_violation: type[Exception],
):
    from cayu.sessions.context_views import (
        CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER,
        ContextViewManifest,
        validate_context_view_manifest_storage,
    )

    if type(manifest) is not ContextViewManifest or type(publication_key) is not str:
        raise TypeError("Context-view publication requires typed manifest and key.")
    if len(publication_key.encode("utf-8")) > 512:
        raise ValueError("publication_key must be at most 512 UTF-8 bytes.")
    publication_key = require_clean_nonblank(publication_key, "publication_key")
    await ensure_ready()
    owner = manifest.source_owner
    async with connect() as conn, conn.cursor() as cur:
        await lock_admission(
            cur,
            owner=owner,
            session_id=manifest.source_session_id,
        )
        await cur.execute(
            "SELECT * FROM cayu_context_views WHERE publication_key = %s",
            (publication_key,),
        )
        existing = await cur.fetchone()
        if existing is not None:
            restored = validate_context_view_manifest_storage(
                ContextViewManifest.model_validate(existing[-1]),
                view_id=existing[0],
                owner_scope=existing[2],
                owner_id=existing[3],
                owner_incarnation=existing[4],
                source_session_id=existing[5],
                source_session_instance_id=existing[6],
                transcript_cursor=existing[7],
                projection_schema=existing[8],
                extension_set_commitment=existing[9],
            )
            if restored != manifest:
                raise ValueError(
                    "Context-view publication key conflicts with its manifest."
                ) from None
            return restored.model_copy(deep=True)
        await cur.execute(
            "SELECT 1 FROM cayu_context_views WHERE view_id = %s",
            (manifest.view_id,),
        )
        if await cur.fetchone() is not None:
            raise ValueError("Context-view ID is already bound to another manifest.")
        source = await load_session(cur, manifest.source_session_id)
        if source is None or source.instance_id != manifest.source_session_instance_id:
            raise LookupError("The source session incarnation is unavailable.")
        await cur.execute(
            "SELECT COUNT(*) FROM cayu_context_views "
            "WHERE source_owner_scope = %s AND source_owner_id = %s "
            "AND source_owner_incarnation = %s",
            (owner.application_scope, owner.owner_id, owner.incarnation),
        )
        if (await cur.fetchone())[0] >= CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER:
            raise OverflowError("Context-view publication quota exceeded for the owner.")
        try:
            await cur.execute(
                """
                    INSERT INTO cayu_context_views (
                        view_id, publication_key, source_owner_scope, source_owner_id,
                        source_owner_incarnation, source_session_id, source_session_instance_id,
                        transcript_cursor, projection_schema, extension_set_commitment, manifest_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                (
                    manifest.view_id,
                    publication_key,
                    owner.application_scope,
                    owner.owner_id,
                    owner.incarnation,
                    manifest.source_session_id,
                    manifest.source_session_instance_id,
                    manifest.transcript_cursor,
                    manifest.projection_schema,
                    manifest.extension_set_commitment,
                    jsonb(manifest.model_dump(mode="json")),
                ),
            )
        except unique_violation:
            await conn.rollback()
            await cur.execute(
                "SELECT * FROM cayu_context_views WHERE publication_key = %s",
                (publication_key,),
            )
            existing = await cur.fetchone()
            if existing is None:
                raise RuntimeError(
                    "Context-view publication conflict was not reconstructable."
                ) from None
            restored = validate_context_view_manifest_storage(
                ContextViewManifest.model_validate(existing[-1]),
                view_id=existing[0],
                owner_scope=existing[2],
                owner_id=existing[3],
                owner_incarnation=existing[4],
                source_session_id=existing[5],
                source_session_instance_id=existing[6],
                transcript_cursor=existing[7],
                projection_schema=existing[8],
                extension_set_commitment=existing[9],
            )
            if restored != manifest:
                raise ValueError(
                    "Context-view publication key conflicts with its manifest."
                ) from None
            return restored.model_copy(deep=True)
        await conn.commit()
        return manifest.model_copy(deep=True)


async def lookup_context_view_publication(
    connect: PostgresConnection, publication_key, *, ensure_ready: Callable[[], Awaitable[None]]
):
    from cayu.sessions.context_views import (
        ContextViewManifest,
        validate_context_view_manifest_storage,
    )

    if type(publication_key) is not str:
        raise TypeError("Context-view publication keys must be strings.")
    if len(publication_key.encode("utf-8")) > 512:
        raise ValueError("publication_key must be at most 512 UTF-8 bytes.")
    publication_key = require_nonblank(publication_key, "publication_key")
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT * FROM cayu_context_views WHERE publication_key = %s",
            (publication_key,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return validate_context_view_manifest_storage(
            ContextViewManifest.model_validate(row[-1]),
            view_id=row[0],
            owner_scope=row[2],
            owner_id=row[3],
            owner_incarnation=row[4],
            source_session_id=row[5],
            source_session_instance_id=row[6],
            transcript_cursor=row[7],
            projection_schema=row[8],
            extension_set_commitment=row[9],
        ).model_copy(deep=True)


async def require_lifecycle_capacity(cur, view_id: str, *, additional_slots: int = 0) -> None:
    from cayu.sessions.context_views import validate_context_view_lifecycle_capacity

    await cur.execute(
        "SELECT COUNT(*) FROM cayu_context_view_lifecycle_events WHERE view_id = %s",
        (view_id,),
    )
    events = (await cur.fetchone())[0]
    await cur.execute(
        "SELECT COUNT(*) FROM cayu_context_view_selections WHERE view_id = %s "
        "AND state IN ('selected', 'adopted', 'transferred')",
        (view_id,),
    )
    unsettled = (await cur.fetchone())[0]
    validate_context_view_lifecycle_capacity(events, unsettled, additional_slots=additional_slots)


async def select_context_view(
    connect: PostgresConnection,
    request,
    *,
    target=None,
    ensure_ready: Callable[[], Awaitable[None]],
    store_now: Callable[[], datetime],
    load_for_update: Callable[[Any, str], Awaitable[Session | None]],
    jsonb: Callable[[Any], Any],
):
    from cayu.sessions._context_selection_fence import (
        require_not_excluded,
        require_selected_participant,
    )
    from cayu.sessions.context_views import (
        CONTEXT_VIEW_EXPIRY_BATCH_SIZE,
        CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER,
        CONTEXT_VIEW_MAX_SELECTIONS_PER_OWNER,
        ContextViewManifest,
        ContextViewSelectionReceipt,
        ContextViewSelectionRequest,
        context_view_manifest_bytes,
        json_commitment,
        validate_context_view_manifest_storage,
        validate_context_view_receipt_storage,
    )
    from cayu.storage._context_selection_fence import (
        lock_selection_key,
        postgres_exclusion,
        reconstruct_exclusion,
    )

    if type(request) is not ContextViewSelectionRequest:
        raise TypeError("Context-view selection requires a typed request.")
    request = ContextViewSelectionRequest.model_validate(request)
    await ensure_ready()
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
    owner = request.source_owner
    async with connect() as conn, conn.cursor() as cur:
        await lock_selection_key(cur, request.selection_key)
        await lock_admission(
            cur,
            owner=owner,
            lifecycle=True,
            session_id=request.source_session_id,
        )
        require_not_excluded(
            request,
            reconstruct_exclusion(await postgres_exclusion(cur, request.selection_key)),
            target=target,
        )
        now_ms = int(store_now().timestamp() * 1000)
        await cur.execute(
            "SELECT selection_key, receipt_json FROM cayu_context_view_selections "
            "WHERE owner_scope = %s AND owner_id = %s AND owner_incarnation = %s "
            "AND expires_at_ms <= %s AND state = 'selected' "
            "ORDER BY (selection_key = %s) DESC, expires_at_ms, selection_key LIMIT %s FOR UPDATE",
            (
                owner.application_scope,
                owner.owner_id,
                owner.incarnation,
                now_ms,
                request.selection_key,
                CONTEXT_VIEW_EXPIRY_BATCH_SIZE,
            ),
        )
        expired_rows = await cur.fetchall()
        await cur.execute(
            """
                WITH expired AS (
                    SELECT selection_key
                    FROM cayu_context_view_selections
                    WHERE owner_scope = %s AND owner_id = %s AND owner_incarnation = %s
                      AND expires_at_ms <= %s
                      AND state = 'selected'
                    ORDER BY (selection_key = %s) DESC, expires_at_ms, selection_key
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE cayu_context_view_selections AS selection
                SET state = 'expired',
                    receipt_json = jsonb_set(selection.receipt_json, '{state}', '"expired"'::jsonb)
                FROM expired
                WHERE selection.selection_key = expired.selection_key
                """,
            (
                owner.application_scope,
                owner.owner_id,
                owner.incarnation,
                now_ms,
                request.selection_key,
                CONTEXT_VIEW_EXPIRY_BATCH_SIZE,
            ),
        )
        from cayu.sessions.context_views import ContextViewLifecycleEvent

        for expired_row in expired_rows:
            expired_receipt = ContextViewSelectionReceipt.model_validate(expired_row[1])
            await require_lifecycle_capacity(cur, expired_receipt.view.view_id, additional_slots=1)
            operation_key = (
                f"expiry:{expired_receipt.selection_key}:{expired_receipt.ownership_revision}"
            )
            event = ContextViewLifecycleEvent(
                event_id="sha256:"
                + sha256(f"context-view-event:{operation_key}".encode()).hexdigest(),
                operation_key=operation_key,
                selection_key=expired_receipt.selection_key,
                view_id=expired_receipt.view.view_id,
                state="expired",
                owner=expired_receipt.owner,
                owner_participant=expired_receipt.owner_participant,
                pin_commitment=expired_receipt.pin_commitment,
                ownership_revision=expired_receipt.ownership_revision,
            )
            await cur.execute(
                "INSERT INTO cayu_context_view_lifecycle_events "
                "(event_id, operation_key, selection_key, view_id, state, owner_scope, owner_id, "
                "owner_incarnation, pin_commitment, ownership_revision, event_json) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (operation_key) DO NOTHING",
                (
                    event.event_id,
                    event.operation_key,
                    event.selection_key,
                    event.view_id,
                    event.state,
                    event.owner.application_scope,
                    event.owner.owner_id,
                    event.owner.incarnation,
                    event.pin_commitment,
                    event.ownership_revision,
                    jsonb(event.model_dump(mode="json")),
                ),
            )

        await cur.execute(
            "SELECT * FROM cayu_context_view_selections WHERE selection_key = %s",
            (request.selection_key,),
        )
        existing = await cur.fetchone()
        if existing is not None:
            if existing[1] != request_commitment:
                raise ValueError("Context-view selection key conflicts with its request.")
            receipt = validate_context_view_receipt_storage(
                ContextViewSelectionReceipt.model_validate(existing[9]),
                selection_key=existing[0],
                view_id=existing[2],
                owner_scope=existing[3],
                owner_id=existing[4],
                owner_incarnation=existing[5],
                state=existing[6],
                pin_commitment=existing[7],
                ownership_revision=existing[10],
            )
            if receipt.state == "selected" and receipt.expires_at_ms <= now_ms:
                raise ValueError("Expired context-view selection lacks cleanup evidence.")
            return receipt
        source = await load_for_update(cur, request.source_session_id)
        if source is None or source.instance_id != request.source_session_instance_id:
            raise LookupError("Context-view source session incarnation is unavailable.")
        if target is not None:
            from cayu.sessions._context_selection_fence import require_selection_source
            from cayu.storage._participant_session_records import reconstruct, row_mapping

            await cur.execute(
                f"SELECT {PARTICIPANT_BINDING_PROJECTION} FROM cayu_participant_session_bindings WHERE session_id = %s",
                (request.source_session_id,),
            )
            binding_row = await cur.fetchone()
            binding = (
                None
                if binding_row is None
                else reconstruct(row_mapping(binding_row), source).binding
            )
            require_selection_source(target, binding, now_ms=now_ms)
        await cur.execute(
            """
                SELECT view_id FROM cayu_context_views
                WHERE source_owner_scope = %s AND source_owner_id = %s
                  AND source_owner_incarnation = %s AND source_session_id = %s
                  AND source_session_instance_id = %s AND projection_schema = %s
                  AND extension_set_commitment = %s
                  AND (%s::text <> 'exact' OR view_id = %s)
                  AND (%s::bigint IS NULL OR transcript_cursor >= %s)
                  ORDER BY transcript_cursor DESC, view_id COLLATE "C" DESC
                  LIMIT %s
                """,
            (
                owner.application_scope,
                owner.owner_id,
                owner.incarnation,
                request.source_session_id,
                request.source_session_instance_id,
                request.projection_schema,
                request.extension_set_commitment,
                request.selector,
                request.exact_view_id,
                request.minimum_transcript_cursor,
                request.minimum_transcript_cursor,
                min(request.limits.max_views, CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER) + 1,
            ),
        )
        rows = await cur.fetchall()
        if not rows:
            raise LookupError("No eligible context view is available.")
        if len(rows) > request.limits.max_views:
            raise OverflowError("Context-view count exceeds the requested limit.")
        await cur.execute("SELECT * FROM cayu_context_views WHERE view_id = %s", (rows[0][0],))
        row = await cur.fetchone()
        selected = validate_context_view_manifest_storage(
            ContextViewManifest.model_validate(row[-1]),
            view_id=row[0],
            owner_scope=row[2],
            owner_id=row[3],
            owner_incarnation=row[4],
            source_session_id=row[5],
            source_session_instance_id=row[6],
            transcript_cursor=row[7],
            projection_schema=row[8],
            extension_set_commitment=row[9],
        )
        await cur.execute(
            "SELECT COUNT(*) FROM cayu_context_view_selections "
            "WHERE owner_scope = %s AND owner_id = %s AND owner_incarnation = %s",
            (owner.application_scope, owner.owner_id, owner.incarnation),
        )
        if (await cur.fetchone())[0] >= CONTEXT_VIEW_MAX_SELECTIONS_PER_OWNER:
            raise OverflowError("Context-view selection quota exceeded for the owner.")
        await cur.execute(
            """
                SELECT COUNT(*) FROM cayu_context_view_selections
                WHERE owner_scope = %s AND owner_id = %s AND owner_incarnation = %s
                  AND state IN ('selected', 'adopted', 'transferred')
                  AND (state <> 'selected' OR expires_at_ms > %s)
                """,
            (owner.application_scope, owner.owner_id, owner.incarnation, now_ms),
        )
        if (await cur.fetchone())[0] >= request.limits.max_pins:
            raise OverflowError("Context-view pin count exceeds the requested limit.")
        await require_lifecycle_capacity(cur, selected.view_id, additional_slots=1)
        selected_bytes = context_view_manifest_bytes(selected)
        if selected_bytes > request.limits.max_view_bytes:
            raise OverflowError("Context-view manifest exceeds the requested byte limit.")
        await cur.execute(
            "SELECT * FROM cayu_context_view_selections "
            "WHERE owner_scope = %s AND owner_id = %s AND owner_incarnation = %s "
            "AND state IN ('selected', 'adopted', 'transferred') "
            "AND (state <> 'selected' OR expires_at_ms > %s)",
            (owner.application_scope, owner.owner_id, owner.incarnation, now_ms),
        )
        retained_rows = await cur.fetchall()
        retained_views = {
            receipt.view.view_id: receipt.view
            for row in retained_rows
            for receipt in [
                validate_context_view_receipt_storage(
                    ContextViewSelectionReceipt.model_validate(row[9]),
                    selection_key=row[0],
                    view_id=row[2],
                    owner_scope=row[3],
                    owner_id=row[4],
                    owner_incarnation=row[5],
                    state=row[6],
                    pin_commitment=row[7],
                    ownership_revision=row[10],
                )
            ]
        }
        retained_views[selected.view_id] = selected
        retained_bytes = sum(context_view_manifest_bytes(view) for view in retained_views.values())
        if retained_bytes > request.limits.max_retained_bytes:
            raise OverflowError("Context-view retained bytes exceed the requested limit.")
        expires_at_ms = int(
            store_now().timestamp() * 1000 + request.limits.max_lifetime_seconds * 1000
        )
        pin_commitment = (
            "sha256:"
            + sha256(f"context-pin:{request.selection_key}:{selected.view_id}".encode()).hexdigest()
        )
        receipt = ContextViewSelectionReceipt(
            selection_key=request.selection_key,
            view=selected,
            owner=owner,
            state="selected",
            pin_commitment=pin_commitment,
            owner_participant=selected.participant,
            expires_at_ms=expires_at_ms,
        )
        require_selected_participant(target, selected)
        await cur.execute(
            """
                INSERT INTO cayu_context_view_selections (
                    selection_key, request_commitment, view_id, owner_scope, owner_id,
                    owner_incarnation, state, pin_commitment, expires_at_ms,
                    ownership_revision, receipt_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
            (
                request.selection_key,
                request_commitment,
                selected.view_id,
                owner.application_scope,
                owner.owner_id,
                owner.incarnation,
                receipt.state,
                receipt.pin_commitment,
                receipt.expires_at_ms,
                receipt.ownership_revision,
                jsonb(receipt.model_dump(mode="json")),
            ),
        )
        await conn.commit()
        return receipt.model_copy(deep=True)


async def lookup_context_view_selection(
    connect: PostgresConnection, selection_key, *, ensure_ready: Callable[[], Awaitable[None]]
):
    from cayu.sessions.context_views import ContextViewSelectionReceipt

    if type(selection_key) is not str:
        raise TypeError("Context-view selection key must be a string.")
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT receipt_json FROM cayu_context_view_selections WHERE selection_key = %s",
            (selection_key,),
        )
        row = await cur.fetchone()
    if row is None:
        return None
    value = row[0]
    if isinstance(value, str):
        return ContextViewSelectionReceipt.model_validate_json(value)
    return ContextViewSelectionReceipt.model_validate(value)


async def transition_context_view_ownership(
    connect: PostgresConnection,
    request,
    *,
    target=None,
    ensure_ready: Callable[[], Awaitable[None]],
    store_now: Callable[[], datetime],
    jsonb: Callable[[Any], Any],
    unique_violation: type[Exception],
):
    from cayu.sessions._context_selection_fence import require_selection_adoption
    from cayu.sessions.context_views import (
        ContextViewLifecycleEvent,
        ContextViewOwnershipRequest,
        ContextViewSelectionReceipt,
        canonical_bounded_durable_json_bytes,
        json_commitment,
        validate_context_view_receipt_storage,
    )
    from cayu.storage._context_selection_fence import (
        lock_selection_key,
        postgres_decision,
        postgres_exclusion,
        reconstruct_exclusion,
    )

    if type(request) is not ContextViewOwnershipRequest:
        raise TypeError("Context-view ownership requires a typed request.")
    request = ContextViewOwnershipRequest.model_validate(request)
    await ensure_ready()
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
    async with connect() as conn, conn.cursor() as cur:
        # Serialize before replay lookup, including concurrent identical
        # requests that would otherwise observe a stale post-transition row.
        await lock_selection_key(cur, request.selection_key)
        await cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            ("context-view-lifecycle",),
        )
        control = reconstruct_exclusion(await postgres_exclusion(cur, request.selection_key))
        require_selection_adoption(
            request,
            await postgres_decision(cur, control.request) if control is not None else None,
            target=target,
        )
        await cur.execute(
            "SELECT request_commitment, receipt_json "
            "FROM cayu_context_view_ownership_operations "
            "WHERE operation_key = %s",
            (request.operation_key,),
        )
        operation = await cur.fetchone()
        if operation is not None:
            if operation[0] != request_commitment:
                raise ValueError("Ownership operation key conflicts with its request.")
            return ContextViewSelectionReceipt.model_validate(operation[1])
        await cur.execute(
            "SELECT * FROM cayu_context_view_selections WHERE selection_key = %s FOR UPDATE",
            (request.selection_key,),
        )
        row = await cur.fetchone()
        if row is None:
            raise LookupError("Context-view selection is unavailable.")
        receipt = validate_context_view_receipt_storage(
            ContextViewSelectionReceipt.model_validate(row[9]),
            selection_key=row[0],
            view_id=row[2],
            owner_scope=row[3],
            owner_id=row[4],
            owner_incarnation=row[5],
            state=row[6],
            pin_commitment=row[7],
            ownership_revision=row[10],
        )
        if (
            receipt.view.view_id != request.view_id
            or receipt.pin_commitment != request.pin_commitment
        ):
            raise ValueError("Context-view pin identity conflicts with the request.")
        if (
            receipt.state != request.expected_state
            or receipt.ownership_revision != request.expected_revision
        ):
            raise ValueError("Context-view ownership state or revision is stale.")
        if receipt.owner != request.current_owner:
            raise PermissionError("The current owner does not control this context-view pin.")
        if (
            request.current_participant is not None
            and receipt.owner_participant is not None
            and receipt.owner_participant != request.current_participant
        ):
            raise PermissionError("The current participant does not control this context-view pin.")
        now_ms = int(store_now().timestamp() * 1000)
        from cayu.sessions._context_selection_fence import require_adoption_deadline

        require_adoption_deadline(target, now_ms)
        await require_lifecycle_capacity(cur, receipt.view.view_id)
        if receipt.state == "selected" and receipt.expires_at_ms <= now_ms:
            expired = receipt.model_copy(update={"state": "expired"}, deep=True)
            operation_key = f"expiry:{request.selection_key}:{receipt.ownership_revision}"
            event = ContextViewLifecycleEvent(
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
            await cur.execute(
                "UPDATE cayu_context_view_selections SET state = 'expired', receipt_json = %s "
                "WHERE selection_key = %s AND ownership_revision = %s",
                (
                    jsonb(expired.model_dump(mode="json")),
                    request.selection_key,
                    request.expected_revision,
                ),
            )
            await cur.execute(
                "INSERT INTO cayu_context_view_lifecycle_events "
                "(event_id, operation_key, selection_key, view_id, state, owner_scope, owner_id, "
                "owner_incarnation, pin_commitment, ownership_revision, event_json) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (operation_key) DO NOTHING",
                (
                    event.event_id,
                    event.operation_key,
                    event.selection_key,
                    event.view_id,
                    event.state,
                    event.owner.application_scope,
                    event.owner.owner_id,
                    event.owner.incarnation,
                    event.pin_commitment,
                    event.ownership_revision,
                    jsonb(event.model_dump(mode="json")),
                ),
            )
            await conn.commit()
            raise LookupError("Context-view selection has expired.")
        if request.operation == "release":
            next_state, next_owner = "released", receipt.owner
        elif request.operation == "adopt":
            assert request.destination_owner is not None
            next_state, next_owner = "adopted", request.destination_owner
        else:
            assert request.destination_owner is not None
            next_state, next_owner = "transferred", request.destination_owner
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
                "state": next_state,
                "owner": next_owner,
                "ownership_revision": receipt.ownership_revision + 1,
                "owner_participant": next_participant,
            },
            deep=True,
        )
        updated_json = jsonb(updated.model_dump(mode="json"))
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
        event_json = jsonb(event.model_dump(mode="json"))
        await require_lifecycle_capacity(
            cur, updated.view.view_id, additional_slots=int(next_state != "released")
        )
        await cur.execute(
            "UPDATE cayu_context_view_selections SET owner_scope = %s, owner_id = %s, "
            "owner_incarnation = %s, state = %s, ownership_revision = %s, receipt_json = %s "
            "WHERE selection_key = %s AND ownership_revision = %s AND state = %s",
            (
                updated.owner.application_scope,
                updated.owner.owner_id,
                updated.owner.incarnation,
                updated.state,
                updated.ownership_revision,
                updated_json,
                request.selection_key,
                request.expected_revision,
                request.expected_state,
            ),
        )
        if cur.rowcount != 1:
            raise ValueError("Context-view ownership transition lost its compare-and-set race.")
        try:
            await cur.execute(
                "INSERT INTO cayu_context_view_ownership_operations "
                "(operation_key, selection_key, request_commitment, receipt_json) VALUES (%s, %s, %s, %s)",
                (
                    request.operation_key,
                    request.selection_key,
                    request_commitment,
                    updated_json,
                ),
            )
        except unique_violation:
            await conn.rollback()
            async with connect() as replay_conn, replay_conn.cursor() as replay_cur:
                await replay_cur.execute(
                    "SELECT request_commitment, receipt_json "
                    "FROM cayu_context_view_ownership_operations "
                    "WHERE operation_key = %s",
                    (request.operation_key,),
                )
                replay = await replay_cur.fetchone()
                if replay is None or replay[0] != request_commitment:
                    raise ValueError(
                        "Ownership operation key conflicts with its request."
                    ) from None
                return ContextViewSelectionReceipt.model_validate(replay[1])
        await cur.execute(
            "INSERT INTO cayu_context_view_lifecycle_events "
            "(event_id, operation_key, selection_key, view_id, state, owner_scope, owner_id, "
            "owner_incarnation, pin_commitment, ownership_revision, event_json) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                event.event_id,
                event.operation_key,
                event.selection_key,
                event.view_id,
                event.state,
                event.owner.application_scope,
                event.owner.owner_id,
                event.owner.incarnation,
                event.pin_commitment,
                event.ownership_revision,
                event_json,
            ),
        )
        await conn.commit()
        return updated.model_copy(deep=True)


async def read_context_view_lifecycle_events(
    connect: PostgresConnection,
    view_id: str,
    *,
    limit: int = 256,
    owner_participant=None,
    ensure_ready: Callable[[], Awaitable[None]],
    jsonb: Callable[[Any], Any],
):
    from cayu.sessions.context_views import (
        ContextViewLifecycleEvent,
        validate_context_view_lifecycle_storage,
    )

    if type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError("Context-view lifecycle event limits must be between 1 and 256.")
    participant_filter = ""
    parameters: list[Any] = [view_id]
    if owner_participant is not None:
        from cayu.collaboration.participants import ParticipantRef

        participant = ParticipantRef.model_validate(owner_participant)
        participant_filter = " AND event_json -> 'owner_participant' = %s"
        parameters.append(jsonb(participant.model_dump(mode="json")))
    parameters.append(limit)
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT event_id, operation_key, selection_key, view_id, state, owner_scope, "
            "owner_id, owner_incarnation, pin_commitment, ownership_revision, event_json "
            "FROM cayu_context_view_lifecycle_events "
            "WHERE view_id = %s"
            + participant_filter
            + " ORDER BY ownership_revision, event_id LIMIT %s",
            parameters,
        )
        rows = await cur.fetchall()
        columns = (
            "event_id",
            "operation_key",
            "selection_key",
            "view_id",
            "state",
            "owner_scope",
            "owner_id",
            "owner_incarnation",
            "pin_commitment",
            "ownership_revision",
        )
        return tuple(
            validate_context_view_lifecycle_storage(
                ContextViewLifecycleEvent.model_validate(row[10]),
                dict(zip(columns, row[:10], strict=True)),
            )
            for row in rows
        )


async def read_context_view(
    connect: PostgresConnection,
    view_id: str,
    *,
    source_session_id: str,
    ensure_ready: Callable[[], Awaitable[None]],
):
    from cayu.sessions.context_views import (
        ContextViewManifest,
        ContextViewReadback,
        validate_context_view_manifest_storage,
    )

    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT * FROM cayu_context_views WHERE view_id = %s AND source_session_id = %s",
            (view_id, source_session_id),
        )
        row = await cur.fetchone()
        if row is None:
            raise LookupError("Context view is unavailable.")
        return ContextViewReadback(
            view=validate_context_view_manifest_storage(
                ContextViewManifest.model_validate(row[-1]),
                view_id=row[0],
                owner_scope=row[2],
                owner_id=row[3],
                owner_incarnation=row[4],
                source_session_id=row[5],
                source_session_instance_id=row[6],
                transcript_cursor=row[7],
                projection_schema=row[8],
                extension_set_commitment=row[9],
            )
        )
