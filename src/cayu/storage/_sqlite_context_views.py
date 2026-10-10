"""Native SQLite context-view publication, selection and ownership.

Operations retain admission, bounded expiry, pin changes and lifecycle evidence
in one native transaction. Store composition supplies explicit capabilities.
"""

from __future__ import annotations

import json
import sqlite3
from asyncio import Lock
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from hashlib import sha256
from typing import Any

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.storage import _sqlite_records as sqlite_records


async def publish_context_view(
    connection: sqlite3.Connection, manifest, *, publication_key, lock: Lock
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
    manifest_json = sqlite_records.json_dumps(manifest.model_dump(mode="json"))
    owner = manifest.source_owner
    async with transaction(connection, lock=lock):
        existing = connection.execute(
            "SELECT * FROM cayu_context_views WHERE publication_key = ?",
            (publication_key,),
        ).fetchone()
        if existing is not None:
            restored = validate_context_view_manifest_storage(
                ContextViewManifest.model_validate(json.loads(existing["manifest_json"])),
                view_id=existing["view_id"],
                owner_scope=existing["source_owner_scope"],
                owner_id=existing["source_owner_id"],
                owner_incarnation=existing["source_owner_incarnation"],
                source_session_id=existing["source_session_id"],
                source_session_instance_id=existing["source_session_instance_id"],
                transcript_cursor=existing["transcript_cursor"],
                projection_schema=existing["projection_schema"],
                extension_set_commitment=existing["extension_set_commitment"],
            )
            if restored != manifest:
                raise ValueError("Context-view publication key conflicts with its manifest.")
            return restored.model_copy(deep=True)
        existing_view = connection.execute(
            "SELECT 1 FROM cayu_context_views WHERE view_id = ?",
            (manifest.view_id,),
        ).fetchone()
        if existing_view is not None:
            raise ValueError("Context-view ID is already bound to another manifest.")
        source = sqlite_records.load_session(connection, manifest.source_session_id)
        if source is None or source.instance_id != manifest.source_session_instance_id:
            raise LookupError("The source session incarnation is unavailable.")
        publication_count = connection.execute(
            "SELECT COUNT(*) AS count FROM cayu_context_views "
            "WHERE source_owner_scope = ? AND source_owner_id = ? "
            "AND source_owner_incarnation = ?",
            (owner.application_scope, owner.owner_id, owner.incarnation),
        ).fetchone()["count"]
        if publication_count >= CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER:
            raise OverflowError("Context-view publication quota exceeded for the owner.")
        connection.execute(
            """
                INSERT INTO cayu_context_views (
                    view_id, publication_key, source_owner_scope, source_owner_id,
                    source_owner_incarnation, source_session_id, source_session_instance_id,
                    transcript_cursor, projection_schema, extension_set_commitment, manifest_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                manifest_json,
            ),
        )
        return manifest.model_copy(deep=True)


async def lookup_context_view_publication(
    connection: sqlite3.Connection, publication_key, *, lock: Lock
):
    from cayu.sessions.context_views import (
        ContextViewManifest,
        validate_context_view_manifest_storage,
    )

    if type(publication_key) is not str:
        raise TypeError("Context-view publication keys must be strings.")
    if len(publication_key.encode("utf-8")) > 512:
        raise ValueError("publication_key must be at most 512 UTF-8 bytes.")
    publication_key = require_clean_nonblank(publication_key, "publication_key")
    async with lock:
        row = connection.execute(
            "SELECT * FROM cayu_context_views WHERE publication_key = ?",
            (publication_key,),
        ).fetchone()
        if row is None:
            return None
        return validate_context_view_manifest_storage(
            ContextViewManifest.model_validate(json.loads(row["manifest_json"])),
            view_id=row["view_id"],
            owner_scope=row["source_owner_scope"],
            owner_id=row["source_owner_id"],
            owner_incarnation=row["source_owner_incarnation"],
            source_session_id=row["source_session_id"],
            source_session_instance_id=row["source_session_instance_id"],
            transcript_cursor=row["transcript_cursor"],
            projection_schema=row["projection_schema"],
            extension_set_commitment=row["extension_set_commitment"],
        ).model_copy(deep=True)


@asynccontextmanager
async def transaction(connection: sqlite3.Connection, *, lock: Lock) -> AsyncIterator[None]:
    async with lock:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            yield


def require_lifecycle_capacity(
    connection: sqlite3.Connection, view_id: str, *, additional_slots: int = 0
) -> None:
    from cayu.sessions.context_views import validate_context_view_lifecycle_capacity

    events = connection.execute(
        "SELECT COUNT(*) FROM cayu_context_view_lifecycle_events WHERE view_id = ?",
        (view_id,),
    ).fetchone()[0]
    unsettled = connection.execute(
        "SELECT COUNT(*) FROM cayu_context_view_selections WHERE view_id = ? "
        "AND state IN ('selected', 'adopted', 'transferred')",
        (view_id,),
    ).fetchone()[0]
    validate_context_view_lifecycle_capacity(events, unsettled, additional_slots=additional_slots)


async def select_context_view(
    connection: sqlite3.Connection,
    request,
    *,
    target=None,
    lock: Lock,
    store_now: Callable[[], datetime],
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
    from cayu.storage._context_selection_fence import reconstruct_exclusion, sqlite_exclusion

    if type(request) is not ContextViewSelectionRequest:
        raise TypeError("Context-view selection requires a typed request.")
    request = ContextViewSelectionRequest.model_validate(request)
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
    async with transaction(connection, lock=lock):
        require_not_excluded(
            request,
            reconstruct_exclusion(sqlite_exclusion(connection, request.selection_key)),
            target=target,
        )
        now_ms = int(store_now().timestamp() * 1000)
        expired_rows = connection.execute(
            "SELECT selection_key, receipt_json FROM cayu_context_view_selections "
            "WHERE owner_scope = ? AND owner_id = ? AND owner_incarnation = ? "
            "AND expires_at_ms <= ? AND state = 'selected' "
            "ORDER BY (selection_key = ?) DESC, expires_at_ms, selection_key LIMIT ?",
            (
                owner.application_scope,
                owner.owner_id,
                owner.incarnation,
                now_ms,
                request.selection_key,
                CONTEXT_VIEW_EXPIRY_BATCH_SIZE,
            ),
        ).fetchall()
        connection.execute(
            """
                UPDATE cayu_context_view_selections
                SET state = 'expired',
                    receipt_json = json_set(receipt_json, '$.state', 'expired')
                WHERE selection_key IN (
                    SELECT selection_key FROM cayu_context_view_selections
                    WHERE owner_scope = ? AND owner_id = ? AND owner_incarnation = ?
                      AND expires_at_ms <= ?
                      AND state = 'selected'
                    ORDER BY (selection_key = ?) DESC, expires_at_ms, selection_key
                    LIMIT ?
                )
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
            expired_receipt = ContextViewSelectionReceipt.model_validate(
                json.loads(expired_row["receipt_json"])
            )
            require_lifecycle_capacity(connection, expired_receipt.view.view_id, additional_slots=1)
            operation_key = (
                f"expiry:{expired_row['selection_key']}:{expired_receipt.ownership_revision}"
            )
            event = ContextViewLifecycleEvent(
                event_id="sha256:"
                + sha256(f"context-view-event:{operation_key}".encode()).hexdigest(),
                operation_key=operation_key,
                selection_key=expired_row["selection_key"],
                view_id=expired_receipt.view.view_id,
                state="expired",
                owner=expired_receipt.owner,
                owner_participant=expired_receipt.owner_participant,
                pin_commitment=expired_receipt.pin_commitment,
                ownership_revision=expired_receipt.ownership_revision,
            )
            connection.execute(
                "INSERT OR IGNORE INTO cayu_context_view_lifecycle_events "
                "(event_id, operation_key, selection_key, view_id, state, owner_scope, owner_id, "
                "owner_incarnation, pin_commitment, ownership_revision, event_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    sqlite_records.json_dumps(event.model_dump(mode="json")),
                ),
            )
        existing = connection.execute(
            "SELECT * FROM cayu_context_view_selections WHERE selection_key = ?",
            (request.selection_key,),
        ).fetchone()
        if existing is not None:
            if existing["request_commitment"] != request_commitment:
                raise ValueError("Context-view selection key conflicts with its request.")
            receipt = validate_context_view_receipt_storage(
                ContextViewSelectionReceipt.model_validate(json.loads(existing["receipt_json"])),
                selection_key=existing["selection_key"],
                view_id=existing["view_id"],
                owner_scope=existing["owner_scope"],
                owner_id=existing["owner_id"],
                owner_incarnation=existing["owner_incarnation"],
                state=existing["state"],
                pin_commitment=existing["pin_commitment"],
                ownership_revision=existing["ownership_revision"],
            )
            if receipt.state == "selected" and receipt.expires_at_ms <= now_ms:
                raise ValueError("Expired context-view selection lacks cleanup evidence.")
            return receipt
        source = sqlite_records.load_session(connection, request.source_session_id)
        if source is None or source.instance_id != request.source_session_instance_id:
            raise LookupError("Context-view source session incarnation is unavailable.")
        if target is not None:
            from cayu.sessions._context_selection_fence import require_selection_source
            from cayu.storage._participant_session_records import reconstruct

            binding_row = connection.execute(
                "SELECT * FROM cayu_participant_session_bindings WHERE session_id = ?",
                (request.source_session_id,),
            ).fetchone()
            binding = (
                None if binding_row is None else reconstruct(dict(binding_row), source).binding
            )
            require_selection_source(target, binding, now_ms=now_ms)
        rows = connection.execute(
            """
                SELECT view_id FROM cayu_context_views
                WHERE source_owner_scope = ? AND source_owner_id = ?
                  AND source_owner_incarnation = ? AND source_session_id = ?
                  AND source_session_instance_id = ? AND projection_schema = ?
                  AND extension_set_commitment = ?
                  AND (? != 'exact' OR view_id = ?)
                  AND (? IS NULL OR transcript_cursor >= ?)
                  ORDER BY transcript_cursor DESC, view_id DESC
                  LIMIT ?
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
        ).fetchall()
        if not rows:
            raise LookupError("No eligible context view is available.")
        if len(rows) > request.limits.max_views:
            raise OverflowError("Context-view count exceeds the requested limit.")
        row = connection.execute(
            "SELECT * FROM cayu_context_views WHERE view_id = ?", (rows[0]["view_id"],)
        ).fetchone()
        selected = validate_context_view_manifest_storage(
            ContextViewManifest.model_validate(json.loads(row["manifest_json"])),
            view_id=row["view_id"],
            owner_scope=row["source_owner_scope"],
            owner_id=row["source_owner_id"],
            owner_incarnation=row["source_owner_incarnation"],
            source_session_id=row["source_session_id"],
            source_session_instance_id=row["source_session_instance_id"],
            transcript_cursor=row["transcript_cursor"],
            projection_schema=row["projection_schema"],
            extension_set_commitment=row["extension_set_commitment"],
        )
        selection_count = connection.execute(
            "SELECT COUNT(*) AS count FROM cayu_context_view_selections "
            "WHERE owner_scope = ? AND owner_id = ? AND owner_incarnation = ?",
            (owner.application_scope, owner.owner_id, owner.incarnation),
        ).fetchone()["count"]
        if selection_count >= CONTEXT_VIEW_MAX_SELECTIONS_PER_OWNER:
            raise OverflowError("Context-view selection quota exceeded for the owner.")
        active = connection.execute(
            """
                SELECT COUNT(*) AS count FROM cayu_context_view_selections
                WHERE owner_scope = ? AND owner_id = ? AND owner_incarnation = ?
                  AND state IN ('selected', 'adopted', 'transferred')
                  AND (state <> 'selected' OR expires_at_ms > ?)
                """,
            (owner.application_scope, owner.owner_id, owner.incarnation, now_ms),
        ).fetchone()["count"]
        if active >= request.limits.max_pins:
            raise OverflowError("Context-view pin count exceeds the requested limit.")
        require_lifecycle_capacity(connection, selected.view_id, additional_slots=1)
        selected_bytes = context_view_manifest_bytes(selected)
        if selected_bytes > request.limits.max_view_bytes:
            raise OverflowError("Context-view manifest exceeds the requested byte limit.")
        retained_rows = connection.execute(
            "SELECT * FROM cayu_context_view_selections "
            "WHERE owner_scope = ? AND owner_id = ? AND owner_incarnation = ? "
            "AND state IN ('selected', 'adopted', 'transferred') "
            "AND (state <> 'selected' OR expires_at_ms > ?)",
            (owner.application_scope, owner.owner_id, owner.incarnation, now_ms),
        ).fetchall()
        retained_views = {}
        for row in retained_rows:
            receipt = validate_context_view_receipt_storage(
                ContextViewSelectionReceipt.model_validate(json.loads(row["receipt_json"])),
                selection_key=row["selection_key"],
                view_id=row["view_id"],
                owner_scope=row["owner_scope"],
                owner_id=row["owner_id"],
                owner_incarnation=row["owner_incarnation"],
                state=row["state"],
                pin_commitment=row["pin_commitment"],
                ownership_revision=row["ownership_revision"],
            )
            retained_views[receipt.view.view_id] = receipt.view
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
        connection.execute(
            """
                INSERT INTO cayu_context_view_selections (
                    selection_key, request_commitment, view_id, owner_scope, owner_id,
                    owner_incarnation, state, pin_commitment, expires_at_ms,
                    ownership_revision, receipt_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                sqlite_records.json_dumps(receipt.model_dump(mode="json")),
            ),
        )
        connection.commit()
        return receipt.model_copy(deep=True)


async def lookup_context_view_selection(
    connection: sqlite3.Connection, selection_key, *, lock: Lock
):
    from cayu.sessions.context_views import ContextViewSelectionReceipt

    if type(selection_key) is not str:
        raise TypeError("Context-view selection key must be a string.")
    async with lock:
        row = connection.execute(
            "SELECT receipt_json FROM cayu_context_view_selections WHERE selection_key = ?",
            (selection_key,),
        ).fetchone()
    if row is None:
        return None
    return ContextViewSelectionReceipt.model_validate_json(row[0])


async def transition_context_view_ownership(
    connection: sqlite3.Connection,
    request,
    *,
    target=None,
    lock: Lock,
    store_now: Callable[[], datetime],
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
        reconstruct_exclusion,
        sqlite_decision,
        sqlite_exclusion,
    )

    if type(request) is not ContextViewOwnershipRequest:
        raise TypeError("Context-view ownership requires a typed request.")
    request = ContextViewOwnershipRequest.model_validate(request)
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
    async with transaction(connection, lock=lock):
        control = reconstruct_exclusion(sqlite_exclusion(connection, request.selection_key))
        require_selection_adoption(
            request,
            sqlite_decision(connection, control.request) if control is not None else None,
            target=target,
        )
        operation = connection.execute(
            "SELECT request_commitment, receipt_json "
            "FROM cayu_context_view_ownership_operations "
            "WHERE operation_key = ?",
            (request.operation_key,),
        ).fetchone()
        if operation is not None:
            if operation["request_commitment"] != request_commitment:
                raise ValueError("Ownership operation key conflicts with its request.")
            return ContextViewSelectionReceipt.model_validate(json.loads(operation["receipt_json"]))
        row = connection.execute(
            "SELECT * FROM cayu_context_view_selections WHERE selection_key = ?",
            (request.selection_key,),
        ).fetchone()
        if row is None:
            raise LookupError("Context-view selection is unavailable.")
        receipt = validate_context_view_receipt_storage(
            ContextViewSelectionReceipt.model_validate(json.loads(row["receipt_json"])),
            selection_key=row["selection_key"],
            view_id=row["view_id"],
            owner_scope=row["owner_scope"],
            owner_id=row["owner_id"],
            owner_incarnation=row["owner_incarnation"],
            state=row["state"],
            pin_commitment=row["pin_commitment"],
            ownership_revision=row["ownership_revision"],
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
        require_lifecycle_capacity(connection, receipt.view.view_id)
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
            connection.execute(
                "UPDATE cayu_context_view_selections SET state = 'expired', receipt_json = ? "
                "WHERE selection_key = ? AND ownership_revision = ?",
                (
                    sqlite_records.json_dumps(expired.model_dump(mode="json")),
                    request.selection_key,
                    request.expected_revision,
                ),
            )
            connection.execute(
                "INSERT OR IGNORE INTO cayu_context_view_lifecycle_events "
                "(event_id, operation_key, selection_key, view_id, state, owner_scope, owner_id, "
                "owner_incarnation, pin_commitment, ownership_revision, event_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    sqlite_records.json_dumps(event.model_dump(mode="json")),
                ),
            )
            connection.commit()
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
        updated_json = sqlite_records.json_dumps(updated.model_dump(mode="json"))
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
        event_json = sqlite_records.json_dumps(event.model_dump(mode="json"))
        require_lifecycle_capacity(
            connection, updated.view.view_id, additional_slots=int(next_state != "released")
        )
        cursor = connection.execute(
            "UPDATE cayu_context_view_selections SET owner_scope = ?, owner_id = ?, "
            "owner_incarnation = ?, state = ?, ownership_revision = ?, receipt_json = ? "
            "WHERE selection_key = ? AND ownership_revision = ? AND state = ?",
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
        if cursor.rowcount != 1:
            raise ValueError("Context-view ownership transition lost its compare-and-set race.")
        connection.execute(
            "INSERT INTO cayu_context_view_ownership_operations "
            "(operation_key, selection_key, request_commitment, receipt_json) VALUES (?, ?, ?, ?)",
            (
                request.operation_key,
                request.selection_key,
                request_commitment,
                updated_json,
            ),
        )
        connection.execute(
            "INSERT INTO cayu_context_view_lifecycle_events "
            "(event_id, operation_key, selection_key, view_id, state, owner_scope, owner_id, "
            "owner_incarnation, pin_commitment, ownership_revision, event_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        connection.commit()
        return updated.model_copy(deep=True)


async def read_context_view_lifecycle_events(
    connection: sqlite3.Connection,
    view_id: str,
    *,
    limit: int = 256,
    owner_participant=None,
    lock: Lock,
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
        for path, value in (
            ("participant_id", participant.participant_id),
            ("incarnation", participant.incarnation),
            ("owner.application_scope", participant.owner.application_scope),
            ("owner.owner_id", participant.owner.owner_id),
            ("owner.incarnation", participant.owner.incarnation),
        ):
            participant_filter += f" AND json_extract(event_json, '$.owner_participant.{path}') = ?"
            parameters.append(value)
    parameters.append(limit)
    async with lock:
        rows = connection.execute(
            "SELECT * FROM cayu_context_view_lifecycle_events "
            "WHERE view_id = ?"
            + participant_filter
            + " ORDER BY ownership_revision, event_id LIMIT ?",
            parameters,
        ).fetchall()
        return tuple(
            validate_context_view_lifecycle_storage(
                ContextViewLifecycleEvent.model_validate(json.loads(row["event_json"])), row
            )
            for row in rows
        )


async def read_context_view(
    connection: sqlite3.Connection, view_id: str, *, source_session_id: str, lock: Lock
):
    from cayu.sessions.context_views import (
        ContextViewManifest,
        ContextViewReadback,
        validate_context_view_manifest_storage,
    )

    async with lock:
        row = connection.execute(
            "SELECT * FROM cayu_context_views WHERE view_id = ?",
            (view_id,),
        ).fetchone()
        if row is None or row["source_session_id"] != source_session_id:
            raise LookupError("Context view is unavailable.")
        return ContextViewReadback(
            view=validate_context_view_manifest_storage(
                ContextViewManifest.model_validate(json.loads(row["manifest_json"])),
                view_id=row["view_id"],
                owner_scope=row["source_owner_scope"],
                owner_id=row["source_owner_id"],
                owner_incarnation=row["source_owner_incarnation"],
                source_session_id=row["source_session_id"],
                source_session_instance_id=row["source_session_instance_id"],
                transcript_cursor=row["transcript_cursor"],
                projection_schema=row["projection_schema"],
                extension_set_commitment=row["extension_set_commitment"],
            )
        )
