"""Native PostgreSQL public-authority registry and key reconciliation.

Codec, connection and execution capabilities are supplied explicitly. Registry
operations retain native transaction, lock and backfill ownership.
"""

from __future__ import annotations

import hmac
import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol
from uuid import uuid4

from cayu.events import Event, EventType
from cayu.runtime.public_authority import (
    PublicAuthorityAliasCodec,
    _authenticated_public_authority_alias_private_value,
    _initialized_alias_store_requires_keyring_message,
    _public_authority_alias_store_key,
)
from cayu.storage._postgres_transcript import PostgresConnection
from cayu.tools.grants import TARGETED_TOOL_REFERENCE_FIELD_NAME


class SchemaTransactionLock(Protocol):
    def __call__(self, conn: Any, cur: Any, *, read_only: bool) -> Awaitable[None]: ...


async def register_public_authority_alias(
    connect: PostgresConnection,
    public_alias: str,
    *,
    field_name: str,
    private_value: str,
    scope_session_id: str | None = None,
    codec: PublicAuthorityAliasCodec | None,
    ensure_ready: Callable[[], Awaitable[None]],
) -> None:
    """Atomically register one codec-authenticated public authority alias."""

    field_name, scope_key, public_alias = _public_authority_alias_store_key(
        public_alias,
        field_name=field_name,
        private_value=private_value,
        scope_session_id=scope_session_id,
    )
    if codec is None or not codec.matches(
        public_alias,
        private_value,
        field_name=field_name,
        session_id=scope_session_id,
    ):
        raise ValueError("Public authority alias lacks valid store-configured provenance.")
    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await register_alias_row(
                    cur,
                    field_name=field_name,
                    scope_key=scope_key,
                    public_alias=public_alias,
                    private_value=private_value,
                )
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise


async def resolve_public_authority_alias(
    connect: PostgresConnection,
    public_alias: str,
    *,
    field_name: str,
    scope_session_id: str | None = None,
    codec: PublicAuthorityAliasCodec | None,
    ensure_ready: Callable[[], Awaitable[None]],
) -> str | None:
    """Resolve one exact alias through its indexed authority scope."""

    field_name, scope_key, public_alias = _public_authority_alias_store_key(
        public_alias,
        field_name=field_name,
        private_value=None,
        scope_session_id=scope_session_id,
    )
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            """
                SELECT private_value
                FROM cayu_public_authority_aliases
                WHERE field_name = %s
                  AND scope_session_id = %s
                  AND public_alias = %s
                """,
            (field_name, scope_key, public_alias),
        )
        row = await cur.fetchone()
        return _authenticated_public_authority_alias_private_value(
            codec,
            public_alias,
            None if row is None else str(row[0]),
            field_name=field_name,
            scope_session_id=scope_session_id,
        )


async def public_authority_private_value_exists(
    connect: PostgresConnection,
    private_value: str,
    *,
    field_name: str,
    scope_session_id: str | None = None,
    codec: PublicAuthorityAliasCodec | None,
    ensure_ready: Callable[[], Awaitable[None]],
) -> bool:
    if codec is None:
        raise RuntimeError("Public authority alias codec is unavailable.")
    probe = codec.encode(
        private_value,
        field_name=field_name,
        session_id=scope_session_id,
    )
    field_name, scope_key, _probe = _public_authority_alias_store_key(
        probe,
        field_name=field_name,
        private_value=private_value,
        scope_session_id=scope_session_id,
    )
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            """
                SELECT EXISTS(
                    SELECT 1
                    FROM cayu_public_authority_aliases
                    WHERE field_name = %s
                      AND scope_session_id = %s
                      AND private_value = %s
                )
                """,
            (field_name, scope_key, private_value),
        )
        row = await cur.fetchone()
        return bool(row is not None and row[0])


async def register_alias_row(
    cur: Any, *, field_name: str, scope_key: str, public_alias: str, private_value: str
) -> None:
    await cur.execute(
        """
            INSERT INTO cayu_public_authority_aliases (
                field_name,
                scope_session_id,
                public_alias,
                private_value
            ) VALUES (%s, %s, %s, %s)
            ON CONFLICT (field_name, scope_session_id, public_alias)
            DO UPDATE SET private_value = cayu_public_authority_aliases.private_value
            RETURNING private_value
            """,
        (field_name, scope_key, public_alias, private_value),
    )
    row = await cur.fetchone()
    if row is None:  # pragma: no cover - RETURNING is unconditional above
        raise RuntimeError("Public authority alias registration was not persisted.")
    stored = str(row[0])
    if not hmac.compare_digest(
        stored.encode("utf-8"),
        private_value.encode("utf-8"),
    ):
        raise ValueError("Public authority alias conflicts with existing private authority.")


async def require_current_configuration(
    connect: PostgresConnection,
    *,
    codec: PublicAuthorityAliasCodec | None,
    keyring_optional: bool = False,
) -> None:
    if codec is None and keyring_optional:
        return
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT active_key_id, keyring_fingerprint "
            "FROM cayu_public_authority_alias_config "
            "WHERE singleton = TRUE"
        )
        row = await cur.fetchone()
    if codec is None:
        if row is not None:
            raise RuntimeError(_initialized_alias_store_requires_keyring_message("Postgres"))
        return
    if (
        row is None
        or str(row[0]) != codec.keyring.active_key_id
        or str(row[1]) != codec.keyring_fingerprint()
    ):
        raise RuntimeError(
            "Postgres public authority alias key configuration is stale; reopen the store."
        )


async def reconcile_keys(
    connect: PostgresConnection,
    *,
    codec: PublicAuthorityAliasCodec | None,
    read_only: bool,
    acquire_schema_lock: SchemaTransactionLock,
    keyring_optional: bool = False,
) -> None:
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await acquire_schema_lock(
                    conn,
                    cur,
                    read_only=read_only,
                )
                await cur.execute(
                    "SELECT key_id, fingerprint, backfill_completed "
                    "FROM cayu_public_authority_alias_keys ORDER BY key_id"
                )
                durable = {str(row[0]): (str(row[1]), bool(row[2])) for row in await cur.fetchall()}
                # See alias_keyring_optional for when a store may skip the keyring.
                if codec is None and not keyring_optional:
                    await cur.execute(
                        "SELECT EXISTS(SELECT 1 FROM cayu_public_authority_alias_config)"
                    )
                    config_exists = await cur.fetchone()
                    if durable or (config_exists is not None and bool(config_exists[0])):
                        raise RuntimeError(
                            _initialized_alias_store_requires_keyring_message("Postgres")
                        )
                if codec is None:
                    await conn.commit()
                    return

                configured = {
                    key_id: codec.key_fingerprint(key_id) for key_id in codec.keyring.key_ids
                }
                unavailable_incomplete = [
                    key_id
                    for key_id, (_fingerprint, completed) in durable.items()
                    if key_id not in configured and not completed
                ]
                if unavailable_incomplete:
                    raise RuntimeError(
                        "Public authority alias backfill is incomplete for an "
                        "unavailable historical key; restore that key before startup."
                    )
                for key_id, fingerprint in configured.items():
                    existing = durable.get(key_id)
                    if existing is not None and not hmac.compare_digest(
                        existing[0].encode("utf-8"),
                        fingerprint.encode("utf-8"),
                    ):
                        raise RuntimeError(
                            "Public authority alias key ID is already bound to "
                            "different key material."
                        )
                missing = [key_id for key_id in configured if key_id not in durable]
                incomplete = [
                    key_id for key_id in configured if key_id in durable and not durable[key_id][1]
                ]
                if read_only and (missing or incomplete):
                    raise RuntimeError(
                        "Read-only Postgres stores require a completed writable "
                        "public authority alias backfill for every configured key."
                    )
                if missing or incomplete:
                    # Fence every identity producer while the new key's reverse
                    # index is backfilled. Writers that started first commit
                    # before this lock; writers that start later observe the key
                    # marker and must register aliases in their own transaction.
                    await cur.execute(
                        "LOCK TABLE cayu_sessions, cayu_events, "
                        "cayu_transcript_messages, cayu_targeted_tool_grants "
                        "IN SHARE ROW EXCLUSIVE MODE"
                    )
                for key_id in missing:
                    await cur.execute(
                        "INSERT INTO cayu_public_authority_alias_keys "
                        "(key_id, fingerprint, backfill_completed) "
                        "VALUES (%s, %s, FALSE)",
                        (key_id, configured[key_id]),
                    )

            if missing or incomplete:
                await _backfill_aliases(conn, codec=codec)
                async with conn.cursor() as cur:
                    await cur.execute(
                        "UPDATE cayu_public_authority_alias_keys "
                        "SET backfill_completed = TRUE WHERE key_id = ANY(%s)",
                        (list(configured),),
                    )
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT active_key_id, keyring_fingerprint, generation, "
                    "retired_key_ids "
                    "FROM cayu_public_authority_alias_config WHERE singleton = TRUE"
                    + ("" if read_only else " FOR UPDATE")
                )
                config = await cur.fetchone()
                desired_active = codec.keyring.active_key_id
                desired_keyring_fingerprint = codec.keyring_fingerprint()
                if config is None:
                    if read_only:
                        raise RuntimeError(
                            "Read-only Postgres store has no active alias-key state."
                        )
                    await cur.execute(
                        "INSERT INTO cayu_public_authority_alias_config "
                        "(singleton, active_key_id, keyring_fingerprint, generation, "
                        "retired_key_ids) VALUES (TRUE, %s, %s, 1, '[]'::jsonb)",
                        (desired_active, desired_keyring_fingerprint),
                    )
                elif (
                    str(config[0]) != desired_active
                    or str(config[1]) != desired_keyring_fingerprint
                ):
                    if read_only:
                        raise RuntimeError(
                            "Read-only Postgres public authority alias active key is stale."
                        )
                    retired_value = config[3]
                    retired = (
                        retired_value
                        if type(retired_value) is list
                        else json.loads(str(retired_value))
                    )
                    if type(retired) is not list or not all(
                        type(value) is str for value in retired
                    ):
                        raise RuntimeError(
                            "Postgres public authority alias rotation state is malformed."
                        )
                    if str(config[0]) != desired_active and desired_active in retired:
                        raise RuntimeError(
                            "A retired public authority alias key cannot become active again."
                        )
                    if str(config[0]) != desired_active:
                        retired.append(str(config[0]))
                    await cur.execute(
                        "UPDATE cayu_public_authority_alias_config "
                        "SET active_key_id = %s, keyring_fingerprint = %s, "
                        "generation = %s, retired_key_ids = %s::jsonb "
                        "WHERE singleton = TRUE",
                        (
                            desired_active,
                            desired_keyring_fingerprint,
                            int(config[2]) + 1,
                            json.dumps(list(dict.fromkeys(retired))),
                        ),
                    )
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise


async def _backfill_aliases(conn: Any, *, codec: PublicAuthorityAliasCodec | None) -> None:
    if codec is None:  # pragma: no cover - guarded by reconciliation
        raise AssertionError("Public authority alias backfill requires a codec.")
    cursor_name = f"cayu_public_authority_backfill_{uuid4().hex}"
    async with conn.cursor(name=cursor_name) as source, conn.cursor() as target:
        await source.execute("SELECT id FROM cayu_sessions ORDER BY id")
        while rows := await source.fetchmany(500):
            for (session_id,) in rows:
                private_session_id = str(session_id)
                for public_alias in codec.aliases(
                    private_session_id,
                    field_name="session_id",
                ):
                    await register_alias_row(
                        target,
                        field_name="session_id",
                        scope_key="",
                        public_alias=public_alias,
                        private_value=private_session_id,
                    )

    interaction_cursor_name = f"{cursor_name}_interactions"
    async with (
        conn.cursor(name=interaction_cursor_name) as source,
        conn.cursor() as target,
    ):
        await source.execute(
            """
                        SELECT DISTINCT authority.session_id, authority.interaction_id
                        FROM (
                            SELECT session_id, interaction_id
                            FROM cayu_events
                            WHERE interaction_id IS NOT NULL
                            UNION
                            SELECT session_id, interaction_id
                            FROM cayu_transcript_messages
                            WHERE interaction_id IS NOT NULL
                            UNION
                            SELECT event.session_id, nested.value #>> '{}' AS interaction_id
                            FROM cayu_events AS event
                            CROSS JOIN LATERAL jsonb_array_elements(
                                CASE
                                    WHEN jsonb_typeof(event.payload -> 'interaction_ids') = 'array'
                                    THEN event.payload -> 'interaction_ids'
                                    ELSE '[]'::jsonb
                                END
                            ) AS nested(value)
                            WHERE event.event_type = 'turn.completed'
                              AND jsonb_typeof(nested.value) = 'string'
                              AND btrim(nested.value #>> '{}') <> ''
                        ) AS authority
                        ORDER BY authority.session_id, authority.interaction_id
                        """
        )
        while rows := await source.fetchmany(500):
            for session_id, interaction_id in rows:
                private_session_id = str(session_id)
                private_interaction_id = str(interaction_id)
                for public_alias in codec.aliases(
                    private_interaction_id,
                    field_name="interaction_id",
                    session_id=private_session_id,
                ):
                    await register_alias_row(
                        target,
                        field_name="interaction_id",
                        scope_key=private_session_id,
                        public_alias=public_alias,
                        private_value=private_interaction_id,
                    )

    grant_cursor_name = f"{cursor_name}_targeted_grants"
    async with (
        conn.cursor(name=grant_cursor_name) as source,
        conn.cursor() as target,
    ):
        await source.execute(
            "SELECT session_id, grant_id "
            "FROM cayu_targeted_tool_grants ORDER BY session_id, grant_id"
        )
        while rows := await source.fetchmany(500):
            for session_id, grant_id in rows:
                private_session_id = str(session_id)
                private_grant_id = str(grant_id)
                for public_alias in codec.aliases(
                    private_grant_id,
                    field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                    session_id=private_session_id,
                ):
                    await register_alias_row(
                        target,
                        field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                        scope_key=private_session_id,
                        public_alias=public_alias,
                        private_value=private_grant_id,
                    )


async def register_authorities(
    cur: Any,
    session_id: str,
    *,
    interaction_ids: tuple[str, ...] = (),
    codec: PublicAuthorityAliasCodec | None,
) -> None:
    if codec is None:
        await cur.execute("SELECT EXISTS(SELECT 1 FROM cayu_public_authority_alias_keys)")
        row = await cur.fetchone()
        if row is not None and row[0] is True:
            raise RuntimeError(_initialized_alias_store_requires_keyring_message("Postgres"))
        return
    await cur.execute(
        "SELECT active_key_id, keyring_fingerprint "
        "FROM cayu_public_authority_alias_config "
        "WHERE singleton = TRUE"
    )
    active = await cur.fetchone()
    if (
        active is None
        or str(active[0]) != codec.keyring.active_key_id
        or str(active[1]) != codec.keyring_fingerprint()
    ):
        raise RuntimeError("Postgres public authority alias writer uses a stale active key.")
    for public_alias in codec.aliases(session_id, field_name="session_id"):
        await register_alias_row(
            cur,
            field_name="session_id",
            scope_key="",
            public_alias=public_alias,
            private_value=session_id,
        )
    for interaction_id in dict.fromkeys(interaction_ids):
        for public_alias in codec.aliases(
            interaction_id,
            field_name="interaction_id",
            session_id=session_id,
        ):
            await register_alias_row(
                cur,
                field_name="interaction_id",
                scope_key=session_id,
                public_alias=public_alias,
                private_value=interaction_id,
            )


async def register_event_authorities(
    cur: Any,
    session_id: str,
    events: list[Event] | tuple[Event, ...],
    *,
    codec: PublicAuthorityAliasCodec | None,
) -> None:
    interaction_ids: list[str] = []
    for event in events:
        if event.interaction_id is not None:
            interaction_ids.append(event.interaction_id)
        if event.type == EventType.TURN_COMPLETED:
            nested = event.payload.get("interaction_ids")
            if type(nested) is list:
                interaction_ids.extend(value for value in nested if type(value) is str and value)
    await register_authorities(cur, session_id, interaction_ids=tuple(interaction_ids), codec=codec)
