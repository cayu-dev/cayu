"""Native SQLite public-authority registry and key reconciliation.

Codec, connection and execution capabilities are supplied explicitly. Registry
operations retain native transaction, lock and backfill ownership.
"""

from __future__ import annotations

import hmac
import json
import sqlite3

from cayu.runtime.public_authority import PublicAuthorityAliasCodec
from cayu.sessions.base import (
    _authenticated_public_authority_alias_private_value,
    _public_authority_alias_store_key,
)
from cayu.storage._sqlite_connection import SQLiteOperationRunner


def _alias_key_fingerprint_matches(value: object, expected: str) -> bool:
    if type(value) is not str:
        return False
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(encoded, expected.encode("ascii"))


def register_sql_functions(
    connection: sqlite3.Connection, *, codec: PublicAuthorityAliasCodec | None
) -> None:

    def public_authority_alias(
        private_value: object,
        field_name: object,
        scope_session_id: object,
    ) -> str | None:
        if codec is None:
            return None
        if type(private_value) is not str or type(field_name) is not str:
            raise ValueError("Public authority alias source must be text.")
        if scope_session_id is not None and type(scope_session_id) is not str:
            raise ValueError("Public authority alias scope must be text or null.")
        return codec.encode(
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        )

    connection.create_function(
        "cayu_public_authority_alias",
        3,
        public_authority_alias,
        deterministic=True,
    )

    def public_authority_aliases(
        private_value: object,
        field_name: object,
        scope_session_id: object,
    ) -> str:
        if codec is None:
            return "[]"
        if type(private_value) is not str or type(field_name) is not str:
            raise ValueError("Public authority alias source must be text.")
        if scope_session_id is not None and type(scope_session_id) is not str:
            raise ValueError("Public authority alias scope must be text or null.")
        return json.dumps(
            codec.aliases(
                private_value,
                field_name=field_name,
                session_id=scope_session_id,
            ),
            separators=(",", ":"),
        )

    connection.create_function(
        "cayu_public_authority_aliases",
        3,
        public_authority_aliases,
        deterministic=True,
    )
    connection.create_function(
        "cayu_public_authority_active_key_id",
        0,
        lambda: None if codec is None else codec.keyring.active_key_id,
        deterministic=True,
    )
    connection.create_function(
        "cayu_public_authority_keyring_fingerprint",
        0,
        lambda: None if codec is None else codec.keyring_fingerprint(),
        deterministic=True,
    )


def initialize_registry(
    connection: sqlite3.Connection, *, codec: PublicAuthorityAliasCodec | None, read_only: bool
) -> None:
    """Validate key continuity and backfill each newly configured signing key."""

    if codec is None:
        initialized = connection.execute(
            "SELECT EXISTS(SELECT 1 FROM cayu_public_authority_alias_config)"
        ).fetchone()[0]
        if initialized:
            raise ValueError(
                "A public authority alias codec is required for this initialized store."
            )
        return

    configured = tuple((key_id, codec.key_fingerprint(key_id)) for key_id in codec.keyring.key_ids)
    if read_only:
        rows: dict[str, tuple[object, object]] = {}
        for row in connection.execute(
            "SELECT key_id, fingerprint, backfill_completed FROM cayu_public_authority_alias_keys"
        ):
            if type(row["key_id"]) is str:
                rows[row["key_id"]] = (row["fingerprint"], row["backfill_completed"])
        for key_id, fingerprint in configured:
            existing = rows.get(key_id)
            if existing is None or existing[1] != 1:
                raise ValueError(
                    "Read-only store has not completed the configured alias-key backfill."
                )
            if not _alias_key_fingerprint_matches(existing[0], fingerprint):
                raise ValueError(
                    "Public authority alias key material conflicts with durable state."
                )
        config = connection.execute(
            "SELECT active_key_id, keyring_fingerprint "
            "FROM cayu_public_authority_alias_config "
            "WHERE singleton = 1"
        ).fetchone()
        if (
            config is None
            or config["active_key_id"] != codec.keyring.active_key_id
            or config["keyring_fingerprint"] != codec.keyring_fingerprint()
        ):
            raise ValueError("Read-only store public authority alias active key is stale.")
        return

    try:
        with connection:
            for key_id, fingerprint in configured:
                connection.execute(
                    """
                        INSERT INTO cayu_public_authority_alias_keys (
                            key_id, fingerprint, backfill_completed
                        ) VALUES (?, ?, 0)
                        ON CONFLICT(key_id) DO NOTHING
                        """,
                    (key_id, fingerprint),
                )
                row = connection.execute(
                    """
                        SELECT fingerprint, backfill_completed
                        FROM cayu_public_authority_alias_keys
                        WHERE key_id = ?
                        """,
                    (key_id,),
                ).fetchone()
                if row is None:  # pragma: no cover - guarded by the insert above
                    raise RuntimeError("Public authority alias key state was not persisted.")
                if not _alias_key_fingerprint_matches(row["fingerprint"], fingerprint):
                    raise ValueError(
                        "Public authority alias key material conflicts with durable state."
                    )

            pending = connection.execute(
                """
                    SELECT EXISTS(
                        SELECT 1
                        FROM cayu_public_authority_alias_keys
                        WHERE key_id IN ({}) AND backfill_completed = 0
                    )
                    """.format(", ".join("?" for _ in configured)),
                tuple(key_id for key_id, _fingerprint in configured),
            ).fetchone()[0]
            if pending:
                _backfill_aliases(connection)
                connection.executemany(
                    """
                        UPDATE cayu_public_authority_alias_keys
                        SET backfill_completed = 1
                        WHERE key_id = ?
                        """,
                    ((key_id,) for key_id, _fingerprint in configured),
                )
            config = connection.execute(
                "SELECT active_key_id, keyring_fingerprint, generation, "
                "retired_key_ids_json "
                "FROM cayu_public_authority_alias_config WHERE singleton = 1"
            ).fetchone()
            desired_active = codec.keyring.active_key_id
            desired_keyring_fingerprint = codec.keyring_fingerprint()
            if config is None:
                connection.execute(
                    "INSERT INTO cayu_public_authority_alias_config "
                    "(singleton, active_key_id, keyring_fingerprint, generation, "
                    "retired_key_ids_json) VALUES (1, ?, ?, 1, '[]')",
                    (desired_active, desired_keyring_fingerprint),
                )
            elif (
                config["active_key_id"] != desired_active
                or config["keyring_fingerprint"] != desired_keyring_fingerprint
            ):
                retired = json.loads(config["retired_key_ids_json"])
                if type(retired) is not list or not all(type(value) is str for value in retired):
                    raise ValueError("Public authority alias rotation state is malformed.")
                if config["active_key_id"] != desired_active and desired_active in retired:
                    raise ValueError(
                        "A retired public authority alias key cannot become active again."
                    )
                if config["active_key_id"] != desired_active:
                    retired.append(str(config["active_key_id"]))
                connection.execute(
                    "UPDATE cayu_public_authority_alias_config "
                    "SET active_key_id = ?, keyring_fingerprint = ?, generation = ?, "
                    "retired_key_ids_json = ? "
                    "WHERE singleton = 1",
                    (
                        desired_active,
                        desired_keyring_fingerprint,
                        int(config["generation"]) + 1,
                        json.dumps(list(dict.fromkeys(retired)), separators=(",", ":")),
                    ),
                )
    except sqlite3.IntegrityError:
        raise ValueError(
            "Public authority alias registry conflicts with durable authority."
        ) from None


def _backfill_aliases(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT
                'session_id',
                '',
                alias.value,
                session.id
            FROM cayu_sessions AS session,
                 json_each(
                     cayu_public_authority_aliases(session.id, 'session_id', NULL)
                 ) AS alias
            """
    )
    connection.execute(
        """
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT
                'tool_ref',
                grant_record.session_id,
                alias.value,
                grant_record.grant_id
            FROM cayu_targeted_tool_grants AS grant_record,
                 json_each(cayu_public_authority_aliases(
                     grant_record.grant_id, 'tool_ref', grant_record.session_id
                 )) AS alias
            """
    )
    connection.execute(
        """
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT DISTINCT
                'interaction_id',
                event.session_id,
                alias.value,
                event.interaction_id
            FROM cayu_events AS event,
                 json_each(cayu_public_authority_aliases(
                     event.interaction_id, 'interaction_id', event.session_id
                 )) AS alias
            WHERE event.interaction_id IS NOT NULL
            """
    )
    connection.execute(
        """
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT DISTINCT
                'interaction_id',
                transcript.session_id,
                alias.value,
                transcript.interaction_id
            FROM cayu_transcript_messages AS transcript,
                 json_each(cayu_public_authority_aliases(
                     transcript.interaction_id, 'interaction_id', transcript.session_id
                 )) AS alias
            WHERE transcript.interaction_id IS NOT NULL
            """
    )
    connection.execute(
        """
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT DISTINCT
                'interaction_id',
                event.session_id,
                alias.value,
                interaction.value
            FROM cayu_events AS event,
                 json_each(event.payload_json, '$.interaction_ids') AS interaction,
                 json_each(cayu_public_authority_aliases(
                     interaction.value, 'interaction_id', event.session_id
                 )) AS alias
            WHERE event.event_type = 'turn.completed'
              AND json_valid(event.payload_json)
              AND json_type(event.payload_json, '$.interaction_ids') = 'array'
              AND interaction.type = 'text'
              AND trim(interaction.value) <> ''
            """
    )


def require_current_configuration(
    connection: sqlite3.Connection, *, codec: PublicAuthorityAliasCodec | None
) -> None:
    row = connection.execute(
        "SELECT active_key_id, keyring_fingerprint "
        "FROM cayu_public_authority_alias_config WHERE singleton = 1"
    ).fetchone()
    if codec is None:
        if row is not None:
            raise RuntimeError("SQLite public authority aliases require the deployment keyring.")
        return
    if (
        row is None
        or row["active_key_id"] != codec.keyring.active_key_id
        or row["keyring_fingerprint"] != codec.keyring_fingerprint()
    ):
        raise RuntimeError(
            "SQLite public authority alias key configuration is stale; reopen the store."
        )


async def register_public_authority_alias(
    run_write: SQLiteOperationRunner,
    public_alias: str,
    *,
    field_name: str,
    private_value: str,
    scope_session_id: str | None = None,
    codec: PublicAuthorityAliasCodec | None,
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

    def statement(connection: sqlite3.Connection) -> None:
        with connection:
            try:
                connection.execute(
                    """
                        INSERT INTO cayu_public_authority_aliases (
                            field_name,
                            scope_session_id,
                            public_alias,
                            private_value
                        ) VALUES (?, ?, ?, ?)
                        ON CONFLICT(field_name, scope_session_id, public_alias) DO NOTHING
                        """,
                    (field_name, scope_key, public_alias, private_value),
                )
            except sqlite3.IntegrityError:
                raise ValueError(
                    "Public authority alias conflicts with existing private authority."
                ) from None
            row = connection.execute(
                """
                    SELECT private_value
                    FROM cayu_public_authority_aliases
                    WHERE field_name = ?
                      AND scope_session_id = ?
                      AND public_alias = ?
                    """,
                (field_name, scope_key, public_alias),
            ).fetchone()
            if row is None:  # pragma: no cover - guarded by the insert above
                raise RuntimeError("Public authority alias registration was not persisted.")
            stored = str(row["private_value"])
            if not hmac.compare_digest(
                stored.encode("utf-8"),
                private_value.encode("utf-8"),
            ):
                raise ValueError(
                    "Public authority alias conflicts with existing private authority."
                )

    await run_write(statement)


async def resolve_public_authority_alias(
    run_read: SQLiteOperationRunner,
    public_alias: str,
    *,
    field_name: str,
    scope_session_id: str | None = None,
    codec: PublicAuthorityAliasCodec | None,
) -> str | None:
    """Resolve one exact alias through its indexed authority scope."""

    field_name, scope_key, public_alias = _public_authority_alias_store_key(
        public_alias,
        field_name=field_name,
        private_value=None,
        scope_session_id=scope_session_id,
    )

    def query(connection: sqlite3.Connection) -> str | None:
        row = connection.execute(
            """
                SELECT private_value
                FROM cayu_public_authority_aliases
                WHERE field_name = ?
                  AND scope_session_id = ?
                  AND public_alias = ?
                """,
            (field_name, scope_key, public_alias),
        ).fetchone()
        if row is None:
            return None
        return _authenticated_public_authority_alias_private_value(
            codec,
            public_alias,
            row["private_value"],
            field_name=field_name,
            scope_session_id=scope_session_id,
        )

    return await run_read(query)


async def public_authority_private_value_exists(
    run_read: SQLiteOperationRunner,
    private_value: str,
    *,
    field_name: str,
    scope_session_id: str | None = None,
    codec: PublicAuthorityAliasCodec | None,
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

    def query(connection: sqlite3.Connection) -> bool:
        return (
            connection.execute(
                """
                    SELECT EXISTS(
                        SELECT 1
                        FROM cayu_public_authority_aliases
                        WHERE field_name = ?
                          AND scope_session_id = ?
                          AND private_value = ?
                    )
                    """,
                (field_name, scope_key, private_value),
            ).fetchone()[0]
            == 1
        )

    return await run_read(query)
