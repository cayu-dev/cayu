"""Read-only public-authority checks for the storage migration boundary."""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from typing import Any

from cayu.runtime.public_authority import PublicAuthorityAliasCodec

_UNCONFIGURED_FINGERPRINT = hashlib.sha256(
    b"cayu.public-authority-alias-configuration.unconfigured.v1"
).hexdigest()


def authority_configuration_receipt(
    codec: PublicAuthorityAliasCodec | None,
) -> dict[str, object]:
    """Return a secret-free identity for the requested deployment authority."""

    if codec is None:
        return {
            "configured": False,
            "active_key_id": None,
            "fingerprint": _UNCONFIGURED_FINGERPRINT,
        }
    return {
        "configured": True,
        "active_key_id": codec.keyring.active_key_id,
        "fingerprint": codec.keyring_fingerprint(),
    }


def preflight_sqlite_public_authority(
    connection: sqlite3.Connection,
    codec: PublicAuthorityAliasCodec | None,
) -> None:
    """Reject an unavailable or incompatible SQLite alias keyring without writes."""

    table_names = (
        "cayu_public_authority_aliases",
        "cayu_public_authority_alias_keys",
        "cayu_public_authority_alias_config",
    )
    present = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?, ?)",
            table_names,
        )
    }
    if not present:
        return
    if present != set(table_names):
        raise RuntimeError(
            "SQLite public authority alias registry is incomplete; restore a known-good "
            "database before migrating."
        )

    durable = {
        str(row[0]): (str(row[1]), bool(row[2]))
        for row in connection.execute(
            "SELECT key_id, fingerprint, backfill_completed "
            "FROM cayu_public_authority_alias_keys ORDER BY key_id"
        )
    }
    config = connection.execute(
        "SELECT active_key_id, keyring_fingerprint, retired_key_ids_json "
        "FROM cayu_public_authority_alias_config WHERE singleton = 1"
    ).fetchone()
    aliases_exist = bool(
        connection.execute("SELECT EXISTS(SELECT 1 FROM cayu_public_authority_aliases)").fetchone()[
            0
        ]
    )
    _validate_public_authority_state(
        backend="SQLite",
        durable=durable,
        config=None if config is None else (config[0], config[1], config[2]),
        aliases_exist=aliases_exist,
        codec=codec,
    )


async def preflight_postgres_public_authority(
    cursor: Any,
    codec: PublicAuthorityAliasCodec | None,
) -> None:
    """Reject an unavailable or incompatible Postgres alias keyring without writes."""

    table_names = (
        "cayu_public_authority_aliases",
        "cayu_public_authority_alias_keys",
        "cayu_public_authority_alias_config",
    )
    present: set[str] = set()
    for table_name in table_names:
        await cursor.execute("SELECT to_regclass(%s)", (table_name,))
        row = await cursor.fetchone()
        if row is not None and row[0] is not None:
            present.add(table_name)
    if not present:
        return
    if present != set(table_names):
        raise RuntimeError(
            "Postgres public authority alias registry is incomplete; restore a known-good "
            "database before migrating."
        )

    await cursor.execute(
        "SELECT key_id, fingerprint, backfill_completed "
        "FROM cayu_public_authority_alias_keys ORDER BY key_id"
    )
    durable = {str(row[0]): (str(row[1]), bool(row[2])) for row in await cursor.fetchall()}
    await cursor.execute(
        "SELECT active_key_id, keyring_fingerprint, retired_key_ids "
        "FROM cayu_public_authority_alias_config WHERE singleton = TRUE"
    )
    config_row = await cursor.fetchone()
    await cursor.execute("SELECT EXISTS(SELECT 1 FROM cayu_public_authority_aliases)")
    aliases_row = await cursor.fetchone()
    _validate_public_authority_state(
        backend="Postgres",
        durable=durable,
        config=(None if config_row is None else (config_row[0], config_row[1], config_row[2])),
        aliases_exist=bool(aliases_row is not None and aliases_row[0]),
        codec=codec,
    )


def _validate_public_authority_state(
    *,
    backend: str,
    durable: dict[str, tuple[str, bool]],
    config: tuple[object, object, object] | None,
    aliases_exist: bool,
    codec: PublicAuthorityAliasCodec | None,
) -> None:
    if codec is None:
        if durable or config is not None or aliases_exist:
            raise RuntimeError(
                f"{backend} public authority aliases are initialized; configure the "
                "deployment's alias keyring before migrating."
            )
        return

    configured = {key_id: codec.key_fingerprint(key_id) for key_id in codec.keyring.key_ids}
    unavailable_incomplete = sorted(
        key_id
        for key_id, (_fingerprint, completed) in durable.items()
        if key_id not in configured and not completed
    )
    if unavailable_incomplete:
        raise RuntimeError(
            f"{backend} public authority alias backfill is incomplete for an unavailable "
            "historical key; restore that key before migrating."
        )
    for key_id, fingerprint in configured.items():
        existing = durable.get(key_id)
        if existing is not None and not hmac.compare_digest(existing[0], fingerprint):
            raise RuntimeError(
                f"{backend} public authority alias key ID is already bound to different "
                "key material."
            )

    if config is None:
        if aliases_exist and not durable:
            raise RuntimeError(
                f"{backend} public authority aliases have no durable signing-key state; "
                "restore a known-good database before migrating."
            )
        return
    active_key_id = str(config[0])
    retired_value = config[2]
    if type(retired_value) is str:
        try:
            retired_value = json.loads(retired_value)
        except json.JSONDecodeError:
            retired_value = None
    if type(retired_value) is not list or not all(type(value) is str for value in retired_value):
        raise RuntimeError(f"{backend} public authority alias rotation state is malformed.")
    desired_active = codec.keyring.active_key_id
    if active_key_id != desired_active and desired_active in retired_value:
        raise RuntimeError("A retired public authority alias key cannot become active again.")


__all__ = [
    "authority_configuration_receipt",
    "preflight_postgres_public_authority",
    "preflight_sqlite_public_authority",
]
