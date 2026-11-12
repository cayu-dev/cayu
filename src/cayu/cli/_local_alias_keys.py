"""The public authority alias keyring a local SQLite store has used.

`cayu keys status` compares it with the keyring in the environment, `cayu check`
reports a store that needs one the environment lacks, and `cayu storage
reset-aliases` drops a local development store's alias data so it opens without
one again.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cayu.runtime.checks import DiagnosticSeverity, ProjectDiagnostic
from cayu.runtime.public_authority import (
    PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV,
    PUBLIC_AUTHORITY_ALIAS_KEYS_ENV,
    PublicAuthorityAliasCodec,
    _AliasStoreKeyringRequired,
    _RecordedAliasKeyring,
    public_authority_alias_codec_from_environment,
)
from cayu.storage._sqlite_public_authority import recorded_keyring

_ALIAS_TABLES = (
    "cayu_public_authority_aliases",
    "cayu_public_authority_alias_keys",
    "cayu_public_authority_alias_config",
    "cayu_public_authority_alias_key_first_uses",
)

KeyState = Literal["matches", "different", "not_in_environment"]


@dataclass(frozen=True)
class StoreKeyring:
    """What one SQLite database records about the alias keyring it has used."""

    recorded: _RecordedAliasKeyring | None
    fingerprints: Mapping[str, str]
    retired_key_ids: frozenset[str]
    alias_count: int

    @property
    def requires_keyring(self) -> bool:
        return self.recorded is not None


@dataclass(frozen=True)
class EnvironmentKeyring:
    codec: PublicAuthorityAliasCodec | None
    error: str | None = None

    @property
    def state(self) -> Literal["unset", "invalid", "set"]:
        if self.error is not None:
            return "invalid"
        return "unset" if self.codec is None else "set"


def environment_keyring(environ: Mapping[str, str] | None = None) -> EnvironmentKeyring:
    try:
        return EnvironmentKeyring(public_authority_alias_codec_from_environment(environ))
    except ValueError as exc:
        return EnvironmentKeyring(None, error=str(exc))


def read_store_keyring(path: Path) -> StoreKeyring | None:
    """Read *path* without writing; None when the file doesn't exist."""

    if not path.is_file():
        return None
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        recorded = recorded_keyring(connection)
        if recorded is None:
            return StoreKeyring(None, {}, frozenset(), 0)
        fingerprints = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT key_id, fingerprint FROM cayu_public_authority_alias_keys"
            )
        }
        config = connection.execute(
            "SELECT retired_key_ids_json FROM cayu_public_authority_alias_config "
            "WHERE singleton = 1"
        ).fetchone()
        retired = json.loads(config[0]) if config is not None else []
        aliases = connection.execute(
            "SELECT COUNT(*) FROM cayu_public_authority_aliases"
        ).fetchone()
        return StoreKeyring(
            recorded,
            fingerprints,
            frozenset(str(key) for key in retired) if isinstance(retired, list) else frozenset(),
            int(aliases[0]),
        )
    finally:
        connection.close()


def key_states(store: StoreKeyring, environment: EnvironmentKeyring) -> dict[str, KeyState]:
    """How each key the store records compares with the environment's keyring."""

    codec = environment.codec
    states: dict[str, KeyState] = {}
    for key_id, fingerprint in sorted(store.fingerprints.items()):
        if codec is None or key_id not in codec.keyring.key_ids:
            states[key_id] = "not_in_environment"
        elif codec.key_fingerprint(key_id) == fingerprint:
            states[key_id] = "matches"
        else:
            states[key_id] = "different"
    return states


def store_opens(store: StoreKeyring | None, environment: EnvironmentKeyring) -> bool:
    """Whether a writable open of this store succeeds with the environment's keyring.

    A store that has used no keyring opens either way. One that has needs a
    keyring whose shared key IDs carry the same material and whose active key
    is not one the store has retired. Keys the environment drops only stop the
    aliases they signed from resolving.
    """

    if environment.state == "invalid":
        return False
    if store is None or not store.requires_keyring:
        return True
    if environment.codec is None:
        return False
    if environment.codec.keyring.active_key_id in store.retired_key_ids:
        return False
    return "different" not in key_states(store, environment).values()


def keyring_required_diagnostic(
    error: _AliasStoreKeyringRequired, *, root: Path
) -> ProjectDiagnostic:
    """`cayu check`'s finding for a store whose keyring the environment lacks.

    It names the database, so a project whose code is fine isn't reported as
    broken because of one local file.
    """

    database = error.database or "the session store"
    with_root = Path(database)
    if with_root.is_absolute() and with_root.is_relative_to(root):
        database = with_root.relative_to(root).as_posix()
    recorded = error.recorded
    details = "" if recorded is None else " " + recorded.describe().strip()
    local = error.backend == "SQLite"
    return ProjectDiagnostic(
        code="SESSION_STORE_ALIAS_KEYRING_REQUIRED",
        severity=DiagnosticSeverity.ERROR,
        subject="session_store",
        path=database,
        message=(
            f"The {error.backend} database {database} has used a public authority alias "
            "keyring and this environment doesn't set one, so the application factory "
            "could not open it. The database is the problem, not the project's code; "
            f"checks that need the loaded application did not run.{details}"
        ),
        hint=(
            f"Set the same {PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV} and "
            f"{PUBLIC_AUTHORITY_ALIAS_KEYS_ENV} as before; `cayu keys status` compares "
            "them with the database."
            + (
                " If this is a development database whose keys are lost, "
                "`cayu storage reset-aliases` drops its alias data after a confirmation."
                if local
                else ""
            )
        ),
        tags=("configuration",),
        parameters={
            "backend": error.backend,
            "database": database,
            "key_ids": [] if recorded is None else [key for key, _used in recorded.keys],
            "active_key_id": None if recorded is None else recorded.active_key_id,
            "first_used_at": None if recorded is None else recorded.first_used_at,
        },
        documentation_anchor="cayu guide diagnostics#session-store-alias-keyring-required",
        verification_command="cayu keys status",
    )


def reset_aliases(path: Path) -> int:
    """Drop every alias row and the recorded keyring; return how many aliases went."""

    connection = sqlite3.connect(path, isolation_level=None, timeout=5)
    try:
        connection.execute("BEGIN IMMEDIATE")
        try:
            present = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?, ?, ?)",
                    _ALIAS_TABLES,
                )
            }
            removed = 0
            if "cayu_public_authority_aliases" in present:
                removed = connection.execute("DELETE FROM cayu_public_authority_aliases").rowcount
            # The config row references a key row, so it goes first.
            if "cayu_public_authority_alias_config" in present:
                connection.execute("DELETE FROM cayu_public_authority_alias_config")
            if "cayu_public_authority_alias_keys" in present:
                connection.execute("DELETE FROM cayu_public_authority_alias_keys")
            if "cayu_public_authority_alias_key_first_uses" in present:
                connection.execute("DELETE FROM cayu_public_authority_alias_key_first_uses")
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        return removed
    finally:
        connection.close()


def require_development_database(path: Path, *, project_root: Path | None) -> None:
    """Refuse anything but a SQLite file inside a project's data/ directory."""

    if project_root is None:
        raise ValueError(
            "reset-aliases runs only inside a Cayu project, on a database in its data/ directory."
        )
    if (project_root / "data").is_symlink():
        raise ValueError(
            f"{project_root / 'data'} is a symbolic link; reset-aliases only resets a "
            "database inside the project's own data/ directory."
        )
    data = (project_root / "data").resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(data) or resolved == data:
        raise ValueError(
            f"reset-aliases only resets a development database inside {data}; "
            f"{resolved} is outside it."
        )
    if os.path.islink(path):
        raise ValueError(f"{path} is a symbolic link; reset the database file itself.")
