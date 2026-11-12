"""`cayu keys`: make a public authority alias keyring, and compare one with a store.

Nothing generates these keys implicitly. ``cayu new`` leaves them unset, because a
store that has used a keyring keeps requiring it, so ``cayu keys generate`` is the
explicit step a project takes when it needs one. ``cayu keys status`` says whether
the project's local database requires a keyring and whether the environment's matches.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from cayu.runtime.public_authority import (
    PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV,
    PUBLIC_AUTHORITY_ALIAS_KEYS_ENV,
    PublicAuthorityAliasKeyring,
    _base64url_encode,
    _require_key_id,
)

_KEY_BYTES = 32
_DEFAULT_KEY_ID = "primary"
_VARIABLES = (PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV, PUBLIC_AUTHORITY_ALIAS_KEYS_ENV)
_DEFINITION = re.compile(
    r"\s*(?P<export>export\s+)?(?P<quote>'?)(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P=quote)\s*=\s*(?P<value>.*)"
)


def add_keys_parser(subparsers: Any) -> None:
    group = subparsers.add_parser(
        "keys",
        help="Generate and check keys that Cayu reads from the environment.",
        description=(
            "Generate keys that Cayu reads from the environment, and check them against "
            "the project's local database. Nothing is written unless you pass --env-file."
        ),
    )
    commands = group.add_subparsers(dest="keys_command", required=True)
    _add_status_parser(commands)
    generate = commands.add_parser(
        "generate",
        help="Generate a public authority alias keyring.",
        description=(
            f"Generate a public authority alias keyring: {PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV} "
            f"names the active key and {PUBLIC_AUTHORITY_ALIAS_KEYS_ENV} is a JSON object "
            "mapping each key ID to the unpadded base64url encoding of 32 random bytes. "
            "Prints both as .env lines unless --env-file or --keys-only is given. Give "
            "every process that opens the same database the same keys."
        ),
    )
    generate.add_argument(
        "--key-id",
        default=_DEFAULT_KEY_ID,
        type=_key_id,
        help=(
            f"ID of the new key (default: {_DEFAULT_KEY_ID}): lowercase letters, digits, "
            "'-' and '_', starting with a letter, at most 32 characters."
        ),
    )
    output = generate.add_mutually_exclusive_group()
    output.add_argument(
        "--env-file",
        metavar="PATH",
        type=Path,
        help=(
            "Add both variables to this .env file, creating it if needed, instead of "
            "printing them. Refuses if the file already sets either variable, unless "
            "--replace is given."
        ),
    )
    output.add_argument(
        "--keys-only",
        action="store_true",
        help=(
            f"Print only the {PUBLIC_AUTHORITY_ALIAS_KEYS_ENV} JSON value, for example to "
            "store as a secret with `cayu cloud env set ... --secret --value-file -`."
        ),
    )
    generate.add_argument(
        "--replace",
        action="store_true",
        help=(
            "With --env-file, replace the values the file already sets. A store that has "
            "used the old keys refuses to open with new ones, so only replace keys that "
            "no database depends on."
        ),
    )


def _add_status_parser(commands: Any) -> None:
    status = commands.add_parser(
        "status",
        help="Show whether the local database requires a keyring and the environment's matches.",
        description=(
            "Show whether the project's local SQLite database has used a public authority "
            "alias keyring (its key IDs and when each was first used), whether "
            f"{PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV} and {PUBLIC_AUTHORITY_ALIAS_KEYS_ENV} "
            "are set, and whether they match. Reads the database without writing. Exits 0 "
            "when a writable open would succeed with this environment, 3 when it would "
            "be refused, 4 when the store is PostgreSQL, which it doesn't inspect, and 1 "
            "when the store can't be resolved or read."
        ),
    )
    status.add_argument(
        "--sqlite",
        metavar="PATH",
        help="SQLite database to inspect (default: the project's session store).",
    )
    status.add_argument("--json", action="store_true", help="Print a JSON object.")


def run_keys(args: argparse.Namespace) -> int:
    if args.keys_command == "status":
        return _run_status(args)
    if args.replace and args.env_file is None:
        print("error: --replace needs --env-file.", file=sys.stderr)
        return 2
    keyring = _generate_keyring(args.key_id)
    encoded_keys = _keys_json(keyring)
    if args.keys_only:
        print(encoded_keys)
        return 0
    lines = {
        PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV: (
            f"{PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV}={keyring.active_key_id}"
        ),
        # Single quotes keep the JSON literal for `uv run --env-file`, python-dotenv
        # and POSIX shells alike.
        PUBLIC_AUTHORITY_ALIAS_KEYS_ENV: f"{PUBLIC_AUTHORITY_ALIAS_KEYS_ENV}='{encoded_keys}'",
    }
    if args.env_file is None:
        for line in lines.values():
            print(line)
        return 0
    try:
        replaced = _write_env_file(args.env_file, lines, replace=args.replace)
    except _EnvFileRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, UnicodeError) as exc:
        print(f"error: could not update {args.env_file}: {exc}", file=sys.stderr)
        return 1
    action = "Replaced" if replaced else "Added"
    preposition = "in" if replaced else "to"
    print(
        f"{action} {PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV} and "
        f"{PUBLIC_AUTHORITY_ALIAS_KEYS_ENV} {preposition} {args.env_file} (active key "
        f"{keyring.active_key_id!r}). Keep this file out of version control."
    )
    return 0


def _run_status(args: argparse.Namespace) -> int:
    from cayu.cli._local_alias_keys import (
        environment_keyring,
        key_states,
        read_store_keyring,
        store_opens,
    )
    from cayu.cli.store_targets import (
        SessionStoreBackend,
        SessionStoreTargetError,
        resolve_session_store_target,
    )

    environment = environment_keyring()
    payload: dict[str, Any] = {
        "environment": {
            "state": environment.state,
            "error": environment.error,
            "active_key_id": (
                None if environment.codec is None else environment.codec.keyring.active_key_id
            ),
            "key_ids": (
                [] if environment.codec is None else list(environment.codec.keyring.key_ids)
            ),
        },
        "store": None,
    }
    try:
        target = resolve_session_store_target(sqlite=args.sqlite)
    except SessionStoreTargetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if target.backend is not SessionStoreBackend.SQLITE or target.sqlite_path is None:
        payload["store"] = {
            "inspected": False,
            "backend": "postgres",
            "reason": "This command reads a local SQLite store; the resolved store is PostgreSQL.",
        }
        payload["writable_open"] = "not_checked"
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            _print_status(payload)
        return 4
    path = target.sqlite_path
    try:
        store = read_store_keyring(path)
    except sqlite3.Error as exc:
        print(f"error: could not read {path}: {exc}", file=sys.stderr)
        return 1
    states = {} if store is None else key_states(store, environment)
    payload["store"] = {
        "inspected": True,
        "backend": "sqlite",
        "database": str(path),
        "exists": store is not None,
        "requires_keyring": store is not None and store.requires_keyring,
        "active_key_id": (
            None if store is None or store.recorded is None else store.recorded.active_key_id
        ),
        "keys": (
            []
            if store is None or store.recorded is None
            else [
                {
                    "key_id": key_id,
                    "first_used_at": used,
                    "environment": states[key_id],
                }
                for key_id, used in store.recorded.keys
            ]
        ),
        "retired_key_ids": [] if store is None else sorted(store.retired_key_ids),
        "aliases": 0 if store is None else store.alias_count,
    }
    opens = store_opens(store, environment)
    payload["writable_open"] = "succeeds" if opens else "refused"
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_status(payload)
    return 0 if opens else 3


def _print_status(payload: dict[str, Any]) -> None:
    environment = payload["environment"]
    if environment["state"] == "set":
        print(
            f"Environment: keyring set, active key {environment['active_key_id']!r}, "
            f"key IDs {', '.join(repr(key) for key in environment['key_ids'])}."
        )
    elif environment["state"] == "invalid":
        print(f"Environment: keyring set but invalid: {environment['error']}")
    else:
        print(
            f"Environment: no keyring ({PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV} and "
            f"{PUBLIC_AUTHORITY_ALIAS_KEYS_ENV} are unset)."
        )
    store = payload["store"]
    if not store or not store.get("inspected"):
        reason = "" if not store else f" {store['reason']}"
        print(f"Store: not inspected.{reason} Whether it opens with this keyring is not checked.")
        return
    if not store["exists"]:
        print(f"Store: {store['database']} doesn't exist yet; it opens with or without a keyring.")
        return
    if not store["requires_keyring"]:
        print(f"Store: {store['database']} has not used a keyring; it opens without one.")
        if environment["state"] == "set":
            print("Opening it with this keyring makes it require the same keyring from then on.")
        return
    print(
        f"Store: {store['database']} requires a keyring (active key {store['active_key_id']!r}, "
        f"{store['aliases']} aliases)."
    )
    labels = {
        "matches": "the environment has the same key",
        "different": "the environment has DIFFERENT material for this key ID",
        "not_in_environment": "not in the environment",
    }
    for key in store["keys"]:
        used = key["first_used_at"] or "an unrecorded time"
        print(f"  {key['key_id']!r}: first used at {used}; {labels[key['environment']]}.")
    if payload["writable_open"] == "succeeds":
        print("A writable open succeeds with this environment.")
    else:
        print(
            "A writable open is refused with this environment. Set the keys this database "
            "used; for a development database whose keys are lost, `cayu storage "
            "reset-aliases` drops its alias data."
        )


class _EnvFileRefused(ValueError):
    """The .env file can't take the new values without losing existing ones."""


def _key_id(value: str) -> str:
    try:
        return _require_key_id(value, field_name="key_id")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be lowercase letters, digits, '-' and '_', start with a letter and be "
            "at most 32 characters"
        ) from exc


def _generate_keyring(key_id: str) -> PublicAuthorityAliasKeyring:
    # The keyring validates the key ID and the encoding, so the printed values are
    # exactly what public_authority_alias_codec_from_environment() accepts.
    key = _base64url_encode(secrets.token_bytes(_KEY_BYTES))
    return PublicAuthorityAliasKeyring(active_key_id=key_id, keys={key_id: SecretStr(key)})


def _keys_json(keyring: PublicAuthorityAliasKeyring) -> str:
    keys = {key_id: secret.get_secret_value() for key_id, secret in keyring.keys.items()}
    return json.dumps(keys, separators=(",", ":"))


def _write_env_file(path: Path, lines: dict[str, str], *, replace: bool) -> bool:
    """Add or replace the variables in ``path``; return whether any were replaced."""

    # Write through a symlink to the file it points at rather than replacing the link.
    target = path.resolve()
    # Read without newline translation so CRLF line endings survive the rewrite.
    existing = target.read_bytes().decode("utf-8") if target.exists() else ""
    rows = existing.splitlines(keepends=True)
    defined: dict[str, list[int]] = {}
    exported: set[str] = set()
    for index, row in enumerate(rows):
        match = _DEFINITION.fullmatch(row.rstrip("\r\n"))
        if match is None or match.group("name") not in lines:
            continue
        if _opens_multiline_value(match.group("value")):
            raise _EnvFileRefused(
                f"{path} sets {match.group('name')} to a value that spans several lines; "
                "edit it by hand."
            )
        name = match.group("name")
        defined.setdefault(name, []).append(index)
        if match.group("export"):
            exported.add(name)
    if defined and not replace:
        names = " and ".join(name for name in _VARIABLES if name in defined)
        raise _EnvFileRefused(
            f"{path} already sets {names}. A store that has used those keys needs them "
            "to open, so they were left unchanged. Pass --replace to overwrite them."
        )

    dropped: set[int] = set()
    for name, indexes in defined.items():
        first, *later = indexes
        ending = rows[first][len(rows[first].rstrip("\r\n")) :] or "\n"
        rows[first] = ("export " if name in exported else "") + lines[name] + ending
        dropped.update(later)
    rows = [row for index, row in enumerate(rows) if index not in dropped]
    missing = [lines[name] + "\n" for name in _VARIABLES if name not in defined]
    if missing and rows and not rows[-1].endswith("\n"):
        rows[-1] += "\n"
    _replace_file(target, "".join(rows + missing))
    return bool(defined)


def _opens_multiline_value(value: str) -> bool:
    value = value.strip()
    if not value or value[0] not in "'\"":
        return False
    quote = value[0]
    escaped = False
    for character in value[1:]:
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == quote:
            return False
    return True


def _replace_file(target: Path, content: str) -> None:
    # A new file holds secrets, so only its owner may read it. An existing file
    # keeps its ownership and mode, so permissions never grant a different group access.
    original = target.stat() if target.exists() else None
    mode = original.st_mode & 0o7777 if original is not None else 0o600
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            if original is not None and os.name == "posix":
                created = os.fstat(stream.fileno())
                if (created.st_uid, created.st_gid) != (original.st_uid, original.st_gid):
                    os.fchown(stream.fileno(), original.st_uid, original.st_gid)
            stream.write(content)
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
