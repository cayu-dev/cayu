"""`cayu keys generate`: make a public authority alias keyring for the environment.

Nothing generates these keys implicitly. ``cayu new`` leaves them unset, because a
store that has used a keyring keeps requiring it, so this command is the explicit
step a project takes when it needs one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
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
        help="Generate keys that Cayu reads from the environment.",
        description=(
            "Generate keys that Cayu reads from the environment. Nothing is written "
            "unless you pass --env-file."
        ),
    )
    commands = group.add_subparsers(dest="keys_command", required=True)
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


def run_keys(args: argparse.Namespace) -> int:
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
