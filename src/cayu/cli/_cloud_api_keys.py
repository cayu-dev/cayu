"""`cayu cloud api-keys create|list|revoke`: a signed-in admin's Organization API keys.

An Organization API key is how CI, Compass and other automated callers reach Cayu Cloud
without a person signing in. Only a signed-in Organization admin manages them; Cloud refuses an
API key, whatever its scopes. `create` generates the key here, writes it to a new file only its
owner can read, and sends Cloud only the key's SHA-256 digest, so the key never leaves this
machine and is never printed. Cloud gives the key the scopes its purpose names (Cayu Cloud's
`docs/runbooks/api-keys.md`), or those `--scope` names.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

if TYPE_CHECKING:
    from cayu.cli._cloud_api import CloudApiClient

#: The purposes and scopes Cayu Cloud accepts; Cloud maps each purpose to its scopes.
PURPOSES = ("verifier", "ci", "release", "developer")
SCOPES = ("read", "report", "deploy", "operate", "developer")
_PATH = "/v1/organization/api-keys"


class ApiKeyFileError(ValueError):
    """The key file can't be written as a new private file."""


def add_api_keys_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    keys = commands.add_parser(
        "api-keys",
        help="Create, list or revoke the Organization's API keys (signed-in admins).",
        description=(
            "Manage the Organization's API keys, which CI and automated callers use. Needs a "
            "signed-in Organization admin (`cayu cloud login`); an API key can't manage keys."
        ),
    )
    actions = keys.add_subparsers(dest="api_keys_command", required=True)
    create = actions.add_parser(
        "create",
        help="Create a key and write it to a new private file.",
        description=(
            "Generate a new Organization API key, write it to --key-file (a new file only you "
            "can read) and give Cloud only its digest. The key is never printed."
        ),
    )
    create.add_argument(
        "--name", required=True, help="What holds the key, such as compass or github-actions."
    )
    use = create.add_mutually_exclusive_group(required=True)
    use.add_argument(
        "--purpose",
        choices=PURPOSES,
        help="Give the key the narrowest scopes that fit: verifier, ci, release or developer.",
    )
    use.add_argument(
        "--scope",
        action="append",
        choices=SCOPES,
        help="A scope to give the key instead of a purpose; repeat for more.",
    )
    create.add_argument(
        "--key-file",
        type=Path,
        required=True,
        help="Where to write the key; it must not exist yet, unless --retry.",
    )
    create.add_argument(
        "--retry",
        action="store_true",
        help=(
            "Send the key already in --key-file again, after a create whose answer was lost. "
            "Cloud answers the key it kept, or keeps it now."
        ),
    )
    actions.add_parser(
        "list",
        help="List the Organization's keys.",
        description="List the Organization's API keys, revoked ones included, never the keys.",
    )
    revoke = actions.add_parser(
        "revoke",
        help="Stop one key working, at once.",
        description="Revoke one Organization API key; it stays listed as inactive.",
    )
    revoke.add_argument("key_id", help="The key's id (key_...), from `api-keys list`.")


def api_keys_command(arguments: argparse.Namespace, *, client: CloudApiClient) -> dict[str, Any]:
    action = arguments.api_keys_command
    if action == "list":
        return {"operation": "api-keys.list", "result": client.request("GET", _PATH)}
    if action == "revoke":
        key = client.request("DELETE", f"{_PATH}/{quote(arguments.key_id, safe='')}")
        return {"operation": "api-keys.revoke", "result": key}
    return {"operation": "api-keys.create", "result": _create(arguments, client=client)}


def _create(arguments: argparse.Namespace, *, client: CloudApiClient) -> dict[str, Any]:
    from cayu.cli._cloud_api import CloudApiError

    key_file = arguments.key_file.expanduser()
    if arguments.retry:
        token = _read_key_file(key_file)
    else:
        token = "cayu_" + secrets.token_urlsafe(32)
        # Written first, so Cloud never keeps a key whose only copy was lost.
        _write_new_private_file(key_file, token)
    payload: dict[str, object] = {
        "name": arguments.name,
        "api_key_digest": hashlib.sha256(token.encode()).hexdigest(),
    }
    if arguments.scope:
        payload["scopes"] = sorted(set(arguments.scope))
    else:
        payload["purpose"] = arguments.purpose
    kept = (
        f"The key file {key_file} is kept, because Cloud may have created the key. Run the same "
        "command with --retry to finish, or `cayu cloud api-keys list` to check."
    )
    try:
        key = client.request("POST", _PATH, payload=payload)
    except CloudApiError as exc:
        status = exc.status_code
        if exc.category == "api_request_rejected" and status is not None and status < 500:
            # Cloud answered that it refused the key, so it keeps none: a new file goes.
            if not arguments.retry:
                key_file.unlink(missing_ok=True)
            raise
        # The answer was lost, unreadable or a server error, which may follow a kept key.
        raise CloudApiError(
            exc.category,
            f"{exc} {kept}",
            status_code=status,
            code=exc.code,
            transport_error=exc.transport_error,
            request=exc.request,
        ) from exc
    except KeyboardInterrupt:
        print(kept, file=sys.stderr)
        raise
    return {**key, "key_file": str(key_file)}


def _read_key_file(path: Path) -> str:
    try:
        token = path.read_text().strip()
    except OSError as exc:
        raise ApiKeyFileError(f"Could not read the key file {path}: {exc.strerror}.") from None
    if not token.startswith("cayu_") or any(character.isspace() for character in token):
        raise ApiKeyFileError(f"The key file {path} holds no Cayu Cloud API key.")
    return token


def _write_new_private_file(path: Path, value: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise ApiKeyFileError(f"The key file {path} already exists; choose a new path.") from None
    except OSError as exc:
        raise ApiKeyFileError(f"Could not create the key file {path}: {exc.strerror}.") from None
    try:
        with os.fdopen(descriptor, "w") as destination:
            destination.write(value + "\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
