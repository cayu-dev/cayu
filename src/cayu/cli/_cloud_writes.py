"""What `cayu cloud`'s command groups share for writes: idempotency keys and message text.

A write takes `--idempotency-key`; without one a fresh key is chosen and printed, on an error
too, so a write whose answer was lost can be sent again under the same key, which Cloud answers
without acting twice.
"""

from __future__ import annotations

import argparse
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cayu.cli._cloud_api import CloudApiClient, CloudApiError


class CloudWriteError(CloudApiError):
    """A refused or unanswered write, with the key to send it again under."""

    def __init__(
        self,
        error: CloudApiError,
        *,
        idempotency_key: str,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            error.category,
            str(error),
            status_code=error.status_code,
            detail=error.detail,
            code=error.code,
            rejected_fields=error.rejected_fields,
            body=error.body,
            transport_error=error.transport_error,
            request=error.request,
        )
        self.idempotency_key = idempotency_key
        self.details: dict[str, object] = {"idempotency_key": idempotency_key, **(details or {})}


def add_idempotency_key_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--idempotency-key",
        help=(
            "Send the write under this key; repeating it with the same key acts once. Without "
            "one a fresh key is chosen and printed."
        ),
    )


def write_key(arguments: argparse.Namespace, operation: str) -> str:
    """The write's `--idempotency-key`, or a fresh one named after ``operation``."""

    given = getattr(arguments, "idempotency_key", None)
    return given or f"{operation.replace('.', '-')}-{secrets.token_hex(12)}"


def send_write(
    client: CloudApiClient,
    method: str,
    path: str,
    *,
    key: str,
    payload: dict[str, object] | None = None,
    query: Mapping[str, str | Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Send one write under ``key``; a refusal or lost answer carries the key."""

    try:
        return client.request(method, path, payload=payload, idempotency_key=key, query=query)
    except CloudApiError as exc:
        raise CloudWriteError(exc, idempotency_key=key) from None


def add_message_arguments(parser: argparse.ArgumentParser) -> None:
    """A write's text, given as `--message` or read from `--message-file`."""

    message = parser.add_mutually_exclusive_group(required=True)
    message.add_argument("--message")
    message.add_argument("--message-file", type=Path, help="Read the message from this file.")


def message_text(arguments: argparse.Namespace) -> str:
    if arguments.message_file is None:
        return str(arguments.message)
    try:
        return arguments.message_file.read_text()
    except OSError as exc:
        raise CloudApiError(
            "invalid_input", f"Could not read the message file {arguments.message_file}."
        ) from exc
