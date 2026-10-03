"""Private, stdlib-only wire format for worker-independent command evidence.

This module can be shipped into the admitted Linux guest without importing Cayu.
A receipt is evidence only after verification with host-retained launch authority;
neither a file's presence nor the disappearance of a process authenticates it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, cast

SCHEMA = "cayu.docker_command_receipt.v1"
MAX_OUTPUT_BYTES = 1024 * 1024
MAX_RECEIPT_BYTES = 3 * 1024 * 1024
IDENTITY_FIELDS = frozenset(
    {
        "operation_id",
        "runner_resource_identity",
        "request_identity",
        "process_identity",
        "output_identity",
        "artifact_identity",
        "cleanup_identity",
    }
)


def guest_program() -> str:
    """Bundle the private wire codec and supervisor for an SDK-free guest.

    Only reviewed package source is sent; launch data and the authentication key
    use a separate private stdin transfer. Do not interpolate data into code.
    """

    codec = Path(__file__).read_text(encoding="utf-8")
    supervisor = (
        Path(__file__).with_name("_docker_command_supervisor.py").read_text(encoding="utf-8")
    )
    return codec + "\n" + supervisor.replace("from __future__ import annotations\n", "", 1)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def validate_identity(value: object) -> dict[str, str]:
    if type(value) is not dict:
        raise ValueError("Invalid command receipt identity.")
    owned = cast("dict[str, Any]", value)
    if (
        set(owned) != IDENTITY_FIELDS | {"schema"}
        or owned.get("schema") != "cayu.durable_runner_operation.v1"
    ):
        raise ValueError("Invalid command receipt identity.")
    for field in IDENTITY_FIELDS:
        item = owned[field]
        if (
            type(item) is not str
            or len(item) != 71
            or not item.startswith("sha256:")
            or any(char not in "0123456789abcdef" for char in item[7:])
        ):
            raise ValueError("Invalid command receipt identity.")
    return dict(owned)


def validate_key(key: object) -> bytes:
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("Invalid command receipt authority.")
    return key


def descriptor(value: object) -> dict[str, Any]:
    """Copy private persisted authority with strict, non-diagnostic validation."""
    if type(value) is not dict:
        raise ValueError("Invalid private command authority.")
    owned = cast("dict[str, Any]", value)
    if set(owned) != {
        "schema",
        "identity",
        "key",
        "container_id",
        "connection_sha256",
        "request_sha256",
        "timeout_seconds",
        "output_limit",
    }:
        raise ValueError("Invalid private command authority.")
    if owned["schema"] != "cayu.docker_command_authority.v1":
        raise ValueError("Invalid private command authority.")
    identity = validate_identity(owned["identity"])
    for field in ("key", "container_id", "connection_sha256", "request_sha256"):
        item = owned[field]
        if (
            type(item) is not str
            or len(item) != 64
            or any(c not in "0123456789abcdef" for c in item)
        ):
            raise ValueError("Invalid private command authority.")
    if (
        type(owned["timeout_seconds"]) is not int
        or not 1 <= owned["timeout_seconds"] <= 86_400
        or type(owned["output_limit"]) is not int
        or not 0 <= owned["output_limit"] <= MAX_OUTPUT_BYTES
    ):
        raise ValueError("Invalid private command authority.")
    return {**owned, "identity": identity}


def seal(payload: dict[str, Any], key: bytes) -> bytes:
    """Authenticate a terminal payload; callers must first prove descendant exit."""

    key = validate_key(key)
    encoded = canonical(payload)
    document = canonical(
        {"payload": payload, "mac": hmac.new(key, encoded, hashlib.sha256).hexdigest()}
    )
    if len(document) > MAX_RECEIPT_BYTES:
        raise ValueError("Command receipt exceeds its limit.")
    return document


def verify(
    document: bytes,
    *,
    identity: dict[str, str],
    key: bytes,
    output_limit: int,
    timeout_seconds: int,
    request_sha256: str,
) -> dict[str, Any]:
    """Validate authentication, the complete expected identity, and result bounds.

    This returns raw private evidence, not a public ToolResult. The caller still
    owns redaction, workspace observation, artifact publication and run fencing.
    """

    expected = validate_identity(identity)
    key = validate_key(key)
    if (
        type(document) is not bytes
        or len(document) > MAX_RECEIPT_BYTES
        or type(output_limit) is not int
        or not 0 <= output_limit <= MAX_OUTPUT_BYTES
        or type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= 86_400
    ):
        raise ValueError("Invalid command receipt bounds.")
    try:
        envelope = json.loads(document)
        if type(envelope) is not dict or set(envelope) != {"payload", "mac"}:
            raise ValueError
        payload, mac = envelope["payload"], envelope["mac"]
        if type(mac) is not str or len(mac) != 64:
            raise ValueError
        if not hmac.compare_digest(
            mac, hmac.new(key, canonical(payload), hashlib.sha256).hexdigest()
        ):
            raise ValueError
        if type(payload) is not dict or set(payload) != {
            "schema",
            "identity",
            "request_sha256",
            "timeout_seconds",
            "output_limit",
            "exit_code",
            "timed_out",
            "stdout",
            "stderr",
            "stdout_bytes",
            "stderr_bytes",
        }:
            raise ValueError
        if (
            payload["schema"] != SCHEMA
            or payload["request_sha256"] != request_sha256
            or validate_identity(payload["identity"]) != expected
            or type(payload["output_limit"]) is not int
            or payload["output_limit"] != output_limit
            or type(payload["timeout_seconds"]) is not int
            or payload["timeout_seconds"] != timeout_seconds
            or type(payload["timed_out"]) is not bool
            or type(payload["exit_code"]) is not int
            or not -64 <= payload["exit_code"] <= 255
        ):
            raise ValueError
        for stream in ("stdout", "stderr"):
            count = payload[stream + "_bytes"]
            value = payload[stream]
            if type(count) is not int or count < 0 or type(value) is not str:
                raise ValueError
            decoded = base64.b64decode(value, validate=True)
            if len(decoded) != min(count, output_limit):
                raise ValueError
        return payload
    except (TypeError, ValueError, KeyError, UnicodeError, RecursionError):
        # Neither the rejected output nor the signing authority belongs in a
        # diagnostic, including an automatically rendered chained exception.
        pass
    raise ValueError("Command receipt is missing, conflicting, or unauthenticated.") from None
