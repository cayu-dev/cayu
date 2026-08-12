"""Inbound-webhook helpers: verify a provider signature, and derive an idempotent
task id from a delivery id.

Cayu does not receive webhooks for you (that is your HTTP layer's job — a route in
your ASGI app or ``cayu.server``). These two provider-agnostic helpers cover the
parts that are easy to get subtly wrong:

- :func:`verify_webhook_signature` — constant-time HMAC verification. Defaults
  match GitHub's ``X-Hub-Signature-256``; override ``algorithm``/``prefix`` for
  Stripe, Slack, etc.
- :func:`webhook_task_id` — a deterministic task id from stable parts (typically a
  delivery id). Feed it to ``TaskCreate(task_id=...)`` and a redelivered webhook is
  rejected by the task store as a duplicate, giving you idempotency for free.

See the [PR-reviewer recipe](../../docs/recipes/pr-reviewer.md) for both in use.
"""

from __future__ import annotations

import hashlib
import hmac

from cayu._validation import require_clean_nonblank

__all__ = [
    "WebhookSignatureError",
    "verify_webhook_signature",
    "webhook_task_id",
]


class WebhookSignatureError(ValueError):
    """Raised when a signature cannot be computed (e.g. an unsupported algorithm)."""


def verify_webhook_signature(
    secret: str | bytes,
    body: bytes,
    signature: str | None,
    *,
    algorithm: str = "sha256",
    prefix: str = "sha256=",
) -> bool:
    """Constant-time HMAC verification of a webhook signature.

    ``secret`` is the non-empty shared webhook secret, ``body`` is the raw request
    body (verify the bytes, never a re-serialized payload), and ``signature`` is
    the provider's signature header value. Returns ``True`` only on an exact,
    constant-time match.

    Defaults verify GitHub's ``X-Hub-Signature-256`` (``sha256`` digest, ``sha256=``
    prefix). For a provider that sends a bare hex digest, pass ``prefix=""``.
    Malformed untrusted signature values return ``False``; invalid verifier
    configuration raises :class:`WebhookSignatureError`.
    """
    if type(prefix) is not str:
        raise WebhookSignatureError("prefix must be a string.")
    try:
        prefix_bytes = prefix.encode("ascii")
    except UnicodeEncodeError as exc:
        raise WebhookSignatureError("prefix must contain only ASCII characters.") from exc
    if not isinstance(body, (bytes, bytearray)):
        raise WebhookSignatureError("body must be bytes.")
    if isinstance(secret, str):
        if not secret:
            raise WebhookSignatureError("secret must be non-empty.")
        key = secret.encode()
    elif isinstance(secret, bytes):
        if not secret:
            raise WebhookSignatureError("secret must be non-empty.")
        key = secret
    else:
        raise WebhookSignatureError("secret must be str or bytes.")
    try:
        mac = hmac.new(key, bytes(body), algorithm)
    except (ValueError, TypeError) as exc:
        raise WebhookSignatureError(f"Unsupported HMAC algorithm: {algorithm!r}") from exc
    if not signature:
        return False
    if type(signature) is not str:
        return False
    try:
        signature_bytes = signature.encode("ascii")
    except UnicodeEncodeError:
        return False
    if not signature_bytes.startswith(prefix_bytes):
        return False
    received_hex = signature_bytes[len(prefix_bytes) :]
    if len(received_hex) != mac.digest_size * 2:
        return False
    try:
        received_digest = bytes.fromhex(received_hex.decode("ascii"))
    except ValueError:
        return False
    if len(received_digest) != mac.digest_size:
        return False
    return hmac.compare_digest(mac.digest(), received_digest)


def webhook_task_id(*parts: str, namespace: str = "webhook") -> str:
    """A deterministic, collision-resistant task id from stable webhook parts.

    The same parts always yield the same id, so passing it as
    ``TaskCreate(task_id=webhook_task_id("github", delivery_id))`` makes a redelivered
    webhook a no-op: the task store rejects the duplicate id. Pass parts that
    uniquely identify the delivery (e.g. provider name + delivery id), not the
    mutable payload.
    """
    namespace = require_clean_nonblank(namespace, "namespace")
    if not parts:
        raise ValueError("webhook_task_id requires one or more non-empty string parts.")
    clean_parts = [require_clean_nonblank(part, "part") for part in parts]
    raw = "\x1f".join([namespace, *clean_parts]).encode()
    return f"{namespace}-{hashlib.sha256(raw).hexdigest()[:32]}"
