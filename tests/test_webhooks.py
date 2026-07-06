"""Tests for ``cayu.webhooks`` — HMAC verification and idempotent task ids."""

from __future__ import annotations

import hmac
from typing import Any

import pytest

from cayu import WebhookSignatureError, verify_webhook_signature, webhook_task_id


def _github_sig(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, "sha256").hexdigest()


def test_verify_accepts_a_valid_github_signature() -> None:
    body = b'{"action": "opened"}'
    assert verify_webhook_signature("s3cret", body, _github_sig("s3cret", body)) is True


def test_verify_rejects_wrong_secret_and_tampered_body() -> None:
    body = b'{"action": "opened"}'
    good = _github_sig("s3cret", body)
    assert verify_webhook_signature("wrong-secret", body, good) is False
    assert verify_webhook_signature("s3cret", body + b" ", good) is False


def test_verify_rejects_missing_signature() -> None:
    assert verify_webhook_signature("s3cret", b"{}", None) is False
    assert verify_webhook_signature("s3cret", b"{}", "") is False


def test_verify_rejects_non_string_signature() -> None:
    signature: Any = b"sha256=x"
    assert verify_webhook_signature("s3cret", b"{}", signature) is False


def test_verify_supports_prefixless_providers() -> None:
    body = b"payload"
    bare = hmac.new(b"key", body, "sha256").hexdigest()
    assert verify_webhook_signature(b"key", body, bare, prefix="") is True
    # With the default GitHub prefix, a bare digest must not match.
    assert verify_webhook_signature(b"key", body, bare) is False


def test_verify_rejects_non_bytes_body() -> None:
    body: Any = "not-bytes"
    with pytest.raises(WebhookSignatureError):
        verify_webhook_signature("s", body, "sha256=x")


def test_verify_rejects_invalid_secret() -> None:
    with pytest.raises(WebhookSignatureError, match="secret must be non-empty"):
        verify_webhook_signature("", b"{}", "sha256=x")
    with pytest.raises(WebhookSignatureError, match="secret must be non-empty"):
        verify_webhook_signature(b"", b"{}", "sha256=x")
    secret: Any = 123
    with pytest.raises(WebhookSignatureError, match="secret must be str or bytes"):
        verify_webhook_signature(secret, b"{}", "sha256=x")


def test_verify_rejects_invalid_prefix() -> None:
    prefix: Any = 123
    with pytest.raises(WebhookSignatureError, match="prefix must be a string"):
        verify_webhook_signature("s", b"{}", "sha256=x", prefix=prefix)


def test_verify_raises_on_unsupported_algorithm() -> None:
    with pytest.raises(WebhookSignatureError):
        verify_webhook_signature("s", b"b", "x", algorithm="not-a-real-hash")


def test_webhook_task_id_is_deterministic_and_distinct() -> None:
    assert webhook_task_id("github", "delivery-1") == webhook_task_id("github", "delivery-1")
    assert webhook_task_id("github", "delivery-1") != webhook_task_id("github", "delivery-2")
    assert webhook_task_id("stripe", "evt_1") != webhook_task_id("github", "evt_1")
    assert webhook_task_id("github", "delivery-1").startswith("webhook-")


def test_webhook_task_id_rejects_empty_parts() -> None:
    with pytest.raises(ValueError):
        webhook_task_id()
    with pytest.raises(ValueError):
        webhook_task_id("github", "")
    with pytest.raises(ValueError):
        webhook_task_id("github", " delivery-1 ")


def test_webhook_task_id_rejects_invalid_namespace() -> None:
    with pytest.raises(ValueError, match="namespace"):
        webhook_task_id("github", "delivery-1", namespace="")
    with pytest.raises(ValueError, match="namespace"):
        webhook_task_id("github", "delivery-1", namespace=" webhook ")
    namespace: Any = 123
    with pytest.raises(ValueError, match="namespace"):
        webhook_task_id("github", "delivery-1", namespace=namespace)
