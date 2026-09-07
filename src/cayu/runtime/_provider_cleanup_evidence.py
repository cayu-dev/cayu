"""Bounded lookup identity for local cleanup receipts; never retry authority."""

from hashlib import sha256


def local_http_cleanup_event_id(stage_id: str) -> str:
    return "model-http-cleanup:" + sha256(stage_id.encode("utf-8")).hexdigest()
