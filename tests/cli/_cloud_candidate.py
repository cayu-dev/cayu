"""Candidate releases as a fake Cayu Cloud returns them, for the `cayu cloud` CLI tests."""

from __future__ import annotations

from typing import Any


def stored_candidate(requested: dict[str, Any]) -> dict[str, Any]:
    """A candidate statement as Cloud returns it: its defaults filled in, with an expiry."""

    return {
        "brief_id": None,
        "evaluated_against_deployment_id": None,
        "expires_at": "2026-10-23T12:00:00Z",
        "expires_in_seconds": 604800,
        "receipts": [],
        "requirement_ids": [],
        "snapshot_root": None,
        **requested,
    }


def staged_candidate(payload: object) -> dict[str, Any]:
    """The `candidate` a create's response carries when its payload asks for one."""

    requested = payload.get("candidate") if isinstance(payload, dict) else None
    if not isinstance(requested, dict):
        return {}
    return {"candidate": stored_candidate(requested)}
