"""Bounded environment activity joins without materializing environments."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from cayu.sessions.environment_activity import (
    ENVIRONMENT_ACTIVITY_BATCH_LIMIT,
    EnvironmentActivityQuery,
    EnvironmentActivitySnapshot,
)
from cayu.sessions.store import SessionStore


def artifact_storage_kind(store: object | None) -> str | None:
    if store is None:
        return None
    try:
        kind = getattr(store, "storage_kind", None)
    except Exception:
        return "unknown"
    return kind if type(kind) is str and kind in {"local_disk", "s3", "unknown"} else "unknown"


async def environment_activity_details(
    store: SessionStore, names: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    empty = {"session_count": None, "last_activity_at": None, "activity_as_of": None}
    if not store.supports_environment_activity:
        return {name: empty.copy() for name in names}
    result: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(names), ENVIRONMENT_ACTIVITY_BATCH_LIMIT):
        query = EnvironmentActivityQuery(
            environment_names=names[offset : offset + ENVIRONMENT_ACTIVITY_BATCH_LIMIT]
        )
        try:
            snapshot = await store.aggregate_environment_activity(query)
        except Exception:
            raise HTTPException(503, "Environment activity is temporarily unavailable.") from None
        try:
            if type(snapshot) is not EnvironmentActivitySnapshot:
                raise ValueError("Invalid environment activity snapshot.")
            snapshot = snapshot.validate_for_query(query)
        except (TypeError, ValueError, AttributeError):
            raise HTTPException(
                500, "The session store returned inconsistent environment activity."
            ) from None
        for row in snapshot.environments:
            result[row.environment_name] = {
                "session_count": row.session_count,
                "last_activity_at": row.last_activity_at,
                "activity_as_of": snapshot.as_of,
            }
    return result
