"""Schema fields shared by event projection and tool-result attestation."""

from __future__ import annotations

WEB_ACCESS_RESULT_AUTHORITY_FIELD = "web_access_result_authority"


_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "outcome",
        "source",
        "signal",
        "destination_fingerprint",
        "status_code",
        "retry_after_seconds",
        "retry_after_unrepresentable",
    }
)


_ROUTE_IDENTITY_KEYS = frozenset({"route_id", "kind", "profile_fingerprint"})


_ROUTE_KEYS = frozenset(
    {
        "schema_version",
        "policy",
        "selected_route",
        "execution_profile_fingerprint",
        "terminal_disposition",
        "history",
        "original_access",
        "next_eligible_at",
    }
)


_HISTORY_KEYS = frozenset(
    {"route", "invoked", "access", "action", "disposition", "next_eligible_at"}
)


WEB_ACCESS_MESSAGE_STRUCTURE_KEYS = (
    frozenset({"access", "access_state", "webbridge_route"})
    | _EVIDENCE_KEYS
    | _ROUTE_IDENTITY_KEYS
    | _ROUTE_KEYS
    | _HISTORY_KEYS
)


def _event_path(*segments: str) -> tuple[str, ...]:
    return ("result", "structured", *segments)


WEB_ACCESS_RESULT_EVENT_SCHEMA_PATHS = frozenset(
    {
        _event_path("access_state"),
        _event_path("access"),
        *(_event_path("access", key) for key in _EVIDENCE_KEYS),
        _event_path("webbridge_route"),
        *(_event_path("webbridge_route", key) for key in _ROUTE_KEYS),
        *(_event_path("webbridge_route", "selected_route", key) for key in _ROUTE_IDENTITY_KEYS),
        *(_event_path("webbridge_route", "original_access", key) for key in _EVIDENCE_KEYS),
        *(_event_path("webbridge_route", "history", "*", key) for key in _HISTORY_KEYS),
        *(
            _event_path("webbridge_route", "history", "*", "route", key)
            for key in _ROUTE_IDENTITY_KEYS
        ),
        *(_event_path("webbridge_route", "history", "*", "access", key) for key in _EVIDENCE_KEYS),
    }
)
