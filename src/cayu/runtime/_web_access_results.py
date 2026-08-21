from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any, Literal, cast

from cayu._validation import canonical_durable_json_bytes, copy_durable_json_value
from cayu.core.events import (
    Event,
    event_payload_authority_is_runtime_generated,
    event_with_runtime_payload_authority,
)
from cayu.core.tools import ToolResult

WEB_ACCESS_RESULT_AUTHORITY_FIELD = "web_access_result_authority"
_AUTHORITY_PREFIX = "cayu.web-access-result.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ResultKind = Literal["browser_session", "routing", "web_fetch"]

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

_OUTCOMES = frozenset(
    {
        "authentication_required",
        "bot_challenge",
        "consent_required",
        "content_unavailable",
        "destination_denied",
        "rate_limited",
        "transient_transport_failure",
    }
)
_SOURCES = frozenset(
    {"browser_response", "egress_policy", "hosted_provider", "http_response", "transport"}
)
_SIGNALS = frozenset(
    {
        "challenge_header",
        "consent_header",
        "egress_denial",
        "provider_status",
        "retry_after",
        "status_code",
        "transport_error",
        "www_authenticate",
    }
)
_ROUTE_KINDS = frozenset({"hosted_provider", "sandboxed_browser", "trusted_local"})
_ROUTE_ACTIONS = frozenset({"fallback", "operator_action", "stop", "wait"})
_ROUTE_DISPOSITIONS = frozenset(
    {
        "durable_authority_unavailable",
        "failure_unrecorded",
        "fallback_succeeded",
        "operator_action",
        "route_conflict",
        "route_failed",
        "stopped",
        "success",
        "success_unrecorded",
        "wait",
    }
)
_ACCESS_STATES = frozenset({"available", "blocked", "unknown"})


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


def attest_runtime_web_access_result(
    event: Event,
    result: ToolResult,
    *,
    tool: object | None,
) -> Event:
    """Attest selected controls produced by one exact built-in web tool."""

    kind = _runtime_web_tool_kind(tool)
    controls = None if kind is None else _controls(result.structured, kind=kind, finite=False)
    if not controls:
        return event
    marker = _marker(kind, controls)
    payload = dict(event.payload)
    payload[WEB_ACCESS_RESULT_AUTHORITY_FIELD] = marker
    return event_with_runtime_payload_authority(
        event.model_copy(update={"payload": payload}),
        WEB_ACCESS_RESULT_AUTHORITY_FIELD,
    )


def restore_attested_tool_result(
    event: Event,
    *,
    original: ToolResult,
    redacted: ToolResult,
) -> tuple[Event, ToolResult]:
    controls = _attested_controls(event, original.structured, trust_persisted=False)
    if controls is None:
        return _without_authority(event), redacted
    return event, _result_with_controls(redacted, controls)


def preserve_attested_controls_across_hook(
    event: Event,
    *,
    original: ToolResult,
    replacement: ToolResult,
) -> ToolResult:
    """Carry only previously attested controls across a hook replacement."""

    controls = _attested_controls(event, original.structured, trust_persisted=False)
    if controls is None:
        return replacement
    preserved = _result_with_controls(replacement, controls)
    if "webbridge_route" not in controls:
        return preserved

    original_structured = _json_object(original.structured)
    preserved_structured = _json_object(preserved.structured)
    if original_structured is None or preserved_structured is None:
        raise AssertionError("Attested routing controls require structured tool results.")
    original_route = original_structured.get("webbridge_route")
    preserved_route = preserved_structured.get("webbridge_route")
    if type(original_route) is not dict or type(preserved_route) is not dict:
        raise AssertionError("Attested routing controls require routing objects.")
    effective_source_url = original_route.get("effective_source_url")
    if type(effective_source_url) is str:
        preserved_route["effective_source_url"] = effective_source_url
    else:
        preserved_route.pop("effective_source_url", None)
    return ToolResult(
        content=preserved.content,
        structured=preserved_structured,
        artifacts=preserved.artifacts,
        is_error=preserved.is_error,
    )


def restore_persisted_web_access_result_authority(event: Event) -> Event:
    """Restore private authority after a coordinator-owned stage is loaded."""

    if WEB_ACCESS_RESULT_AUTHORITY_FIELD not in event.payload:
        return event
    result = event.payload.get("result")
    controls = (
        None
        if type(result) is not dict
        else _attested_controls(
            event,
            result.get("structured"),
            trust_persisted=True,
        )
    )
    if controls is None:
        raise ValueError("Persisted web-access result authority is malformed.")
    return event_with_runtime_payload_authority(
        event,
        WEB_ACCESS_RESULT_AUTHORITY_FIELD,
    )


def restore_attested_event_result(
    event: Event,
    *,
    redacted_payload: dict[str, Any],
    trust_persisted: bool,
    reject_malformed: bool,
) -> None:
    if WEB_ACCESS_RESULT_AUTHORITY_FIELD not in event.payload:
        return
    source_result = event.payload.get("result")
    target_result = redacted_payload.get("result")
    controls = (
        None
        if type(source_result) is not dict
        else _attested_controls(
            event,
            source_result.get("structured"),
            trust_persisted=trust_persisted,
        )
    )
    target_structured = None if type(target_result) is not dict else target_result.get("structured")
    if controls is None or type(target_structured) is not dict:
        if reject_malformed:
            raise ValueError("Web-access result authority is malformed.")
        return
    _overlay(target_structured, controls)


def finite_message_controls(structured: object) -> dict[str, Any]:
    """Return only closed controls safe to preserve without private provenance."""

    source = _json_object(structured)
    if source is None:
        return {}
    kind: _ResultKind | None = None
    if "webbridge_route" in source:
        kind = "routing"
    elif "access_state" in source:
        kind = "browser_session"
    elif "access" in source:
        kind = "web_fetch"
    return {} if kind is None else _controls(source, kind=kind, finite=True)


def persisted_web_access_control_paths(
    value: object,
) -> frozenset[tuple[str, ...]]:
    """Return exact result-control paths covered by one persisted attestation."""

    if type(value) is not dict:
        return frozenset()
    try:
        event = Event.model_validate(value)
    except (TypeError, ValueError):
        return frozenset()
    raw_result = event.payload.get("result")
    structured = None if type(raw_result) is not dict else raw_result.get("structured")
    controls = _attested_controls(event, structured, trust_persisted=True)
    if controls is None:
        return frozenset()
    return frozenset(
        {
            ("payload", WEB_ACCESS_RESULT_AUTHORITY_FIELD),
            *(
                ("payload", "result", "structured", *path)
                for path in _string_control_paths(controls)
            ),
        }
    )


def restore_finite_message_controls(original: object, redacted: object) -> None:
    controls = finite_message_controls(original)
    if controls and type(redacted) is dict:
        _overlay(cast("dict[str, Any]", redacted), controls)


def _runtime_web_tool_kind(tool: object | None) -> _ResultKind | None:
    if tool is None:
        return None
    from cayu.tools.browser_session import BrowserSessionTool
    from cayu.tools.web import WebFetchTool
    from cayu.tools.web_access import WebAccessRoutingTool

    if type(tool) is WebAccessRoutingTool:
        return "routing"
    if type(tool) is BrowserSessionTool:
        return "browser_session"
    if type(tool) is WebFetchTool:
        return "web_fetch"
    return None


def _controls(structured: object, *, kind: _ResultKind, finite: bool) -> dict[str, Any]:
    source = _json_object(structured)
    if source is None:
        return {}
    controls: dict[str, Any] = {}
    access = _evidence(source.get("access"), finite=finite)
    if access:
        controls["access"] = access
    if kind == "web_fetch":
        return controls
    if kind == "browser_session":
        state = source.get("access_state")
        if type(state) is not str or state not in _ACCESS_STATES:
            return {}
        controls["access_state"] = state
        return controls
    route = _route(source.get("webbridge_route"), finite=finite)
    if route is None:
        return {}
    controls["webbridge_route"] = route
    return controls


def _evidence(value: object, *, finite: bool) -> dict[str, Any] | None:
    source = _json_object(value)
    if source is None or source.get("schema_version") != 1:
        return None
    if not finite:
        return _pick(source, _EVIDENCE_KEYS)
    outcome, evidence_source, signal = (
        source.get("outcome"),
        source.get("source"),
        source.get("signal"),
    )
    if outcome not in _OUTCOMES or evidence_source not in _SOURCES or signal not in _SIGNALS:
        return None
    return {
        "schema_version": 1,
        "outcome": outcome,
        "source": evidence_source,
        "signal": signal,
    }


def _route_identity(value: object, *, finite: bool) -> dict[str, Any] | None:
    source = _json_object(value)
    if source is None or source.get("kind") not in _ROUTE_KINDS:
        return None
    return {"kind": source["kind"]} if finite else _pick(source, _ROUTE_IDENTITY_KEYS)


def _route(value: object, *, finite: bool) -> dict[str, Any] | None:
    source = _json_object(value)
    if source is None or source.get("schema_version") != 1 or source.get("policy") != "explicit":
        return None
    selected = _route_identity(source.get("selected_route"), finite=finite)
    disposition = source.get("terminal_disposition")
    history = source.get("history")
    if selected is None or disposition not in _ROUTE_DISPOSITIONS or type(history) is not list:
        return None
    projected_history: list[dict[str, Any]] = []
    for item in history:
        projected = _history(item, finite=finite)
        if projected is None:
            return None
        projected_history.append(projected)
    controls: dict[str, Any] = {
        "schema_version": 1,
        "policy": "explicit",
        "selected_route": selected,
        "terminal_disposition": disposition,
        "history": projected_history,
    }
    if not finite and "execution_profile_fingerprint" in source:
        controls["execution_profile_fingerprint"] = source["execution_profile_fingerprint"]
    original = _evidence(source.get("original_access"), finite=finite)
    if original:
        controls["original_access"] = original
    if not finite and "next_eligible_at" in source:
        controls["next_eligible_at"] = source["next_eligible_at"]
    return controls


def _history(value: object, *, finite: bool) -> dict[str, Any] | None:
    source = _json_object(value)
    route = None if source is None else _route_identity(source.get("route"), finite=finite)
    if source is None or route is None or type(source.get("invoked")) is not bool:
        return None
    controls: dict[str, Any] = {"route": route}
    if not finite:
        controls["invoked"] = source["invoked"]
    access = _evidence(source.get("access"), finite=finite)
    if access:
        controls["access"] = access
    action = source.get("action")
    if action is not None:
        if action not in _ROUTE_ACTIONS:
            return None
        controls["action"] = action
    disposition = source.get("disposition")
    if disposition is not None:
        if disposition not in _ROUTE_DISPOSITIONS:
            return None
        controls["disposition"] = disposition
    if not finite and "next_eligible_at" in source:
        controls["next_eligible_at"] = source["next_eligible_at"]
    return controls


def _marker(kind: _ResultKind | None, controls: Mapping[str, Any]) -> str:
    if kind is None:
        raise AssertionError("Web-access result authority requires a result kind.")
    digest = hashlib.sha256(
        b"cayu.web-access-result.v1\0"
        + kind.encode("ascii")
        + b"\0"
        + canonical_durable_json_bytes(controls, "web_access_result_controls")
    ).hexdigest()
    return f"{_AUTHORITY_PREFIX}:{kind}:{digest}"


def _attested_controls(
    event: Event,
    structured: object,
    *,
    trust_persisted: bool,
) -> dict[str, Any] | None:
    marker = event.payload.get(WEB_ACCESS_RESULT_AUTHORITY_FIELD)
    if type(marker) is not str:
        return None
    parts = marker.split(":")
    if (
        len(parts) != 3
        or parts[0] != _AUTHORITY_PREFIX
        or parts[1] not in {"browser_session", "routing", "web_fetch"}
        or _SHA256.fullmatch(parts[2]) is None
    ):
        return None
    kind = cast("_ResultKind", parts[1])
    if not trust_persisted and not event_payload_authority_is_runtime_generated(
        event,
        field_name=WEB_ACCESS_RESULT_AUTHORITY_FIELD,
        value=marker,
    ):
        return None
    if event.tool_name != ("browser_session" if kind == "browser_session" else "web_fetch"):
        return None
    controls = _controls(structured, kind=kind, finite=False)
    return controls if controls and marker == _marker(kind, controls) else None


def _json_object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    copied = copy_durable_json_value(value, "web_access_result")
    return copied if type(copied) is dict else None


def _pick(source: Mapping[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    return {key: value for key, value in source.items() if key in keys}


def _string_control_paths(
    value: object,
    path: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if type(value) is str:
        return {path}
    if type(value) is list:
        return {observed for item in value for observed in _string_control_paths(item, path)}
    if type(value) is not dict:
        return set()
    observed_paths: set[tuple[str, ...]] = set()
    for key, item in value.items():
        if type(key) is not str:
            continue
        observed_paths.update(_string_control_paths(item, (*path, key)))
    return observed_paths


def _without_authority(event: Event) -> Event:
    if WEB_ACCESS_RESULT_AUTHORITY_FIELD not in event.payload:
        return event
    payload = dict(event.payload)
    payload.pop(WEB_ACCESS_RESULT_AUTHORITY_FIELD, None)
    return event.model_copy(update={"payload": payload})


def _result_with_controls(result: ToolResult, controls: Mapping[str, Any]) -> ToolResult:
    structured = _json_object(result.structured)
    if structured is None:
        structured = {}
    _overlay(structured, controls)
    return ToolResult(
        content=result.content,
        structured=structured,
        artifacts=result.artifacts,
        is_error=result.is_error,
    )


def _overlay(target: dict[str, Any], controls: Mapping[str, Any]) -> None:
    for key, control in controls.items():
        current = target.get(key)
        if type(control) is dict and type(current) is dict:
            _overlay(
                cast("dict[str, Any]", current),
                cast("Mapping[str, Any]", control),
            )
        elif type(control) is list and type(current) is list and len(control) == len(current):
            for index, item in enumerate(control):
                if type(item) is dict and type(current[index]) is dict:
                    _overlay(
                        cast("dict[str, Any]", current[index]),
                        cast("Mapping[str, Any]", item),
                    )
                else:
                    current[index] = copy_durable_json_value(item, key)
        else:
            target[key] = copy_durable_json_value(control, key)
