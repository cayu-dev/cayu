"""Invocation-local projections; no store reads or lifecycle decisions."""

from __future__ import annotations

from typing import Any

from cayu.budgets.usage import (
    add_aggregate_usage,
    build_aggregate_usage_metrics,
    usage_metrics_from_event_payload,
)
from cayu.events import Event, EventType
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.sessions._terminal_evidence import (
    _INTERRUPTION_TYPE_LIMIT_REACHED,
    _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
    _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
    _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
)
from cayu.tools.outcomes import _TOOL_CALL_TERMINAL_DISPOSITIONS, ToolCallSummary


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


class InvocationProjection:
    """Only consumed public evidence, retaining first-observed call order."""

    def __init__(self) -> None:
        self.usage = build_aggregate_usage_metrics()
        self.model_steps = self.unmeasured_model_steps = 0
        self.provider_name: str | None = None
        self.requested_model: str | None = None
        self.model: str | None = None
        self.interaction_id: str | None = None
        self.interruption: dict[str, Any] | None = None
        self.seen: set[str] = set()
        self.calls: dict[tuple[str, str | None, str], dict[str, Any]] = {}

    def observe(self, event: Event, index: int) -> bool:
        if event.id in self.seen:
            return False
        self.seen.add(event.id)
        payload = event.payload
        if (
            event.type
            in {
                EventType.MODEL_STARTED,
                EventType.MODEL_TEXT_DELTA,
                EventType.MODEL_COMPLETED,
                EventType.SESSION_STARTED,
                EventType.SESSION_RESUMED,
            }
            and event.interaction_id is not None
        ):
            self.interaction_id = event.interaction_id
        if event.type in {EventType.SESSION_STARTED, EventType.SESSION_RESUMED}:
            self.interruption = None
        if event.type == EventType.MODEL_STARTED:
            self.provider_name = self.requested_model = self.model = None
        elif event.type == EventType.MODEL_COMPLETED:
            self.provider_name = _text(payload.get("provider_name"))
            self.requested_model = _text(payload.get("requested_model"))
            self.model = _text(payload.get("model"))
            self.model_steps += 1
            try:
                metrics = usage_metrics_from_event_payload(payload)
            except (TypeError, ValueError):
                metrics = None
            if metrics is None:
                self.unmeasured_model_steps += 1
            else:
                self.usage = add_aggregate_usage(self.usage, metrics)
        elif event.type in {
            EventType.SESSION_COMPLETED,
            EventType.SESSION_FAILED,
            EventType.SESSION_INTERRUPTED,
        }:
            if event.interaction_id is not None:
                self.interaction_id = event.interaction_id
            self.interruption = (
                _interruption(payload) if event.type == EventType.SESSION_INTERRUPTED else None
            )
        if (
            event.type not in _TOOL_CALL_TERMINAL_DISPOSITIONS
            and event.type != EventType.TOOL_CALL_STARTED
        ):
            return True
        call_id = _text(payload.get("tool_call_id"))
        if call_id is None:
            return True
        round_id = _text(payload.get("tool_round_id"))
        correlation_id = _text(payload.get("tool_call_correlation_id"))
        key = (
            (event.session_id, None, correlation_id)
            if correlation_id
            else (
                event.session_id,
                round_id,
                call_id,
            )
        )
        call = self.calls.setdefault(
            key,
            {
                "session_id": event.session_id,
                "interaction_id": event.interaction_id,
                "tool_call_id": call_id,
                "correlation_id": correlation_id,
                "tool_round_id": round_id,
                "disposition": "incomplete",
            },
        )
        if event.type == EventType.TOOL_CALL_STARTED:
            # A new start is a new attempt, even when recovery retains the call
            # identity. Keep call order, but discard the prior terminal evidence.
            call.clear()
            call.update(
                session_id=event.session_id,
                interaction_id=event.interaction_id,
                tool_call_id=call_id,
                correlation_id=correlation_id,
                tool_round_id=round_id,
                disposition="incomplete",
                start_event_id=event.id,
                start_index=index,
            )
        if event.tool_name is not None:
            call["tool_name"] = event.tool_name
        if (idempotency_key := _text(payload.get("idempotency_key"))) is not None:
            call["idempotency_key"] = (
                None if idempotency_key == PRIVATE_EVENT_AUTHORITY else idempotency_key
            )
        if event.type == EventType.TOOL_CALL_STARTED:
            return True
        call.update(
            disposition=_TOOL_CALL_TERMINAL_DISPOSITIONS[EventType(event.type)],
            terminal_event_id=event.id,
            terminal_index=index,
            reason=_text(payload.get("reason")),
            blocked_by=_text(payload.get("denied_by")) or _text(payload.get("blocked_by")),
        )
        # A terminal may explicitly withhold arguments that an earlier event
        # exposed. Never reconstruct them from private state or a prior start.
        arguments = payload.get("arguments")
        available = payload.get("arguments_state") == "finalized" and isinstance(arguments, dict)
        call["requested_arguments"] = arguments if available else None
        exact = payload.get("arguments_exact")
        call["arguments_exact"] = exact if available and type(exact) is bool else None
        effective = payload.get("effective_arguments", arguments)
        call["effective_arguments"] = (
            effective if available and isinstance(effective, dict) else None
        )
        result = payload.get("result")
        call["result"] = result if isinstance(result, dict) else None
        return True

    def result(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "requested_model": self.requested_model,
            "model": self.model,
            "model_steps": self.model_steps,
            "unmeasured_model_steps": self.unmeasured_model_steps,
            "usage": self.usage,
            "tool_calls": tuple(_summary(call) for call in self.calls.values()),
            "interaction_id": self.interaction_id,
            "interruption": self.interruption,
        }


def _interruption(payload: dict[str, Any]) -> dict[str, Any]:
    source = _text(payload.get("interruption_type"))
    kind = {
        _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED: "tool_approval",
        _INTERRUPTION_TYPE_USER_INPUT_REQUIRED: "user_input",
        _INTERRUPTION_TYPE_OPERATOR_REQUESTED: "operator",
        _INTERRUPTION_TYPE_LIMIT_REACHED: "limit",
        "waiting_on_child_action": "delegated_action",
    }.get(source or "", "unclassified")
    if payload.get("manual_recovery_required") is True:
        kind = "manual_recovery"
    approval = payload.get("approval")
    approval = approval if isinstance(approval, dict) else {}
    user_input = payload.get("user_input")
    user_input = user_input if isinstance(user_input, dict) else {}
    resumable = payload.get("resumable")
    if kind in {"tool_approval", "user_input", "manual_recovery", "delegated_action"}:
        resumable = False
    return {
        "kind": kind,
        "reason": _text(payload.get("reason")) or source,
        "resumable": resumable if type(resumable) is bool else None,
        "approval_id": _text(payload.get("approval_id")) or _text(approval.get("approval_id")),
        "input_id": _text(payload.get("input_id")) or _text(user_input.get("input_id")),
        "tool_round_id": _text(payload.get("tool_round_id")),
        "tool_call_id": _text(payload.get("tool_call_id")),
        "limit": _text(payload.get("limit")),
    }


def _summary(call: dict[str, Any]) -> ToolCallSummary:
    """Validate once after consumption; malformed optional details stay unavailable.

    The raw events retain the original evidence. Derived presentation data must
    never stop a live execution or turn its successful terminal event into failure.
    """
    from pydantic import ValidationError

    try:
        return ToolCallSummary(**call)
    except ValidationError as exc:
        invalid = {error["loc"][0] for error in exc.errors(include_input=False)}
        sanitized = dict(call)
        if invalid & {"requested_arguments", "effective_arguments", "arguments_exact"}:
            sanitized.update(
                requested_arguments=None, effective_arguments=None, arguments_exact=None
            )
        if "result" in invalid:
            sanitized["result"] = None
        return ToolCallSummary(**sanitized)
