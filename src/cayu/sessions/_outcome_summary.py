"""Bounded presentation exports of terminal outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cayu._validation import JsonUtf8SizeCounter, copy_json_value, json_utf8_size_within_limit
from cayu.runtime._diagnostics import bound_diagnostic_text

if TYPE_CHECKING:
    from cayu.sessions.outcomes import RunOutcome


def summarize_outcome(
    outcome: RunOutcome,
    *,
    max_bytes: int,
    include_events: bool,
    include_arguments: bool,
    include_structured: bool,
    include_artifacts: bool,
) -> dict[str, Any]:
    if type(max_bytes) is not int or not 1024 <= max_bytes <= 4_194_304:
        raise ValueError("max_bytes must be between 1024 and 4194304.")
    if any(
        type(flag) is not bool
        for flag in (
            include_events,
            include_arguments,
            include_structured,
            include_artifacts,
        )
    ):
        raise TypeError("Summary inclusion options must be bools.")
    result: dict[str, Any] = {
        "status": outcome.status.value,
        "ok": outcome.ok,
        "model_steps": outcome.model_steps,
        "unmeasured_model_steps": outcome.unmeasured_model_steps,
        "tool_call_count": outcome.tool_call_count,
        "truncated": False,
        "omitted_fields": [],
        "tool_calls": [],
    }
    omitted: set[str] = set()
    # Reserve enough for the finite set of omission labels. Nothing is silently
    # dropped, and user data never supplies an unbounded label or dictionary key.
    budget = max_bytes - 512
    counter = JsonUtf8SizeCounter(budget)
    counter.value(result)
    remaining = counter.remaining

    def reserve(value: Any, overhead: int) -> bool:
        nonlocal remaining
        if remaining < overhead:
            return False
        counter = JsonUtf8SizeCounter(remaining - overhead)
        if not counter.value(value):
            return False
        remaining = counter.remaining
        return True

    def put(container: dict[str, Any], key: str, value: Any, label: str) -> None:
        # The singleton's braces replace the parent's existing braces. Account
        # for just its member and the separator, without re-walking prior data.
        if reserve({key: value}, int(bool(container)) - 2):
            container[key] = value
        else:
            omitted.add(label)

    def append(container: list[Any], value: Any, label: str) -> bool:
        if reserve(value, int(bool(container))):
            container.append(value)
            return True
        omitted.add(label)
        return False

    def text(value: str, label: str, limit: int = 1024) -> str:
        bounded = bound_diagnostic_text(value, max_bytes=min(limit, budget // 4))
        if bounded != value:
            omitted.add(label)
        return bounded

    for key in ("session_id", "interaction_id", "provider_name", "requested_model", "model"):
        put(result, key, getattr(outcome, key), key)
    if outcome.interruption is not None:
        put(result, "interruption", outcome.interruption.model_dump(mode="json"), "interruption")
    put(result, "usage", outcome.usage.model_dump(mode="json"), "usage")
    put(result, "final_text", text(outcome.final_text, "final_text", 4096), "final_text")
    put(result, "error", None if outcome.error is None else text(outcome.error, "error"), "error")
    if outcome.tool_completion is not None:
        completion = outcome.tool_completion
        item = {
            "tool_call_id": completion.call.tool_call_id,
            "tool_name": completion.call.tool_name,
            "content": text(completion.result.content, "tool_completion"),
        }
        put(result, "tool_completion", item, "tool_completion")
        if "tool_completion" in result:
            if include_arguments:
                put(item, "arguments", completion.call.arguments, "tool_completion")
            if include_structured:
                put(item, "structured", completion.result.structured, "tool_completion")
            if include_artifacts:
                put(item, "artifacts", completion.result.artifacts, "tool_completion")
    for call in outcome.tool_calls:
        item: dict[str, Any] = {
            "tool_call_id": call.tool_call_id,
            "correlation_id": call.correlation_id,
            "tool_round_id": call.tool_round_id,
            "tool_name": call.tool_name,
            "disposition": call.disposition,
            "start_event_id": call.start_event_id,
            "terminal_event_id": call.terminal_event_id,
        }
        if not append(result["tool_calls"], item, "tool_calls"):
            break
        for key in ("reason", "blocked_by"):
            value = getattr(call, key)
            if value is not None:
                put(item, key, text(value, "tool_calls.details"), "tool_calls.details")
        if call.result is not None:
            put(
                item,
                "content",
                text(call.result.content, "tool_calls.content"),
                "tool_calls.content",
            )
            put(item, "is_error", call.result.is_error, "tool_calls.details")
            if include_structured:
                put(item, "structured", call.result.structured, "tool_calls.structured")
            if include_artifacts:
                put(item, "artifacts", call.result.artifacts, "tool_calls.artifacts")
        if include_arguments:
            for key in ("requested_arguments", "effective_arguments", "arguments_exact"):
                put(item, key, getattr(call, key), "tool_calls.arguments")
    if include_structured and outcome.structured_output is not None:
        put(
            result,
            "structured_output",
            {
                "output": outcome.structured_output.output,
                "name": outcome.structured_output.name,
                "attempt": outcome.structured_output.attempt,
                "max_retries": outcome.structured_output.max_retries,
            },
            "structured_output",
        )
    if include_events:
        put(result, "events", [], "events")
        if "events" in result:
            for event in outcome.events:
                if not append(result["events"], event.model_dump(mode="json"), "events"):
                    break
    result["truncated"] = bool(omitted)
    result["omitted_fields"] = sorted(omitted)
    # Final validation also protects against mutations to nested public models
    # between outcome construction and export.
    if not json_utf8_size_within_limit(result, max_bytes=max_bytes):
        raise ValueError("Outcome summary metadata exceeds max_bytes.")
    return copy_json_value(result, "summary")
