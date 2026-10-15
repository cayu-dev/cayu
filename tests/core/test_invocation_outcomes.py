"""Terminal convenience results preserve public evidence without store reads."""

import asyncio
import json

import pytest

from cayu import Event, EventType, Message, RunOutcome, RunRequest, SessionStatus, run_to_completion
from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.budgets import UsageMetrics


def event(kind, *, interaction_id="answer", **payload):
    return Event(type=kind, session_id="session", interaction_id=interaction_id, payload=payload)


def completed(**payload):
    return event(EventType.MODEL_COMPLETED, **payload)


def tool_event(kind, call_id="call", round_id="round", **payload):
    return event(kind, tool_call_id=call_id, tool_round_id=round_id, **payload)


def test_interaction_attribution_ignores_unrelated_extension_events():
    outcome = RunOutcome.from_events(
        [
            event(EventType.MODEL_STARTED),
            event(EventType.MODEL_TEXT_DELTA, delta="answer"),
            completed(provider_name="provider", requested_model="alias", model="resolved"),
            event(EventType.SESSION_COMPLETED, interaction_id=None),
            event("custom.observation", interaction_id="unrelated"),
        ]
    )
    assert outcome.ok
    assert outcome.final_text == "answer"
    assert outcome.interaction_id == "answer"
    assert (outcome.provider_name, outcome.requested_model, outcome.model) == (
        "provider",
        "alias",
        "resolved",
    )
    assert len(outcome.events) == 5


@pytest.mark.parametrize("terminal", ["empty", "failed"])
def test_later_model_turn_clears_previous_text_and_model_identity(terminal):
    events = [
        event(EventType.MODEL_STARTED),
        event(EventType.MODEL_TEXT_DELTA, delta="old draft"),
        completed(provider_name="old", model="old"),
        event(EventType.MODEL_STARTED),
    ]
    if terminal == "empty":
        events += [completed(provider_name="new", model="new"), event(EventType.SESSION_COMPLETED)]
    else:
        events += [
            event(EventType.MODEL_TEXT_DELTA, delta="partial"),
            event(EventType.SESSION_FAILED, error="provider failed"),
        ]
    outcome = RunOutcome.from_events(events)
    assert outcome.final_text == ""
    assert outcome.model == ("new" if terminal == "empty" else None)
    assert outcome.error == (None if terminal == "empty" else "provider failed")


def test_usage_is_invocation_local_deduplicated_and_can_exceed_single_step_limits():
    metrics = UsageMetrics(
        input_tokens=MAX_DURABLE_JSON_INTEGER, total_tokens=MAX_DURABLE_JSON_INTEGER
    ).model_dump(mode="json")
    first = completed(usage_metrics=metrics)
    events = [
        first,
        Event.model_validate_json(first.model_dump_json()),
        completed(usage_metrics=metrics),
        completed(usage_normalization_failed=True, usage={"input_tokens": 100}),
        completed(usage_metrics={"input_tokens": -1}),
        event(EventType.MODEL_RETRY, usage_metrics=metrics),
        event(EventType.SESSION_COMPLETED, usage_summary={"total_tokens": 999}),
    ]
    outcome = RunOutcome.from_events(events)
    assert outcome.model_steps == 4
    assert outcome.unmeasured_model_steps == 2
    assert outcome.usage.input_tokens == 2 * MAX_DURABLE_JSON_INTEGER
    assert outcome.usage.total_tokens == 2 * MAX_DURABLE_JSON_INTEGER
    assert outcome.summary()["usage"]["total_tokens"] == str(2 * MAX_DURABLE_JSON_INTEGER)
    assert len(outcome.events) == len(events)
    later = RunOutcome.from_events([completed(usage_metrics={"total_tokens": 7})])
    assert later.usage.total_tokens == 7


def test_parallel_tool_calls_keep_start_order_and_round_identity_after_roundtrip():
    events = [
        tool_event(EventType.TOOL_CALL_STARTED, "a", arguments={"quarantined": "secret"}),
        tool_event(EventType.TOOL_CALL_STARTED, "b"),
        tool_event(
            EventType.TOOL_CALL_COMPLETED,
            "b",
            arguments_state="finalized",
            arguments={},
            result={"content": "B"},
        ),
        tool_event(
            EventType.TOOL_CALL_FAILED,
            "a",
            arguments_state="finalized",
            arguments={"x": 1},
            effective_arguments={"x": 2},
            idempotency_key="stable",
            result={"content": "A"},
        ),
        tool_event(EventType.TOOL_CALL_STARTED, "a", "later"),
        tool_event(
            EventType.TOOL_CALL_COMPLETED,
            "a",
            "later",
            arguments_state="unavailable",
            arguments={"must_not_publish": True},
            result={"content": "later"},
        ),
        tool_event(EventType.TOOL_CALL_STARTED, "incomplete"),
    ]
    outcome = RunOutcome.from_events(Event.model_validate_json(e.model_dump_json()) for e in events)
    assert outcome.tool_call_count == 4
    a, b, later, incomplete = outcome.tool_calls
    assert (a.tool_call_id, b.tool_call_id, later.tool_round_id) == ("a", "b", "later")
    assert (a.start_index, a.terminal_index) == (0, 3)
    assert a.start_event_id == events[0].id
    assert a.terminal_event_id == events[3].id
    assert a.idempotency_key == "stable"
    assert a.requested_arguments == {"x": 1}
    assert a.effective_arguments == {"x": 2}
    assert a.disposition == "failed"  # The event determines failure, not result.is_error.
    assert a.result.is_error is False
    assert b.effective_arguments == {}
    assert later.requested_arguments is None
    assert later.effective_arguments is None
    assert incomplete.disposition == "incomplete"
    assert incomplete.result is None
    events[3].payload["arguments"]["x"] = 100
    outcome.events[3].payload["result"]["content"] = "mutated"
    assert a.requested_arguments == {"x": 1}
    assert a.result.content == "A"


@pytest.mark.parametrize("kind", [EventType.TOOL_CALL_BLOCKED, EventType.TOOL_CALL_APPROVAL_DENIED])
def test_policy_denial_without_a_start_is_one_blocked_call(kind):
    terminal = tool_event(
        kind,
        denied_by="policy",
        reason="Not allowed",
        result={"content": "denied", "is_error": False},
    )
    outcome = RunOutcome.from_events([terminal, terminal])
    assert outcome.tool_call_count == 1
    call = outcome.tool_calls[0]
    assert call.disposition == "blocked"
    assert call.blocked_by == "policy"
    assert call.reason == "Not allowed"
    assert call.start_event_id is None


@pytest.mark.parametrize(
    "payload, kind, resumable",
    [
        (
            {"interruption_type": "tool_approval_required", "approval_id": "approval"},
            "tool_approval",
            False,
        ),
        (
            {"interruption_type": "user_input_required", "user_input": {"input_id": "input"}},
            "user_input",
            False,
        ),
        (
            {"interruption_type": "runtime_interrupted", "manual_recovery_required": True},
            "manual_recovery",
            False,
        ),
        ({"interruption_type": "waiting_on_child_action"}, "delegated_action", False),
        ({"interruption_type": "operator_requested"}, "operator", None),
        (
            {"interruption_type": "limit_reached", "limit": "model_steps", "resumable": True},
            "limit",
            True,
        ),
        ({"interruption_type": "future_reason"}, "unclassified", None),
    ],
)
def test_interruption_describes_evidence_without_granting_resume_authority(
    payload, kind, resumable
):
    outcome = RunOutcome.from_events([event(EventType.SESSION_INTERRUPTED, **payload)])
    assert outcome.interruption.kind == kind
    assert outcome.resumable is resumable
    if kind == "tool_approval":
        assert outcome.interruption.approval_id == "approval"
    if kind == "user_input":
        assert outcome.interruption.input_id == "input"
    assert outcome.status is SessionStatus.INTERRUPTED
    assert not outcome.ok


def test_summary_payloads_require_opt_in_and_are_detached():
    outcome = RunOutcome.from_events(
        [
            tool_event(
                EventType.TOOL_CALL_COMPLETED,
                arguments_state="finalized",
                arguments={"x": [1]},
                result={
                    "content": "result",
                    "structured": {"x": [2]},
                    "artifacts": [{"path": "a"}],
                },
            ),
            event(
                EventType.STRUCTURED_OUTPUT_VALIDATED, output={"x": [3]}, attempt=1, max_retries=0
            ),
            event(EventType.SESSION_COMPLETED),
        ]
    )
    basic = outcome.summary()
    assert "events" not in basic
    assert "structured_output" not in basic
    assert (
        not {"requested_arguments", "effective_arguments", "structured", "artifacts"}
        & basic["tool_calls"][0].keys()
    )
    full = outcome.summary(
        include_events=True, include_arguments=True, include_structured=True, include_artifacts=True
    )
    call = full["tool_calls"][0]
    assert call["structured"] == {"x": [2]}
    assert call["artifacts"] == [{"path": "a"}]
    assert full["structured_output"]["output"] == {"x": [3]}
    call["requested_arguments"]["x"].append(10)
    full["events"][0]["payload"]["result"]["structured"]["x"].append(10)
    assert outcome.tool_calls[0].requested_arguments == {"x": [1]}
    assert outcome.tool_calls[0].result.structured == {"x": [2]}
    assert not full["truncated"]


@pytest.mark.parametrize("max_bytes", [1024, 2048, 16384])
@pytest.mark.parametrize(
    "text", ["🚀" * 10000, "\n" * 10000, '"\\' * 10000], ids=["unicode", "escaped", "quotes"]
)
def test_summary_obeys_utf8_json_byte_limit_with_large_payloads(max_bytes, text):
    events = [
        event(EventType.MODEL_STARTED),
        event(EventType.MODEL_TEXT_DELTA, delta=text),
        completed(),
    ]
    events += [
        tool_event(
            EventType.TOOL_CALL_COMPLETED,
            str(i),
            arguments_state="finalized",
            arguments={"large": text},
            result={"content": text, "structured": {"x": text}},
        )
        for i in range(10)
    ]
    events += [event(EventType.SESSION_FAILED, error=text)]
    outcome = RunOutcome.from_events(events)
    summary = outcome.summary(
        max_bytes=max_bytes,
        include_events=True,
        include_arguments=True,
        include_structured=True,
        include_artifacts=True,
    )
    assert (
        len(json.dumps(summary, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        <= max_bytes
    )
    assert summary["truncated"]
    assert summary["omitted_fields"]
    assert summary["tool_call_count"] == 10
    assert len(outcome.events) == len(events)
    assert outcome.tool_calls[0].result.structured == {"x": text}


@pytest.mark.parametrize(
    "options", [{"max_bytes": 1023}, {"max_bytes": True}, {"include_events": 1}]
)
def test_summary_rejects_invalid_options(options):
    with pytest.raises((TypeError, ValueError)):
        RunOutcome.from_events([]).summary(**options)


def test_malformed_projection_keeps_nonraising_live_contract_and_stops_consumption():
    consumed = []

    class App:
        async def run(self, request):
            for value in [
                event(EventType.MODEL_STARTED),
                event(EventType.STRUCTURED_OUTPUT_VALIDATED, output={}),
                event(EventType.SESSION_COMPLETED),
            ]:
                consumed.append(value)
                yield value

    outcome = asyncio.run(
        run_to_completion(
            App(),
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
            ),
        )
    )
    assert outcome.status is SessionStatus.FAILED
    assert outcome.error_detail is not None
    assert len(consumed) == len(outcome.events) == 2
    assert "KeyError" in outcome.error
    with pytest.raises(KeyError):
        RunOutcome.from_events(consumed)


@pytest.mark.parametrize("terminal", [EventType.SESSION_COMPLETED, EventType.SESSION_INTERRUPTED])
def test_latest_terminal_clears_an_earlier_failure_message(terminal):
    outcome = RunOutcome.from_events(
        [
            event(EventType.SESSION_FAILED, error="old failure"),
            event(terminal),
        ]
    )
    assert outcome.error is None
    assert outcome.status is (
        SessionStatus.COMPLETED
        if terminal == EventType.SESSION_COMPLETED
        else SessionStatus.INTERRUPTED
    )


def test_withheld_arguments_clear_earlier_available_arguments_and_preserve_precision():
    first = tool_event(
        EventType.TOOL_CALL_COMPLETED,
        arguments_state="finalized",
        arguments={"redacted": "[REDACTED]"},
        arguments_exact=False,
        idempotency_key="[PRIVATE_EVENT_AUTHORITY]",
    )
    initial = RunOutcome.from_events([first])
    assert initial.tool_calls[0].arguments_exact is False
    assert initial.tool_calls[0].idempotency_key is None
    assert initial.summary(include_arguments=True)["tool_calls"][0]["arguments_exact"] is False
    withheld = tool_event(EventType.TOOL_CALL_COMPLETED, arguments_state="unavailable")
    final = RunOutcome.from_events([first, withheld])
    assert final.tool_call_count == 1
    assert final.tool_calls[0].requested_arguments is None
    assert final.tool_calls[0].effective_arguments is None
    assert final.tool_calls[0].arguments_exact is None
