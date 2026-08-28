from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError
from tests._session_provenance import fixture_session_invocation

from cayu.core.events import Event, EventType
from cayu.core.messages import Message, ToolCallPart
from cayu.core.tools import ToolResult
from cayu.evals.corpus import (
    EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS,
    EvaluationEvidencePolicySpec,
    ToolArgumentsContainAssertionSpec,
    ToolResultContainsAssertionSpec,
    _content_revision,
)
from cayu.evals.evidence import AssertionEvidenceView, project_assertion_evidence_view
from cayu.evals.json_subset import JsonSubsetOutcome, compare_json_subset, equal_json_values
from cayu.evals.models import EvalOutcome, Trajectory
from cayu.evals.portable_assertions import compile_assertion_spec
from cayu.evals.portable_evaluation import evaluate_assertion_spec
from cayu.evals.published import (
    PublishedToolArgumentsContainDetail,
    PublishedToolResultContainsDetail,
)
from cayu.runtime.app import CayuApp
from cayu.runtime.sessions import Session, SessionStatus
from cayu.runtime.usage import session_usage_summary
from cayu.vaults import REDACTED_SECRET, SecretRedactor


def _trajectory(
    calls: tuple[tuple[str, dict[str, Any], ToolResult], ...],
) -> Trajectory:
    session_id = "tool-json-session"
    events: list[Event] = []
    transcript_calls: list[ToolCallPart] = []
    for index, (tool_name, arguments, result) in enumerate(calls, start=1):
        call_id = f"call-{index}"
        transcript_calls.append(
            ToolCallPart(
                tool_call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
            )
        )
        events.extend(
            (
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name=tool_name,
                    payload={"tool_call_id": call_id},
                ),
                Event(
                    type=(
                        EventType.TOOL_CALL_FAILED
                        if result.is_error
                        else EventType.TOOL_CALL_COMPLETED
                    ),
                    session_id=session_id,
                    tool_name=tool_name,
                    payload={
                        "tool_call_id": call_id,
                        "arguments_state": "finalized",
                        "effective_arguments": arguments,
                        "result": result.model_dump(mode="json"),
                    },
                ),
            )
        )
    events.append(Event(type=EventType.SESSION_COMPLETED, session_id=session_id))
    return Trajectory(
        session=Session(
            id=session_id,
            agent_name="agent",
            provider_name="fixture",
            model="fixture-model",
            invocation=fixture_session_invocation(session_id),
            status=SessionStatus.COMPLETED,
        ),
        events=tuple(events),
        transcript=(Message.tool_call(calls=transcript_calls),),
        usage_summary=session_usage_summary(session_id, events),
    )


@pytest.mark.parametrize(
    ("expected", "actual", "outcome"),
    (
        ({"query": "cayu"}, {"query": "cayu", "limit": 5}, JsonSubsetOutcome.MATCHED),
        (
            {"filters": {"team": "core"}},
            {"filters": {"team": "core", "active": True}},
            JsonSubsetOutcome.MATCHED,
        ),
        ({"ids": [1, 2]}, {"ids": [1, 2]}, JsonSubsetOutcome.MATCHED),
        ({"ids": [1]}, {"ids": [1, 2]}, JsonSubsetOutcome.MISMATCHED),
        ({"value": 1}, {"value": 1.0}, JsonSubsetOutcome.MATCHED),
        ({"value": True}, {"value": 1}, JsonSubsetOutcome.MISMATCHED),
        (
            {"selected": "ok"},
            {"selected": "ok", "ignored": REDACTED_SECRET},
            JsonSubsetOutcome.MATCHED,
        ),
        (
            {"selected": "secret"},
            {"selected": REDACTED_SECRET},
            JsonSubsetOutcome.REDACTED,
        ),
        (
            {"mismatch": 1, "redacted": 2},
            {"mismatch": 9, "redacted": REDACTED_SECRET},
            JsonSubsetOutcome.MISMATCHED,
        ),
        (
            {"redacted": 2, "mismatch": 1},
            {"redacted": REDACTED_SECRET, "mismatch": 9},
            JsonSubsetOutcome.MISMATCHED,
        ),
        (
            {"items": [1, 2]},
            {"items": [9, REDACTED_SECRET]},
            JsonSubsetOutcome.MISMATCHED,
        ),
    ),
)
def test_json_subset_semantics_are_explicit_and_deterministic(expected, actual, outcome):
    assert compare_json_subset(expected, actual) is outcome


def test_retained_json_equality_treats_redaction_as_safe_data_and_preserves_json_kinds():
    retained = {"selected": "ok", "private": REDACTED_SECRET, "count": 1}

    assert equal_json_values(retained, dict(retained))
    assert equal_json_values({"count": 1}, {"count": 1.0})
    assert not equal_json_values({"count": True}, {"count": 1})


@pytest.mark.parametrize(
    "detail_type",
    (PublishedToolArgumentsContainDetail, PublishedToolResultContainsDetail),
)
@pytest.mark.parametrize(
    ("actual", "matched", "message"),
    (
        ({"query": "other"}, True, "contradicts its retained evidence"),
        ({"query": "cayu"}, False, "contradicts its retained evidence"),
        ({"query": REDACTED_SECRET}, True, "cannot be redacted"),
    ),
)
def test_published_tool_json_details_reject_contradictory_comparisons(
    detail_type,
    actual,
    matched,
    message,
):
    with pytest.raises(ValidationError, match=message):
        detail_type(
            tool_name="search",
            occurrence=1,
            expected_subset={"query": "cayu"},
            observation_state="available",
            invocation_index=1,
            invocation_revision="sha256:" + "1" * 64,
            actual=actual,
            matched=matched,
        )


def test_tool_json_specs_enforce_bounded_objects_and_public_result_roots():
    with pytest.raises(ValidationError, match="JSON object"):
        ToolArgumentsContainAssertionSpec(
            id="arguments",
            tool_name="search",
            expected_subset=["not", "an", "object"],
        )
    with pytest.raises(ValidationError, match="encoded JSON byte limit"):
        ToolArgumentsContainAssertionSpec(
            id="arguments",
            tool_name="search",
            expected_subset={"query": "x" * 4096},
        )
    with pytest.raises(ValidationError, match="support only"):
        ToolResultContainsAssertionSpec(
            id="result",
            tool_name="search",
            expected_subset={"artifacts": []},
        )
    with pytest.raises(ValidationError, match="at least one"):
        ToolResultContainsAssertionSpec(
            id="result",
            tool_name="search",
            expected_subset={},
        )


def test_standard_evidence_publishes_redacted_arguments_but_results_require_opt_in():
    secret = "secret-token"
    trajectory = _trajectory(
        (
            (
                "search",
                {"query": secret, "limit": 5},
                ToolResult(
                    content=f"found {secret}",
                    structured={"status": "ok", "private": secret},
                ),
            ),
        )
    )
    app = CayuApp(enable_logging=False, secret_redactor=SecretRedactor(secret))

    standard = project_assertion_evidence_view(
        app,
        trajectory,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    retained = project_assertion_evidence_view(
        app,
        trajectory,
        evidence_policy=EvaluationEvidencePolicySpec.create(include_tool_results=True),
    )

    assert standard.tool_calls[0].arguments.value == {
        "query": REDACTED_SECRET,
        "limit": 5,
    }
    assert standard.tool_calls[0].result.state == "unsupported"
    assert retained.tool_calls[0].result.value == {
        "content": f"found {REDACTED_SECRET}",
        "structured": {"status": "ok", "private": REDACTED_SECRET},
        "is_error": False,
    }
    assert secret not in retained.model_dump_json()

    unsupported = evaluate_assertion_spec(
        ToolResultContainsAssertionSpec(
            id="unsupported-result",
            tool_name="search",
            expected_subset={"structured": {"status": "ok"}},
        ),
        standard,
    )
    assert unsupported.outcome is EvalOutcome.UNAVAILABLE
    assert unsupported.metadata["observation_state"] == "unsupported"


def test_tool_argument_and_result_assertions_select_a_stable_occurrence():
    trajectory = _trajectory(
        (
            (
                "search",
                {"query": "first", "filters": {"team": "core"}},
                ToolResult(content="first", structured={"count": 1}),
            ),
            (
                "search",
                {"query": "second", "filters": {"team": "runtime", "active": True}},
                ToolResult(
                    content="second",
                    structured={"count": 2, "status": "ok"},
                ),
            ),
        )
    )
    evidence = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        trajectory,
        evidence_policy=EvaluationEvidencePolicySpec.create(include_tool_results=True),
    )
    argument_spec = ToolArgumentsContainAssertionSpec(
        id="arguments",
        tool_name="search",
        occurrence=2,
        expected_subset={"query": "second", "filters": {"team": "runtime"}},
    )
    result_spec = ToolResultContainsAssertionSpec(
        id="result",
        tool_name="search",
        occurrence=2,
        expected_subset={"structured": {"status": "ok"}, "is_error": False},
    )

    argument_result = evaluate_assertion_spec(argument_spec, evidence)
    result_result = evaluate_assertion_spec(result_spec, evidence)

    assert argument_result.outcome is EvalOutcome.PASSED
    assert result_result.outcome is EvalOutcome.PASSED
    assert argument_result.metadata["invocation_index"] == 2
    assert (
        result_result.metadata["invocation_revision"] == evidence.tool_calls[1].invocation_revision
    )


def test_tool_json_assertions_distinguish_mismatch_absence_redaction_and_truncation():
    secret = "secret-token"
    app = CayuApp(enable_logging=False, secret_redactor=SecretRedactor(secret))
    evidence = project_assertion_evidence_view(
        app,
        _trajectory(
            (
                (
                    "search",
                    {"query": secret, "public": "yes"},
                    ToolResult(content="ok", structured={"ids": [1, 2]}),
                ),
                (
                    "oversized",
                    {"value": "x" * 5000},
                    ToolResult(content="ok"),
                ),
            )
        ),
        evidence_policy=EvaluationEvidencePolicySpec.create(include_tool_results=True),
    )

    mismatch = evaluate_assertion_spec(
        ToolResultContainsAssertionSpec(
            id="mismatch",
            tool_name="search",
            expected_subset={"structured": {"ids": [1]}},
        ),
        evidence,
    )
    absent = evaluate_assertion_spec(
        ToolArgumentsContainAssertionSpec(
            id="absent",
            tool_name="search",
            occurrence=2,
            expected_subset={"query": "anything"},
        ),
        evidence,
    )
    redacted = evaluate_assertion_spec(
        ToolArgumentsContainAssertionSpec(
            id="redacted",
            tool_name="search",
            expected_subset={"query": secret},
        ),
        evidence,
    )
    unrelated_redaction = evaluate_assertion_spec(
        ToolArgumentsContainAssertionSpec(
            id="public",
            tool_name="search",
            expected_subset={"public": "yes"},
        ),
        evidence,
    )
    truncated = evaluate_assertion_spec(
        ToolArgumentsContainAssertionSpec(
            id="truncated",
            tool_name="oversized",
            expected_subset={"value": "x"},
        ),
        evidence,
    )

    assert mismatch.outcome is EvalOutcome.FAILED
    assert mismatch.metadata["observation_state"] == "available"
    assert absent.outcome is EvalOutcome.FAILED
    assert absent.metadata["observation_state"] == "absent"
    assert redacted.outcome is EvalOutcome.UNAVAILABLE
    assert redacted.metadata["observation_state"] == "redacted"
    assert unrelated_redaction.outcome is EvalOutcome.PASSED
    assert truncated.outcome is EvalOutcome.UNAVAILABLE
    assert truncated.metadata["observation_state"] == "truncated"


def test_tool_json_assertions_distinguish_malformed_data_and_incompatible_identity():
    base = _trajectory((("search", {"query": "cayu"}, ToolResult(content="ok", structured={})),))
    unpublished_terminal = Event.model_validate(
        {
            **base.events[1].model_dump(mode="python"),
            "payload": {
                "tool_call_id": "call-1",
                "arguments_state": "unavailable",
                "result": base.events[1].payload["result"],
            },
        }
    )
    unpublished_trajectory = Trajectory(
        session=base.session,
        events=(base.events[0], unpublished_terminal, base.events[2]),
        transcript=base.transcript,
        usage_summary=base.usage_summary,
    )
    unpublished_evidence = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        unpublished_trajectory,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    unpublished = evaluate_assertion_spec(
        ToolArgumentsContainAssertionSpec(
            id="unpublished",
            tool_name="search",
            expected_subset={"query": "cayu"},
        ),
        unpublished_evidence,
    )
    assert unpublished.outcome is EvalOutcome.UNAVAILABLE
    assert unpublished.metadata["observation_state"] == "unavailable"

    malformed_terminal = Event.model_validate(
        {
            **base.events[1].model_dump(mode="python"),
            "payload": {
                **base.events[1].payload,
                "result": {"content": 42},
            },
        }
    )
    malformed_trajectory = Trajectory(
        session=base.session,
        events=(base.events[0], malformed_terminal, base.events[2]),
        transcript=base.transcript,
        usage_summary=base.usage_summary,
    )
    malformed_evidence = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        malformed_trajectory,
        evidence_policy=EvaluationEvidencePolicySpec.create(include_tool_results=True),
    )
    malformed = evaluate_assertion_spec(
        ToolResultContainsAssertionSpec(
            id="malformed",
            tool_name="search",
            expected_subset={"content": "ok"},
        ),
        malformed_evidence,
    )
    assert malformed.outcome is EvalOutcome.UNAVAILABLE
    assert malformed.metadata["observation_state"] == "malformed"

    incompatible_start = Event.model_validate(
        {
            **base.events[0].model_dump(mode="python"),
            "payload": {},
        }
    )
    incompatible_trajectory = Trajectory(
        session=base.session,
        events=(incompatible_start, base.events[1], base.events[2]),
        transcript=base.transcript,
        usage_summary=base.usage_summary,
    )
    incompatible_evidence = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        incompatible_trajectory,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    incompatible = evaluate_assertion_spec(
        ToolArgumentsContainAssertionSpec(
            id="incompatible",
            tool_name="search",
            expected_subset={"query": "cayu"},
        ),
        incompatible_evidence,
    )
    assert incompatible.outcome is EvalOutcome.UNAVAILABLE
    assert incompatible.metadata["observation_state"] == "incompatible"

    reordered_trajectory = Trajectory(
        session=base.session,
        events=(base.events[1], base.events[0], base.events[2]),
        transcript=base.transcript,
        usage_summary=base.usage_summary,
    )
    reordered_evidence = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        reordered_trajectory,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    reordered = evaluate_assertion_spec(
        ToolArgumentsContainAssertionSpec(
            id="reordered",
            tool_name="search",
            expected_subset={"query": "cayu"},
        ),
        reordered_evidence,
    )
    assert reordered.outcome is EvalOutcome.UNAVAILABLE
    assert reordered.metadata["observation_state"] == "incompatible"


def test_tool_call_evidence_reports_bounded_cardinality_without_ambiguous_absence():
    evidence = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        _trajectory(
            tuple(
                (
                    "search",
                    {"index": index},
                    ToolResult(content="ok", structured={"index": index}),
                )
                for index in range(EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS + 1)
            )
        ),
        evidence_policy=EvaluationEvidencePolicySpec.create(include_tool_results=True),
    )

    assert evidence.tool_call_evidence_state == "limit_exceeded"
    assert evidence.tool_calls_started == EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS + 1
    assert len(evidence.tool_calls) == EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS
    assert evidence.tool_calls[-1].occurrence == EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS


def test_compiler_rejects_assertions_incompatible_with_the_selected_policy():
    app = CayuApp(enable_logging=False)
    arguments = ToolArgumentsContainAssertionSpec(
        id="arguments",
        tool_name="search",
        expected_subset={"query": "cayu"},
    )
    result = ToolResultContainsAssertionSpec(
        id="result",
        tool_name="search",
        expected_subset={"structured": {"status": "ok"}},
    )

    with pytest.raises(ValueError, match="published argument evidence"):
        compile_assertion_spec(
            arguments,
            app=app,
            evidence_policy=EvaluationEvidencePolicySpec.create(include_tool_arguments=False),
            trusted_pricing=None,
        )
    with pytest.raises(ValueError, match="retained result evidence"):
        compile_assertion_spec(
            result,
            app=app,
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            trusted_pricing=None,
        )


def test_tool_evidence_policy_revisions_cover_every_supported_flag_combination():
    policies = tuple(
        EvaluationEvidencePolicySpec.create(
            include_tool_arguments=include_arguments,
            include_tool_results=include_results,
        )
        for include_arguments in (False, True)
        for include_results in (False, True)
    )

    assert len({policy.revision for policy in policies}) == 4
    assert EvaluationEvidencePolicySpec.supported_revisions() == frozenset(
        policy.revision for policy in policies
    )


@pytest.mark.parametrize("include_arguments", (False, True))
@pytest.mark.parametrize("include_results", (False, True))
def test_assertion_evidence_binds_tool_value_states_to_policy_revision(
    include_arguments: bool,
    include_results: bool,
):
    policy = EvaluationEvidencePolicySpec.create(
        include_tool_arguments=include_arguments,
        include_tool_results=include_results,
    )
    evidence = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        _trajectory((("search", {"query": "cayu"}, ToolResult(structured={"status": "ok"})),)),
        evidence_policy=policy,
    )

    call = evidence.tool_calls[0]
    assert call.arguments.state == ("available" if include_arguments else "unsupported")
    assert call.result.state == ("available" if include_results else "unsupported")

    contradictory_policies = (
        (
            "tool-argument",
            EvaluationEvidencePolicySpec.create(
                include_tool_arguments=not include_arguments,
                include_tool_results=include_results,
            ),
        ),
        (
            "tool-result",
            EvaluationEvidencePolicySpec.create(
                include_tool_arguments=include_arguments,
                include_tool_results=not include_results,
            ),
        ),
    )
    for evidence_area, contradictory_policy in contradictory_policies:
        forged = evidence.model_dump(mode="json")
        forged["policy_revision"] = contradictory_policy.revision
        revision_document = dict(forged)
        revision_document.pop("revision")
        forged["revision"] = _content_revision(revision_document, "assertion evidence")
        if evidence.total_tokens is not None:
            forged["total_tokens"] = int(evidence.total_tokens)
        with pytest.raises(ValidationError, match=f"{evidence_area} evidence"):
            AssertionEvidenceView.model_validate(forged)
