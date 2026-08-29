from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError
from tests._session_provenance import fixture_session_invocation

from cayu.core.events import Event, EventType
from cayu.core.messages import Message, ToolCallPart
from cayu.evals._memory_attribution import eval_memory_attribution_evidence_from_trajectory
from cayu.evals.corpus import (
    EVIDENCE_MAX_FINAL_OUTPUT_CHARS,
    EvaluationEvidencePolicySpec,
    pricing_profile_identity,
)
from cayu.evals.evidence import (
    AssertionEvidenceView,
    _canonical_decimal,
    project_assertion_evidence_view,
)
from cayu.evals.memory_attribution import standard_eval_memory_attribution_bounds
from cayu.evals.models import Trajectory
from cayu.memory_attribution import MemoryAttribution, MemoryAttributionStatus
from cayu.runtime.app import CayuApp
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.sessions import Session, SessionStatus
from cayu.runtime.usage import SessionUsageSummary, UsageMetrics, session_usage_summary
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor


def _price_book(*, reverse: bool = False, input_rate: str = "1") -> PriceBook:
    prices = (
        ModelPrice.fixed(
            provider_name="provider-b",
            model="model-b",
            input_per_million=Decimal("2"),
            output_per_million=Decimal("3"),
            currency="EUR",
        ),
        ModelPrice.fixed(
            provider_name="provider-a",
            model="model-a",
            input_per_million=Decimal(input_rate),
            output_per_million=Decimal("2"),
            currency="USD",
        ),
    )
    return PriceBook(
        price_book_version="2026-08-05",
        generated_at="2026-08-05T00:00:00Z",
        prices=tuple(reversed(prices)) if reverse else prices,
    )


def _terminal_event(session_id: str, status: SessionStatus) -> Event:
    event_type = {
        SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
        SessionStatus.FAILED: EventType.SESSION_FAILED,
    }[status]
    return Event(type=event_type, session_id=session_id)


def _child(parent_id: str, *, status: SessionStatus) -> Trajectory:
    session_id = f"child-{status.value}"
    events = (_terminal_event(session_id, status),)
    return Trajectory(
        session=Session(
            id=session_id,
            agent_name="private-child-agent",
            provider_name="private-provider",
            model="private-model",
            causal_budget_id="private-budget",
            parent_session_id=parent_id,
            invocation=fixture_session_invocation(
                session_id,
                parent_session_id=parent_id,
            ),
            status=status,
        ),
        events=events,
        usage_summary=session_usage_summary(session_id, list(events)),
    )


def _trajectory(
    *,
    final_output: str = "Approved secret-token",
    children_incomplete: bool = False,
) -> Trajectory:
    session_id = "private-root-session"
    events = (
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id=session_id,
            payload={
                "provider_name": "provider-a",
                "model": "model-a",
                "usage_metrics": {
                    "provider_name": "provider-a",
                    "model": "model-a",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            },
        ),
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id=session_id,
            tool_name="lookup-secret-token",
            payload={
                "arguments": {"customer": "secret-token"},
                "tool_call_id": "private-tool-call",
            },
        ),
        Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id=session_id,
            tool_name="lookup-secret-token",
            payload={
                "arguments_state": "finalized",
                "effective_arguments": {"customer": "secret-token"},
                "tool_call_id": "private-tool-call",
            },
        ),
        _terminal_event(session_id, SessionStatus.COMPLETED),
    )
    transcript = (
        Message.tool_call(
            calls=[
                ToolCallPart(
                    tool_call_id="private-tool-call",
                    tool_name="lookup-secret-token",
                    arguments={"customer": "secret-token"},
                )
            ]
        ),
        Message.text("assistant", final_output),
    )
    return Trajectory(
        session=Session(
            id=session_id,
            agent_name="private-agent",
            provider_name="provider-a",
            model="model-a",
            causal_budget_id="private-budget",
            invocation=fixture_session_invocation(session_id),
            status=SessionStatus.COMPLETED,
        ),
        events=events,
        transcript=transcript,
        usage_summary=session_usage_summary(session_id, list(events)),
        final_output=final_output,
        children=(_child(session_id, status=SessionStatus.FAILED),),
        children_incomplete=children_incomplete,
    )


def _large_usage_trajectory() -> Trajectory:
    session_id = "large-usage-session"
    per_step_tokens = 2**62
    events = (
        *(
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
                payload={
                    "usage_metrics": {
                        "input_tokens": per_step_tokens,
                        "output_tokens": 0,
                        "total_tokens": per_step_tokens,
                    }
                },
            )
            for _ in range(2)
        ),
        _terminal_event(session_id, SessionStatus.COMPLETED),
    )
    return Trajectory(
        session=Session(
            id=session_id,
            agent_name="agent",
            provider_name="provider-a",
            model="model-a",
            invocation=fixture_session_invocation(session_id),
            status=SessionStatus.COMPLETED,
        ),
        events=events,
        usage_summary=session_usage_summary(session_id, list(events)),
    )


def test_assertion_evidence_reuses_the_authoritative_eval_memory_bounds() -> None:
    attribution = MemoryAttribution(
        status=MemoryAttributionStatus.COMPLETE,
        truncated=False,
        observed_receipt_count=0,
        observed_exposure_count=0,
        observed_item_count=0,
        omitted_receipt_count_at_least=0,
        omitted_exposure_count_at_least=0,
        omitted_item_count_at_least=0,
    )
    trajectory = _trajectory().model_copy(
        update={"children": (), "memory_attribution": attribution}
    )
    standard_bounds = standard_eval_memory_attribution_bounds()
    selected_bounds = standard_bounds.model_copy(
        update={"max_source_bytes": 1024, "max_projection_bytes": 1024}
    )
    selected = eval_memory_attribution_evidence_from_trajectory(
        trajectory,
        effective_bounds=selected_bounds,
    )

    view = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        trajectory,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        memory_attribution_evidence=selected,
    )

    assert view.memory_attribution == selected
    assert view.memory_attribution.effective_bounds == selected_bounds


def test_pricing_identity_is_canonical_and_behavior_versioned():
    first = pricing_profile_identity(_price_book())
    reordered = pricing_profile_identity(_price_book(reverse=True))
    changed = pricing_profile_identity(_price_book(input_rate="1.1"))

    assert first == reordered
    assert first.fingerprint != changed.fingerprint
    assert first.pricing_semantics_version == 1
    assert first.currencies == ("EUR", "USD")
    assert "prices" not in first.model_dump(mode="json")


def test_public_evidence_is_redacted_alias_free_and_cost_bounded():
    app = CayuApp(
        enable_logging=False,
        secret_redactor=SecretRedactor("secret-token"),
    )
    view = project_assertion_evidence_view(
        app,
        _trajectory(),
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        pricing=_price_book(),
        cost_currencies=("USD",),
    )

    assert view.root_evidence_available is True
    assert view.root_status == "completed"
    assert view.child_statuses == ("failed",)
    assert view.child_evidence_state == "complete"
    assert view.final_output == f"Approved {REDACTED_SECRET}"
    assert view.requested_tool_names == (f"lookup-{REDACTED_SECRET}",)
    assert view.started_tool_names == (f"lookup-{REDACTED_SECRET}",)
    assert view.tool_calls_started == 1
    assert view.model_steps == 1
    assert view.total_tokens == 15
    assert view.costs[0].currency == "USD"
    assert view.costs[0].model_steps == 1
    assert view.costs[0].unpriced_model_steps == 0

    encoded = view.model_dump_json()
    for forbidden in (
        "secret-token",
        "private-root-session",
        "private-tool-call",
        "private-agent",
        '"provider_name"',
        '"model"',
        '"payload"',
        '"session_id"',
    ):
        assert forbidden not in encoded
    assert view.tool_calls[0].arguments.value == {"customer": REDACTED_SECRET}
    assert view.tool_calls[0].result.state == "unsupported"


def test_evidence_revision_covers_policy_content_and_pricing():
    app = CayuApp(enable_logging=False)
    policy = EvaluationEvidencePolicySpec.standard()
    original = project_assertion_evidence_view(
        app,
        _trajectory(final_output="Approved"),
        evidence_policy=policy,
        pricing=_price_book(),
        cost_currencies=("USD",),
    )
    changed_output = project_assertion_evidence_view(
        app,
        _trajectory(final_output="Declined"),
        evidence_policy=policy,
        pricing=_price_book(),
        cost_currencies=("USD",),
    )
    changed_pricing = project_assertion_evidence_view(
        app,
        _trajectory(final_output="Approved"),
        evidence_policy=policy,
        pricing=_price_book(input_rate="1.1"),
        cost_currencies=("USD",),
    )

    assert original.revision != changed_output.revision
    assert original.revision != changed_pricing.revision
    assert original.pricing_profile_fingerprint != changed_pricing.pricing_profile_fingerprint


def test_incomplete_children_and_oversized_output_are_explicit_gaps():
    app = CayuApp(enable_logging=False)
    view = project_assertion_evidence_view(
        app,
        _trajectory(
            final_output="x" * (EVIDENCE_MAX_FINAL_OUTPUT_CHARS + 1),
            children_incomplete=True,
        ),
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )

    assert view.child_evidence_state == "unavailable"
    assert view.child_statuses == ()
    assert view.final_output_state == "limit_exceeded"
    assert len(view.final_output) == EVIDENCE_MAX_FINAL_OUTPUT_CHARS


def test_cost_evidence_requires_requested_trusted_pricing():
    app = CayuApp(enable_logging=False)
    without_pricing = project_assertion_evidence_view(
        app,
        _trajectory(),
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        cost_currencies=("USD",),
    )
    without_cost_request = project_assertion_evidence_view(
        app,
        _trajectory(),
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        pricing=_price_book(),
    )

    assert without_pricing.pricing_profile_fingerprint is None
    assert without_pricing.costs == ()
    assert without_cost_request.pricing_profile_fingerprint is None
    assert without_cost_request.costs == ()


def test_cost_currencies_share_one_detached_pricing_snapshot(monkeypatch):
    import cayu.evals.evidence as evidence_module

    pricing_ids: list[int] = []
    estimate = evidence_module._estimate_session_cost

    def capture_pricing_snapshot(**kwargs):
        pricing_ids.append(id(kwargs["pricing"]))
        return estimate(**kwargs)

    monkeypatch.setattr(
        evidence_module,
        "_estimate_session_cost",
        capture_pricing_snapshot,
    )

    view = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        _trajectory(),
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        pricing=_price_book(),
        cost_currencies=("EUR", "USD"),
    )

    assert tuple(cost.currency for cost in view.costs) == ("EUR", "USD")
    assert len(pricing_ids) == 2
    assert len(set(pricing_ids)) == 1


def test_cost_decimal_rendering_rejects_unbounded_fixed_point_expansion():
    with pytest.raises(ValueError, match="at most 128"):
        _canonical_decimal(Decimal("1e999999999"))


def test_evidence_preserves_large_exact_aggregate_usage_on_the_json_wire():
    view = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        _large_usage_trajectory(),
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )

    assert view.total_tokens == 2**63
    assert view.usage_evidence_state == "limit_exceeded"
    encoded = view.model_dump_json()
    assert f'"total_tokens":"{2**63}"' in encoded
    assert AssertionEvidenceView.model_validate_json(encoded) == view


def test_evidence_ignores_usage_without_a_durable_root():
    trajectory = Trajectory(
        usage_summary=SessionUsageSummary(
            session_id="orphan",
            model_steps=1,
            usage=UsageMetrics(total_tokens=1),
        )
    )

    view = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        trajectory,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )

    assert view.root_evidence_available is False
    assert view.root_status is None
    assert view.model_step_evidence_state == "unavailable"
    assert view.model_steps is None
    assert view.usage_evidence_state == "unavailable"
    assert view.total_tokens is None


def test_evidence_projection_rejects_noncanonical_currencies_and_forged_trajectories():
    app = CayuApp(enable_logging=False)
    trajectory = _trajectory()

    with pytest.raises(ValueError, match="uppercase"):
        project_assertion_evidence_view(
            app,
            trajectory,
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            pricing=_price_book(),
            cost_currencies=("usd",),
        )
    with pytest.raises(TypeError, match="cost_currencies must be an ordered sequence"):
        project_assertion_evidence_view(
            app,
            trajectory,
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            pricing=_price_book(),
            cost_currencies={"EUR", "USD"},
        )

    view = project_assertion_evidence_view(
        app,
        trajectory,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    unordered_view = view.model_dump(mode="python")
    unordered_view["requested_tool_names"] = set(view.requested_tool_names)
    with pytest.raises(ValidationError, match="requested_tool_names.*ordered array"):
        AssertionEvidenceView.model_validate(unordered_view)

    forged = trajectory.model_copy(update={"final_output": "forged"})
    with pytest.raises(ValueError, match="final_output must match"):
        project_assertion_evidence_view(
            app,
            forged,
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
        )


def test_evidence_rejects_stale_revision_and_numeric_schema_version():
    view = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        _trajectory(),
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    stale = view.model_dump(mode="python")
    stale["total_tokens"] = 16
    with pytest.raises(ValidationError, match="revision does not match"):
        type(view).model_validate(stale)

    wrong_version = view.model_dump(mode="python")
    wrong_version["schema_version"] = True
    with pytest.raises(ValidationError, match="integer 5"):
        type(view).model_validate(wrong_version)

    unknown_policy = view.model_dump(mode="python")
    unknown_policy["policy_revision"] = "sha256:" + "f" * 64
    unknown_policy["revision"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="policy revision is not supported"):
        type(view).model_validate(unknown_policy)


@pytest.mark.parametrize(
    "trajectory_updates",
    [
        {"transcript": (), "final_output": ""},
        {
            "events": (
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="private-root-session",
                    payload={
                        "provider_name": "provider-a",
                        "model": "model-a",
                        "usage_metrics": {
                            "provider_name": "provider-a",
                            "model": "model-a",
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="private-root-session",
                    tool_name="different-tool",
                    payload={"tool_call_id": "private-tool-call"},
                ),
                _terminal_event("private-root-session", SessionStatus.COMPLETED),
            )
        },
        {
            "events": (
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="private-root-session",
                    tool_name="lookup-secret-token",
                    payload={"tool_call_id": "different-call"},
                ),
                _terminal_event("private-root-session", SessionStatus.COMPLETED),
            )
        },
    ],
)
def test_contradictory_started_tool_evidence_is_unavailable(trajectory_updates):
    source = _trajectory()
    data = source.model_dump(mode="python")
    data.update(trajectory_updates)
    if "events" in trajectory_updates:
        data["usage_summary"] = session_usage_summary(
            "private-root-session",
            list(data["events"]),
        )
    trajectory = Trajectory.model_validate(data)
    view = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        trajectory,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )

    assert view.tool_evidence_state == "unavailable"
    assert view.requested_tool_names == ()
    assert view.started_tool_names == ()
    assert view.tool_calls_started is None


def test_requested_tool_without_a_started_event_remains_conclusive():
    source = _trajectory()
    data = source.model_dump(mode="python")
    data["events"] = tuple(
        event for event in source.events if event.type != EventType.TOOL_CALL_STARTED
    )
    data["usage_summary"] = session_usage_summary(
        "private-root-session",
        list(data["events"]),
    )
    trajectory = Trajectory.model_validate(data)
    view = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        trajectory,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )

    assert view.tool_evidence_state == "complete"
    assert view.requested_tool_names == ("lookup-secret-token",)
    assert view.started_tool_names == ()
    assert view.tool_calls_started == 0


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        (
            {"final_output_state": "limit_exceeded", "final_output": "short"},
            "bounded prefix",
        ),
        (
            {"tool_evidence_state": "complete", "tool_calls_started": None},
            "started-call count",
        ),
        (
            {
                "requested_tool_names": (),
                "started_tool_names": ("lookup-secret-token",),
            },
            "must originate",
        ),
        (
            {"model_step_evidence_state": "limit_exceeded", "model_steps": 1},
            "must exceed",
        ),
        (
            {"usage_evidence_state": "limit_exceeded"},
            "must exceed",
        ),
        ({"total_tokens": -1}, "greater than or equal"),
    ],
)
def test_evidence_rejects_impossible_completeness_metadata(updates, match):
    view = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        _trajectory(),
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    forged = {**view.model_dump(mode="python"), **updates}
    forged["revision"] = "sha256:" + "0" * 64

    with pytest.raises(ValidationError, match=match):
        type(view).model_validate(forged)
