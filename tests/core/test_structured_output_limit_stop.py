"""A limit reached while waiting for structured output stops the run as a limit."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest

from cayu import AgentSpec, CayuApp, Message, RunRequest, SessionStatus
from cayu.budgets.base import BudgetLimit
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.budgets.run_limits import RunLimits
from cayu.context.structured_output import StructuredOutputSpec
from cayu.evals import ScriptedModelProvider, scripted_structured_output
from cayu.events import EventType
from cayu.providers import ModelStreamEvent
from cayu.sessions._pending_tool_round_reader import pending_tool_round_from_checkpoint
from cayu.sessions.outcomes import RunOutcome, run_to_completion

_USAGE = {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10}
_SPEC = StructuredOutputSpec(
    name="reading",
    json_schema={
        "type": "object",
        "properties": {"total": {"type": "integer"}},
        "required": ["total"],
    },
)


def _run(**request_fields) -> RunOutcome:
    call, _ = scripted_structured_output({"total": 12})
    completed = ModelStreamEvent.completed({"finish_reason": "tool_calls", "usage": _USAGE})

    async def scenario() -> RunOutcome:
        app = CayuApp()
        app.register_provider(
            ScriptedModelProvider(events=[[call, completed], [call, completed]]),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="model"))
        return await run_to_completion(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="s1",
                messages=[Message.text("user", "read it")],
                structured_output=_SPEC,
                **request_fields,
            ),
        )

    return asyncio.run(scenario())


def test_run_limit_during_structured_output_interrupts_at_the_limit() -> None:
    outcome = _run(limits=RunLimits(max_output_tokens=1))

    assert outcome.status is SessionStatus.INTERRUPTED
    assert outcome.error is None
    assert outcome.limit_reached == "output_tokens"
    assert outcome.interruption is not None
    assert outcome.interruption.kind == "limit"
    assert outcome.structured_output is None
    types = [event.type for event in outcome.events]
    assert "session.failed" not in types
    assert "structured_output.validated" not in types
    (call,) = outcome.tool_calls
    assert call.disposition == "failed"
    assert call.reason == "limit_reached"


def test_budget_limit_during_structured_output_interrupts_at_the_limit() -> None:
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="scripted",
                model="model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
        )
    )
    outcome = _run(
        budget_limits=(
            BudgetLimit(
                scope="session",
                max_estimated_cost=Decimal("0.00000001"),
                pricing=pricing,
            ),
        )
    )

    assert outcome.status is SessionStatus.INTERRUPTED, outcome.error
    assert outcome.limit_reached == "estimated_cost"
    assert "session.failed" not in [event.type for event in outcome.events]


def test_limit_reached_is_none_for_a_run_that_was_not_stopped_by_a_limit() -> None:
    outcome = _run()

    assert outcome.status is SessionStatus.COMPLETED
    assert outcome.limit_reached is None
    assert outcome.structured_output is not None
    assert outcome.structured_output.output == {"total": 12}


class _InvocationStub:
    """Only the fields the closed-round reader touches before the limit-skip branch."""

    def __init__(self, interaction_id: str) -> None:
        self.binding = SimpleNamespace(interaction_id=interaction_id)

    def require_runtime_authority(self) -> None:
        return None


async def _closed_limit_round(*, terminal_reason: str = "limit_reached"):
    call, _ = scripted_structured_output({"total": 12})
    completed = ModelStreamEvent.completed({"finish_reason": "tool_calls", "usage": _USAGE})
    app = CayuApp()
    app.register_provider(ScriptedModelProvider(events=[[call, completed]]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="model"))
    store = app.session_store
    stage_ids: list[str] = []
    complete = store.complete_model_completion_stage

    async def record_stage(session_id, *, stage_id, publication):
        stage_ids.append(stage_id)
        return await complete(session_id, stage_id=stage_id, publication=publication)

    store.complete_model_completion_stage = record_stage  # ty: ignore[invalid-assignment]
    outcome = await run_to_completion(
        app,
        RunRequest(
            agent_name="assistant",
            session_id="s1",
            messages=[Message.text("user", "read it")],
            structured_output=_SPEC,
            limits=RunLimits(max_output_tokens=1),
        ),
    )
    assert outcome.limit_reached == "output_tokens"
    session = await store.load("s1")
    (stage_id,) = stage_ids
    stage = await store.load_model_completion_stage("s1", stage_id)
    assert stage is not None and stage.publication is not None
    pending = pending_tool_round_from_checkpoint(
        {
            operation.key: operation.value
            for operation in stage.publication.mutation.operations
            if operation.action == "set"
        }
    )
    assert pending is not None
    round_id = pending.tool_round_id
    receipt = await store.load_runtime_publication_receipt("s1", f"tool-round:{round_id}")
    assert receipt is not None and "auxiliary" not in receipt.intent
    if terminal_reason != "limit_reached":
        query = store.query_events

        async def altered(event_query):
            records = await query(event_query)
            return [
                record.model_copy(
                    update={
                        "event": record.event.model_copy(
                            update={"payload": {**record.event.payload, "reason": terminal_reason}}
                        )
                    }
                )
                if record.event.type == EventType.TOOL_CALL_FAILED
                else record
                for record in records
            ]

        store.query_events = altered  # ty: ignore[invalid-assignment]
    return app._model_completion_recovery.load_closed_structured_output_events(
        session,
        stage,
        _InvocationStub(receipt.interaction_id),  # ty: ignore[invalid-argument-type]
        receipt,
        expected_spec=_SPEC,
    )


def test_recovery_reads_a_closed_round_a_limit_skipped_as_having_no_output() -> None:
    async def scenario():
        return await (await _closed_limit_round())

    assert asyncio.run(scenario()) == ()


def test_recovery_still_rejects_a_closed_round_without_validation_or_a_limit_skip() -> None:
    async def scenario():
        return await (await _closed_limit_round(terminal_reason="tool_error"))

    with pytest.raises(RuntimeError, match="malformed validation authority"):
        asyncio.run(scenario())
