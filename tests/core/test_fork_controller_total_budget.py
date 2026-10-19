"""A host's optional total for one fork controller, across its whole lineage, on every store."""

from __future__ import annotations

import asyncio
from contextlib import aclosing, suppress
from decimal import Decimal

import pytest
from tests.core.session_store_conformance.support import (
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_fork_controller_model_runs import (
    _cap_id,
    _in_store,
    _invocation,
    _limit_reached,
    _refusal,
    _with_ledger,
)
from tests.core.test_session_fork_overlays import (
    CONTROLLER,
    FAKE_PRICES,
    HOST_AGENT,
    SUBJECT,
    ScriptedProvider,
    _calls,
    _completed,
    _controller_toolset,
    _drain,
    _fork_request,
    _host_app,
    _overlay,
    _run_trunk,
    fork_budget_limit,
)

from cayu import EventType, Message, RunRequest
from cayu.sessions.event_queries import EventQuery

__all__ = ["conformance_postgres_dsn", "session_store_case"]

# Every scripted completion costs 0.15 USD under ``FAKE_PRICES``, and each model
# step reserves that much, so a 0.30 total admits two steps across all forks.
TOTAL = "0.30"


def total_budget_limit(amount: str = TOTAL):
    return fork_budget_limit(amount)


def _total_id(app) -> str:
    from cayu.fork_controllers import fork_controller_total_budget_limits

    [limit] = fork_controller_total_budget_limits(app._agent_registry.fork_controller(CONTROLLER))
    return limit.budget_limit_id


async def _abandon(stream) -> bool:
    """Stop reading at the first generated text, before the provider reports usage."""

    async with aclosing(stream) as events:
        async for event in events:
            if event.type is EventType.MODEL_TEXT_DELTA:
                return True
    return False


class GatedProvider(ScriptedProvider):
    """Holds one request open after its first text until the test releases it."""

    def __init__(self, batches, *, hold: int) -> None:
        super().__init__(batches)
        self.hold = hold
        self.holding = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, request):
        self.requests.append(request)
        index = len(self.requests) - 1
        for position, event in enumerate(self.batches[index]):
            yield event
            if index == self.hold and position == 0:
                self.holding.set()
                await self.release.wait()


def test_a_total_is_optional_and_as_hard_as_the_per_fork_cap() -> None:
    from cayu.budgets import BudgetLimit, BudgetReservation, BudgetWindow
    from cayu.fork_controllers import ForkController, fork_controller_total_budget_limits

    def enroll(**options):
        return ForkController(
            name="compass",
            subjects=(SUBJECT,),
            agents=("finance-monthend",),
            allow_model_runs=True,
            fork_budget_limit=fork_budget_limit(),
            **options,
        )

    assert enroll().total_budget_limit is None
    assert fork_controller_total_budget_limits(enroll()) == ()
    reservation = BudgetReservation(max_input_tokens=1000, max_output_tokens=100)
    refused = {
        "must have scope='session'": BudgetLimit.usd(5, scope="run"),
        "must use action='interrupt'": BudgetLimit.usd(5, action="notify"),
        "cannot allow unpriced models": BudgetLimit.usd(5, allow_unpriced=True),
        "whole life": BudgetLimit.usd(
            5, window=BudgetWindow.rolling(seconds=3600), reservation=reservation
        ),
        "needs a reservation": BudgetLimit.usd(5),
    }
    for message, limit in refused.items():
        with pytest.raises(ValueError, match=f"^total_budget_limit .*{message}"):
            enroll(total_budget_limit=limit)
    with pytest.raises(TypeError, match="total_budget_limit must be a BudgetLimit"):
        enroll(total_budget_limit={"max_estimated_cost": "5"})
    mine = total_budget_limit("5")
    enrolled = enroll(total_budget_limit=mine)
    mine.max_estimated_cost = Decimal("5000")
    assert enrolled.total_budget_limit is not None
    assert enrolled.total_budget_limit.max_estimated_cost == Decimal("5")
    # One ledger identity for the whole lineage, kept when the amount changes.
    [first] = fork_controller_total_budget_limits(enrolled)
    [changed] = fork_controller_total_budget_limits(enroll(total_budget_limit=total_budget_limit()))
    assert first.budget_limit_id == changed.budget_limit_id


def test_the_enrollment_shows_the_total_without_its_price_book(session_store_case):
    async def scenario(store):
        app, _journal = _host_app(
            store, ScriptedProvider([]), total_budget_limit=total_budget_limit("12.50")
        )
        return app.fork_controller_access(SUBJECT).enrollment()

    enrollment = _in_store(session_store_case, scenario)
    assert enrollment.total_budget_limit is not None
    assert enrollment.total_budget_limit.model_dump() == {
        "max_estimated_cost": Decimal("12.50"),
        "currency": "USD",
    }


def test_the_total_counts_every_fork_and_session_in_the_lineage(session_store_case):
    async def scenario(store):
        toolset, _server = _controller_toolset()
        provider = ScriptedProvider(
            [
                _completed("closed"),
                _completed("host work"),
                _completed("first fork"),
                _completed("under the first fork"),
                _completed("never"),
            ]
        )
        app, _journal = _host_app(
            store, provider, toolset=toolset, total_budget_limit=total_budget_limit()
        )
        await _run_trunk(app)
        # Host spend outside the lineage never counts toward the controller's total.
        await _drain(
            app.run(
                RunRequest(
                    agent_name=HOST_AGENT,
                    session_id="host-only",
                    messages=[Message.text("user", "Close November.")],
                )
            )
        )
        controller = app.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request("fork-a")))
        await _drain(controller.fork(_fork_request("fork-b")))
        first = await _drain(controller.resume(_invocation("fork-a")))
        # A session the host starts under a fork joins the controller's lineage.
        under = await _drain(
            app.run(
                RunRequest(
                    agent_name=HOST_AGENT,
                    session_id="under-fork-a",
                    parent_session_id="fork-a",
                    messages=[Message.text("user", "Investigate.")],
                )
            )
        )
        requests_at_total = len(provider.requests)
        # The second fork has spent nothing, yet the lineage has reached its total.
        stopped = await _drain(controller.resume(_invocation("fork-b")))
        durable = [
            record.event
            for record in await store.query_events(
                EventQuery(session_id="fork-b", event_type=EventType.BUDGET_LIMIT_REACHED)
            )
        ]
        return first, under, requests_at_total, stopped, durable, provider, _total_id(app)

    first, under, requests_at_total, stopped, durable, provider, total_id = _in_store(
        session_store_case, scenario
    )
    assert first[-1].type is EventType.SESSION_COMPLETED
    # The session under the fork ran its step, which brought the lineage to its total.
    assert EventType.MODEL_COMPLETED in {event.type for event in under}
    assert _limit_reached(under)
    assert requests_at_total == 4
    [reached] = _limit_reached(stopped)
    assert reached.payload["actual"] == TOTAL
    assert durable[0].payload["budget_limit_id"] == total_id
    assert len(provider.requests) == 4


def test_concurrent_forks_cannot_jointly_overshoot_the_total(session_store_case):
    async def scenario(store, ledger):
        toolset, _server = _controller_toolset()
        provider = GatedProvider(
            [_completed("closed"), _completed("held"), _completed("never")], hold=1
        )
        app, _journal = _host_app(
            store,
            provider,
            toolset=toolset,
            total_budget_limit=total_budget_limit("0.15"),
            budget_ledger=ledger,
        )
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request("fork-a")))
        await _drain(controller.fork(_fork_request("fork-b")))
        # Fork A's request is in flight and holds its reservation; no usage is known yet.
        held = asyncio.create_task(_drain(controller.resume(_invocation("fork-a"))))
        await asyncio.wait_for(provider.holding.wait(), timeout=30)
        try:
            concurrent = await _drain(controller.resume(_invocation("fork-b")))
        finally:
            provider.release.set()
        finished = await held
        return concurrent, finished, provider

    concurrent, finished, provider = _with_ledger(session_store_case, scenario)
    # Fork B could not reserve against the total fork A was already holding.
    assert EventType.BUDGET_RESERVATION_FAILED in {event.type for event in concurrent}
    # Fork A's step finished and used the whole total.
    assert EventType.MODEL_COMPLETED in {event.type for event in finished}
    assert _limit_reached(finished)
    assert len(provider.requests) == 2  # the trunk and fork A


def test_abandoned_requests_and_restarts_keep_counting_against_the_total(session_store_case):
    async def scenario(store, ledger):
        toolset, _server = _controller_toolset()
        first, _journal = _host_app(
            store,
            ScriptedProvider([_completed("closed"), _completed("spent"), _completed("partial")]),
            toolset=toolset,
            total_budget_limit=total_budget_limit(),
            budget_ledger=ledger,
        )
        await _run_trunk(first)
        controller = first.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request("fork-a")))
        await _drain(controller.fork(_fork_request("fork-b")))
        await _drain(controller.fork(_fork_request("fork-c")))
        # Fork A spends 0.15; fork B abandons a request, which stays charged at 0.15.
        await _drain(controller.resume(_invocation("fork-a")))
        abandoned = await _abandon(controller.resume(_invocation("fork-b")))

        # The host restarts with the same total; fork C has spent nothing itself.
        toolset, _server = _controller_toolset()
        provider = ScriptedProvider([_completed("never")])
        restarted, _journal = _host_app(
            store,
            provider,
            toolset=toolset,
            total_budget_limit=total_budget_limit(),
            budget_ledger=ledger,
        )
        # The outcome checked is that no request reached the provider.
        with suppress(Exception):
            await _drain(restarted.fork_controller_access(SUBJECT).resume(_invocation("fork-c")))
        return abandoned, provider

    abandoned, provider = _with_ledger(session_store_case, scenario)
    assert abandoned is True
    assert provider.requests == []


def test_lineage_cost_reads_count_only_the_controllers_sessions(session_store_case):
    from tests.core.test_fork_controller_lineage import FORK_LINEAGE_METADATA_KEY, _write_metadata

    from cayu.events import Event

    async def scenario(store):
        toolset, _server = _controller_toolset()
        provider = ScriptedProvider([_completed(text) for text in ("trunk", "fork", "under")])
        app, _journal = _host_app(store, provider, toolset=toolset)
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request("fork-a")))
        await _drain(controller.resume(_invocation("fork-a")))
        await _drain(
            app.run(
                RunRequest(
                    agent_name=HOST_AGENT,
                    session_id="under-fork-a",
                    parent_session_id="fork-a",
                    messages=[Message.text("user", "Investigate.")],
                )
            )
        )
        # As the release before lineage markers stored it: only the parent link remains.
        under = await store.load("under-fork-a")
        metadata = dict(under.metadata)
        del metadata[FORK_LINEAGE_METADATA_KEY]
        await _write_metadata(store, session_store_case, "under-fork-a", metadata)

        async def total(owner: str | None, *pending: Event) -> Decimal:
            snapshot = await store.read_cost_accounting(
                EventQuery(), FAKE_PRICES, additional_events=pending, fork_lineage_owner=owner
            )
            assert owner is None or snapshot.cursor is None
            return snapshot.totals.total_cost

        [completed] = [
            record.event
            for record in await store.query_events(
                EventQuery(session_id="trunk", event_type=EventType.MODEL_COMPLETED)
            )
        ]

        def unpublished(session_id: str) -> Event:
            return completed.model_copy(
                update={"id": f"pending-{session_id}", "session_id": session_id}
            )

        # Fork A's step reserved against its own per-fork cap; the session under it did not.
        unreserved = await store.read_cost_accounting(
            EventQuery(),
            FAKE_PRICES,
            fork_lineage_owner=CONTROLLER,
            without_reservation_for=await _cap_id(app, "fork-a"),
        )
        with pytest.raises(ValueError, match="durable evidence only"):
            await store.read_cost_accounting(
                EventQuery(),
                FAKE_PRICES,
                additional_events=(unpublished("fork-a"),),
                without_reservation_for=await _cap_id(app, "fork-a"),
            )
        # A filtered read of one session never hands out a cursor an ordinary
        # read could adopt as its baseline.
        cursors = [
            (
                await store.read_cost_accounting(
                    EventQuery(session_id="trunk"), FAKE_PRICES, **filters
                )
            ).cursor
            for filters in (
                {"fork_lineage_owner": CONTROLLER},
                {"without_reservation_for": await _cap_id(app, "fork-a")},
            )
        ]
        return {
            "everything": await total(None),
            "lineage": await total(CONTROLLER),
            "other controller": await total("someone-else"),
            "pending in lineage": await total(CONTROLLER, unpublished("fork-a")),
            "pending outside": await total(CONTROLLER, unpublished("trunk")),
            "unreserved": unreserved.totals.total_cost,
        }, cursors

    totals, cursors = _in_store(session_store_case, scenario)
    assert totals == {
        "everything": Decimal("0.45"),
        # The fork and the unmarked session under it, which settles through its parent.
        "lineage": Decimal("0.30"),
        "other controller": Decimal("0"),
        "pending in lineage": Decimal("0.45"),
        "pending outside": Decimal("0.30"),
        "unreserved": Decimal("0.15"),
    }
    assert cursors == [None, None]


def test_a_total_set_after_the_forks_spent_fails_closed(session_store_case):
    async def scenario(store, ledger):
        toolset, _server = _controller_toolset()
        first, _journal = _host_app(
            store,
            ScriptedProvider([_completed("closed"), _completed("spent")]),
            toolset=toolset,
            budget_ledger=ledger,
        )
        await _run_trunk(first)
        controller = first.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request("fork-a")))
        await _drain(controller.fork(_fork_request("fork-b")))
        # Fork A spends 0.15 before the host has a total.
        await _drain(controller.resume(_invocation("fork-a")))

        # The host restarts with a total. The ledger holds none of fork A's
        # spend against it, so concurrent forks could pass the total; they
        # cannot run under it instead.
        toolset, _server = _controller_toolset()
        provider = ScriptedProvider([_completed("never")])
        restarted, _journal = _host_app(
            store,
            provider,
            toolset=toolset,
            total_budget_limit=total_budget_limit(),
            budget_ledger=ledger,
        )
        refused = await _refusal(
            restarted.fork_controller_access(SUBJECT).resume(_invocation("fork-b"))
        )
        return refused, provider

    refused, provider = _with_ledger(session_store_case, scenario)
    assert refused is not None and "spent before its total_budget_limit was set" in refused
    assert provider.requests == []


def test_unreserved_hosted_tool_evidence_also_fails_closed(session_store_case):
    from cayu.events import Event

    async def scenario(store, ledger):
        toolset, _server = _controller_toolset()
        first, _journal = _host_app(
            store, ScriptedProvider([_completed("closed")]), toolset=toolset, budget_ledger=ledger
        )
        await _run_trunk(first)
        controller = first.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request("fork-a")))
        await _drain(controller.fork(_fork_request("fork-b")))
        # A request fork A abandoned left provider-hosted evidence but no
        # completion, so no step counter records it.
        await store.append_event(
            "fork-a",
            Event(
                session_id="fork-a",
                type=EventType.MODEL_HOSTED_TOOL_CALL,
                payload={
                    "provider_name": "fake",
                    "model": "fake-model",
                    "model_attempt_id": "abandoned-attempt",
                    "tool_type": "web_search",
                    "call_id": "abandoned-search",
                    "status": "completed",
                },
            ),
        )

        toolset, _server = _controller_toolset()
        provider = ScriptedProvider([_completed("never")])
        restarted, _journal = _host_app(
            store,
            provider,
            toolset=toolset,
            total_budget_limit=total_budget_limit(),
            budget_ledger=ledger,
        )
        refused = await _refusal(
            restarted.fork_controller_access(SUBJECT).resume(_invocation("fork-b"))
        )
        return refused, provider

    refused, provider = _with_ledger(session_store_case, scenario)
    assert refused is not None and "spent before its total_budget_limit was set" in refused
    assert provider.requests == []


def test_a_tighter_total_reservation_bounds_each_request(session_store_case):
    from cayu.budgets import BudgetLimit, BudgetReservation

    async def scenario(store):
        toolset, _server = _controller_toolset()
        provider = ScriptedProvider([_completed("closed"), _completed("bounded")])
        app, _journal = _host_app(
            store,
            provider,
            toolset=toolset,
            # The per-fork cap reserves 500 output tokens; the total only 400.
            total_budget_limit=BudgetLimit.usd(
                "100",
                pricing=FAKE_PRICES,
                reservation=BudgetReservation(max_input_tokens=50_000, max_output_tokens=400),
            ),
        )
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request("fork-a")))
        await _drain(controller.resume(_invocation("fork-a")))
        return provider

    provider = _in_store(session_store_case, scenario)
    assert provider.requests[-1].options["fake"]["max_output_tokens"] == 400


def test_auxiliary_inference_runs_under_the_total(session_store_case):
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.providers import ModelRequest
    from cayu.sessions import ForkSessionRequest
    from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec
    from cayu.tools.inference import AuxiliaryInferencePolicy, InferenceLimits

    bounds = InferenceLimits(max_input_tokens=50_000, max_output_tokens=500, timeout_seconds=30)

    class Summarize(Tool):
        spec = ToolSpec(
            name="summarize",
            input_schema={"type": "object"},
            effect=ToolEffect.NONE,
            auxiliary_inference=AuxiliaryInferencePolicy(limits=bounds, purposes=("summary",)),
        )

        async def run(self, ctx, args):
            response = await ctx.inference.invoke(
                ModelRequest(model="fake-model", messages=[Message.text("user", "nested")]),
                purpose="summary",
                limits=bounds,
            )
            return ToolResult(content=response.text)

    async def scenario(store):
        provider = ScriptedModelProvider(
            [
                _completed("closed"),
                _calls("summarize", {}),
                _completed("summary"),
                _completed("ok"),
                _completed("again"),
            ],
            name="fake",
        )
        app, _journal = _host_app(
            store,
            provider,
            extra_tools=(Summarize(),),
            host_tools=("summarize",),
            total_budget_limit=total_budget_limit("100"),
        )
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(
            controller.fork(
                ForkSessionRequest(
                    source_session_id="trunk", session_id="fork-a", overlay=_overlay(tools=())
                )
            )
        )
        first = await _drain(controller.resume(_invocation("fork-a")))
        # Every attempt so far, the auxiliary one included, reserved against the total.
        again = await _drain(controller.resume(_invocation("fork-a")))
        return first, again, provider

    first, again, provider = _in_store(session_store_case, scenario)
    assert first[-1].type is EventType.SESSION_COMPLETED
    [result] = [e for e in first if e.type is EventType.TOOL_CALL_COMPLETED]
    assert result.payload.get("is_error") is not True
    assert again[-1].type is EventType.SESSION_COMPLETED
    assert len(provider.requests) == 5
