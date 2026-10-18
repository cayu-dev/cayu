"""The host's model-run gate for fork controllers, on every built-in session store."""

from __future__ import annotations

import asyncio
from contextlib import aclosing, suppress
from decimal import Decimal

import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_session_fork_overlays import (
    CONTROLLER,
    HOST_AGENT,
    LOOKUP,
    SUBJECT,
    ScriptedProvider,
    _calls,
    _completed,
    _controller_server,
    _controller_toolset,
    _drain,
    _fork_request,
    _host_app,
    _http,
    _overlay,
    _run_trunk,
    fork_budget_limit,
)

from cayu import EventType, Message, ResumeRequest, RunRequest
from cayu.fork_controllers import ForkControllerDenied
from cayu.sessions.event_queries import EventQuery

__all__ = ["conformance_postgres_dsn", "session_store_case"]

DENIED = "The host has not allowed this controller to run its forks."
CHILD = "compass-child"
RUN_CHILD = "compass-run-child"


def _in_store(case, scenario):
    async def run():
        store = await _open_store(case)
        try:
            return await scenario(store)
        finally:
            await _close_store(store)

    return asyncio.run(run())


def _with_ledger(case, scenario):
    """Run ``scenario(store, ledger)`` with a budget ledger every restarted app shares.

    A per-fork cap reserves through the budget ledger, so a restart needs the
    durable ledger the store's backend provides, as for any strict budget.
    """

    async def run():
        from cayu.budgets.base import InMemoryBudgetLedger
        from cayu.storage.budget_ledger import SQLiteBudgetLedger
        from cayu.storage.budget_postgres import PostgresBudgetLedger
        from cayu.storage.migrations import SchemaMode

        store_kind, tmp_path, postgres_dsn = case
        if store_kind == "memory":
            ledger = InMemoryBudgetLedger()
        elif store_kind == "sqlite":
            ledger = SQLiteBudgetLedger(tmp_path / "budget.sqlite")
        else:
            ledger = PostgresBudgetLedger(postgres_dsn, schema_mode=SchemaMode.CREATE)
        store = await _open_store(case)
        try:
            return await scenario(store, ledger)
        finally:
            await _close_store(store)
            close = getattr(ledger, "close", None)
            if close is not None:
                await close()

    return asyncio.run(run())


async def _refusal(stream) -> str | None:
    try:
        await _drain(stream)
    except ForkControllerDenied as exc:
        return str(exc)
    return None


def _invocation(session_id: str) -> ResumeRequest:
    return ResumeRequest(session_id=session_id, messages=[Message.text("user", "Investigate.")])


def _invoking_fork(source, session_id: str = RUN_CHILD):
    return _fork_request(
        session_id,
        expected_source=source,
        initial_invocation=_invocation(session_id),
        initial_dispatch_id=f"{session_id}-dispatch",
    )


def test_default_enrollment_creates_forks_but_never_runs_them(session_store_case):
    async def scenario(store):
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider([_completed("closed")])
        app, _journal = _host_app(store, provider, toolset=toolset, allow_model_runs=False)
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        server = _controller_server(app)
        source = await controller.snapshot_source("trunk")
        refused = {
            "sdk_fork": await _refusal(controller.fork(_invoking_fork(source))),
            "http_fork": await _http(
                server,
                "POST",
                "/fork-controller/fork",
                json=_invoking_fork(source).model_dump(mode="json"),
            ),
        }
        invoked_child = await store.load(RUN_CHILD)
        # A fork without an invocation still works; running it does not.
        forked = await _drain(controller.fork(_fork_request(CHILD)))
        created = await store.query_events(EventQuery(session_id=CHILD, limit=5000))
        refused["sdk_resume"] = await _refusal(controller.resume(_invocation(CHILD)))
        refused["http_resume"] = await _http(
            server,
            "POST",
            "/fork-controller/resume",
            json=_invocation(CHILD).model_dump(mode="json"),
        )
        after = await store.query_events(EventQuery(session_id=CHILD, limit=5000))
        return refused, invoked_child, forked, (created, after), provider, controller_server

    refused, invoked_child, forked, (created, after), provider, controller_server = _in_store(
        session_store_case, scenario
    )
    assert refused["sdk_fork"] == DENIED and refused["sdk_resume"] == DENIED
    for name in ("http_fork", "http_resume"):
        assert refused[name].status_code == 403
        assert refused[name].json() == {"detail": "Fork controller access denied."}
    # Nothing was created, dispatched or run for the refused requests.
    assert invoked_child is None
    assert [event.type for event in forked] == [EventType.SESSION_FORKED]
    assert [record.sequence for record in after] == [record.sequence for record in created]
    assert EventType.MODEL_STARTED not in {record.event.type for record in after}
    assert len(provider.requests) == 1  # the host's own trunk run
    assert controller_server.calls == []


def test_an_allowed_controller_runs_its_forks(session_store_case):
    async def scenario(store):
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider(
            [_completed("closed"), _calls(LOOKUP, {"q": "accruals"}), _completed("finding")]
        )
        app, _journal = _host_app(store, provider, toolset=toolset)
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        server = _controller_server(app)
        source = await controller.snapshot_source("trunk")
        invoked = await _http(
            server,
            "POST",
            "/fork-controller/fork",
            json=_invoking_fork(source).model_dump(mode="json"),
        )
        await _drain(controller.fork(_fork_request(CHILD)))
        resumed = await _drain(controller.resume(_invocation(CHILD)))
        return invoked, await store.load(RUN_CHILD), resumed, provider, controller_server

    invoked, invoked_child, resumed, provider, controller_server = _in_store(
        session_store_case, scenario
    )
    assert invoked.status_code == 200 and f"{RUN_CHILD}-dispatch" in invoked.text
    assert invoked_child is not None
    assert resumed[-1].type is EventType.SESSION_COMPLETED
    assert len(provider.requests) == 3
    assert controller_server.calls == [("lookup", {"q": "accruals"})]


def test_narrowing_the_gate_on_restart_stops_existing_forks(session_store_case):
    async def scenario(store):
        toolset, _server = _controller_toolset()
        first, _journal = _host_app(
            store, ScriptedProvider([_completed("closed"), _completed("ran")]), toolset=toolset
        )
        await _run_trunk(first)
        controller = first.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request(CHILD)))
        await _drain(controller.resume(_invocation(CHILD)))
        before = await store.query_events(EventQuery(session_id=CHILD, limit=5000))
        # The host restarts with the same enrollment but without model runs.
        toolset, restarted_server = _controller_toolset()
        provider = ScriptedProvider([])
        restarted, _journal = _host_app(store, provider, toolset=toolset, allow_model_runs=False)
        narrowed = restarted.fork_controller_access(SUBJECT)
        sdk = await _refusal(narrowed.resume(_invocation(CHILD)))
        http = await _http(
            _controller_server(restarted),
            "POST",
            "/fork-controller/resume",
            json=_invocation(CHILD).model_dump(mode="json"),
        )
        after = await store.query_events(EventQuery(session_id=CHILD, limit=5000))
        return before, after, sdk, http, provider, restarted_server

    before, after, sdk, http, provider, restarted_server = _in_store(session_store_case, scenario)
    assert EventType.MODEL_STARTED in {record.event.type for record in before}
    assert sdk == DENIED
    assert http.status_code == 403
    assert [record.sequence for record in after] == [record.sequence for record in before]
    assert provider.requests == []
    assert restarted_server.calls == []


def test_allow_model_runs_must_be_a_bool() -> None:
    from cayu.fork_controllers import ForkController

    for value in (1, "true", None):
        with pytest.raises(TypeError, match="allow_model_runs must be a bool"):
            ForkController(
                name="compass",
                subjects=(SUBJECT,),
                agents=("finance-monthend",),
                allow_model_runs=value,  # ty: ignore[invalid-argument-type]
            )


# --- The host's per-fork spend cap -------------------------------------------------
#
# Every scripted completion costs 0.15 USD under ``FAKE_PRICES``, so a 0.30 cap
# admits exactly two model steps of one fork.
CAP = "0.30"


def _looking_up(call_id: str):
    return _calls(LOOKUP, {"q": "accruals"}, call_id)


def _limit_reached(events) -> list:
    return [event for event in events if event.type is EventType.BUDGET_LIMIT_REACHED]


async def _cap_id(app, session_id: str = CHILD) -> str:
    from cayu.fork_controllers import fork_controller_budget_limits

    session = await app.session_store.load(session_id)
    controller = app._agent_registry.fork_controller(CONTROLLER)
    [limit] = fork_controller_budget_limits(controller, session_id, session.instance_id)
    return limit.budget_limit_id


def test_model_runs_require_a_hard_per_fork_cap() -> None:
    from cayu.budgets import BudgetLimit, BudgetReservation, BudgetWindow
    from cayu.fork_controllers import ForkController

    def enroll(**options):
        return ForkController(
            name="compass", subjects=(SUBJECT,), agents=("finance-monthend",), **options
        )

    reservation = BudgetReservation(max_input_tokens=1000, max_output_tokens=100)
    with pytest.raises(ValueError, match="allow_model_runs=True requires fork_budget_limit"):
        enroll(allow_model_runs=True)
    refused = {
        "must have scope='session'": BudgetLimit.usd(5, scope="run"),
        "must use action='interrupt'": BudgetLimit.usd(5, action="notify"),
        "cannot allow unpriced models": BudgetLimit.usd(5, allow_unpriced=True),
        "whole life": BudgetLimit.usd(
            5, window=BudgetWindow.rolling(seconds=3600), reservation=reservation
        ),
        # Without a reservation, an abandoned request would cost nothing.
        "needs a reservation": BudgetLimit.usd(5),
    }
    for message, limit in refused.items():
        with pytest.raises(ValueError, match=message):
            enroll(allow_model_runs=True, fork_budget_limit=limit)
    with pytest.raises(TypeError, match="fork_budget_limit must be a BudgetLimit"):
        enroll(allow_model_runs=True, fork_budget_limit={"max_estimated_cost": "5"})
    # The host keeps its own copy: changing the caller's limit later changes nothing.
    mine = fork_budget_limit("5")
    enrolled = enroll(allow_model_runs=True, fork_budget_limit=mine)
    mine.max_estimated_cost = Decimal("5000")
    assert enrolled.fork_budget_limit is not None
    assert enrolled.fork_budget_limit.max_estimated_cost == Decimal("5")
    # A cap may stay configured while model runs are off.
    assert enroll(fork_budget_limit=fork_budget_limit()).allow_model_runs is False


def test_a_fork_stops_at_its_cap_and_stays_stopped(session_store_case):
    async def scenario(store):
        toolset, controller_server = _controller_toolset()
        provider = ScriptedProvider(
            [
                _completed("closed"),
                _looking_up("call-1"),
                _looking_up("call-2"),
                _completed("never"),
            ]
        )
        app, _journal = _host_app(
            store, provider, toolset=toolset, budget_limit=fork_budget_limit(CAP)
        )
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request(CHILD)))
        capped = await _drain(controller.resume(_invocation(CHILD)))
        requests_at_cap = len(provider.requests)
        again = await _drain(controller.resume(_invocation(CHILD)))
        durable = [
            record.event
            for record in await store.query_events(
                EventQuery(session_id=CHILD, event_type=EventType.BUDGET_LIMIT_REACHED)
            )
        ]
        return (
            capped,
            again,
            durable,
            requests_at_cap,
            provider,
            controller_server,
            await _cap_id(app),
        )

    capped, again, durable, requests_at_cap, provider, controller_server, cap_id = _in_store(
        session_store_case, scenario
    )
    # Two fork steps ran (0.15 USD each). The second reached the cap, so the run
    # stopped before its tool call and before a third model request.
    assert requests_at_cap == 3
    assert controller_server.calls == [("lookup", {"q": "accruals"})]
    [reached] = _limit_reached(capped)
    # Public events hide the limit's ledger identity; the durable event names the cap.
    assert durable[0].payload["budget_limit_id"] == cap_id
    assert reached.payload["scope"] == "session"
    assert reached.payload["actual"] == "0.30"
    # A later resume stops before any model request.
    assert _limit_reached(again)
    assert len(provider.requests) == 3


def test_host_resume_and_dispatch_are_gated_and_capped(session_store_case):
    from cayu.tasks.dispatch import DispatchRequest

    def dispatch(dispatch_id: str) -> DispatchRequest:
        return DispatchRequest(
            session_id=CHILD,
            dispatch_id=dispatch_id,
            messages=[Message.text("user", "Investigate.")],
        )

    async def scenario(store, ledger):
        toolset, _server = _controller_toolset()
        first, _journal = _host_app(
            store, ScriptedProvider([_completed("closed")]), toolset=toolset, budget_ledger=ledger
        )
        await _run_trunk(first)
        await _drain(first.fork_controller_access(SUBJECT).fork(_fork_request(CHILD)))

        # Restarted without model runs: no entrance the host has runs the fork.
        toolset, _server = _controller_toolset()
        closed_provider = ScriptedProvider([])
        closed, _journal = _host_app(
            store, closed_provider, toolset=toolset, allow_model_runs=False, budget_ledger=ledger
        )
        refused = {"resume": await _refusal(closed.resume(_invocation(CHILD)))}
        try:
            await closed.dispatch(dispatch("closed-dispatch"))
        except ForkControllerDenied as exc:
            refused["dispatch"] = str(exc)
        refused["dispatch_inline"] = await _refusal(
            closed.dispatch_inline(dispatch("closed-inline"))
        )
        events = await store.query_events(EventQuery(session_id=CHILD, limit=5000))

        # Restarted with model runs under a cap: host runs are capped too.
        toolset, _server = _controller_toolset()
        capped_provider = ScriptedProvider(
            [_looking_up("call-1"), _looking_up("call-2"), _completed("never")]
        )
        capped, _journal = _host_app(
            store,
            capped_provider,
            toolset=toolset,
            budget_limit=fork_budget_limit(CAP),
            budget_ledger=ledger,
        )
        host_run = await _drain(capped.resume(_invocation(CHILD)))
        inline = await _drain(capped.dispatch_inline(dispatch("capped-inline")))
        return refused, events, closed_provider, host_run, inline, capped_provider

    refused, events, closed_provider, host_run, inline, capped_provider = _with_ledger(
        session_store_case, scenario
    )
    assert refused == dict.fromkeys(("resume", "dispatch", "dispatch_inline"), DENIED)
    assert closed_provider.requests == []
    assert EventType.MODEL_STARTED not in {record.event.type for record in events}
    assert _limit_reached(host_run)
    assert len(capped_provider.requests) == 2
    assert _limit_reached(inline)
    assert len(capped_provider.requests) == 2


def test_spend_before_a_restart_counts_against_the_cap_after_it(session_store_case):
    async def scenario(store, ledger):
        toolset, _server = _controller_toolset()
        first, _journal = _host_app(
            store,
            ScriptedProvider([_completed("closed"), _looking_up("call-1"), _completed("noted")]),
            toolset=toolset,
            budget_ledger=ledger,
        )
        await _run_trunk(first)
        controller = first.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request(CHILD)))
        before = await _drain(controller.resume(_invocation(CHILD)))

        # The host restarts and lowers the cap to what the fork already spent.
        toolset, _server = _controller_toolset()
        provider = ScriptedProvider([_completed("never")])
        restarted, _journal = _host_app(
            store,
            provider,
            toolset=toolset,
            budget_limit=fork_budget_limit(CAP),
            budget_ledger=ledger,
        )
        after = await _drain(restarted.fork_controller_access(SUBJECT).resume(_invocation(CHILD)))
        return before, after, provider

    before, after, provider = _with_ledger(session_store_case, scenario)
    assert before[-1].type is EventType.SESSION_COMPLETED
    [reached] = _limit_reached(after)
    assert reached.payload["actual"] == "0.30"
    assert provider.requests == []


def test_a_new_run_under_a_closed_controllers_fork_fails_before_any_model_request(
    session_store_case,
):
    async def scenario(store):
        toolset, _server = _controller_toolset()
        first, _journal = _host_app(
            store, ScriptedProvider([_completed("closed")]), toolset=toolset
        )
        await _run_trunk(first)
        await _drain(first.fork_controller_access(SUBJECT).fork(_fork_request(CHILD)))
        toolset, _server = _controller_toolset()
        provider = ScriptedProvider([])
        closed, _journal = _host_app(store, provider, toolset=toolset, allow_model_runs=False)
        # A session the host starts under the fork joins the controller's lineage.
        events = await _drain(
            closed.run(
                RunRequest(
                    agent_name=HOST_AGENT,
                    session_id="under-the-fork",
                    parent_session_id=CHILD,
                    messages=[Message.text("user", "Investigate.")],
                )
            )
        )
        return events, await store.load("under-the-fork"), provider

    events, child, provider = _in_store(session_store_case, scenario)
    assert events[-1].type is EventType.SESSION_FAILED
    assert child is not None and child.status.value == "failed"
    assert provider.requests == []


def test_an_unmarked_session_under_a_fork_is_gated_through_its_parents(session_store_case):
    from tests.core.test_fork_controller_lineage import FORK_LINEAGE_METADATA_KEY, _write_metadata

    async def scenario(store):
        toolset, _server = _controller_toolset()
        first, _journal = _host_app(
            store,
            ScriptedProvider([_completed("closed"), _completed("noted")]),
            toolset=toolset,
        )
        await _run_trunk(first)
        await _drain(first.fork_controller_access(SUBJECT).fork(_fork_request(CHILD)))
        await _drain(
            first.run(
                RunRequest(
                    agent_name=HOST_AGENT,
                    session_id="under-the-fork",
                    parent_session_id=CHILD,
                    messages=[Message.text("user", "Investigate.")],
                )
            )
        )
        # As the release before lineage markers stored it: only the parent link remains.
        under = await store.load("under-the-fork")
        metadata = dict(under.metadata)
        del metadata[FORK_LINEAGE_METADATA_KEY]
        await _write_metadata(store, session_store_case, "under-the-fork", metadata)

        toolset, _server = _controller_toolset()
        provider = ScriptedProvider([])
        closed, _journal = _host_app(store, provider, toolset=toolset, allow_model_runs=False)
        refused = await _refusal(closed.resume(_invocation("under-the-fork")))
        return refused, provider

    refused, provider = _in_store(session_store_case, scenario)
    assert refused == DENIED
    assert provider.requests == []


def test_abandoned_requests_count_against_the_cap(session_store_case):
    """A controller that drops its stream before usage arrives still pays the reservation."""

    async def abandon(stream) -> bool:
        # Stop reading at the first generated text, before the provider reports usage.
        async with aclosing(stream) as events:
            async for event in events:
                if event.type is EventType.MODEL_TEXT_DELTA:
                    return True
        return False

    async def scenario(store):
        toolset, _server = _controller_toolset()
        provider = ScriptedProvider([_completed("closed"), *([_completed("partial")] * 3)])
        app, _journal = _host_app(
            store, provider, toolset=toolset, budget_limit=fork_budget_limit(CAP)
        )
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request(CHILD)))
        abandoned: list[object] = []
        for _attempt in range(3):
            try:
                abandoned.append(await abandon(controller.resume(_invocation(CHILD))))
            except Exception as exc:  # the outcome is what the test records
                abandoned.append(type(exc).__name__)
        return abandoned, provider

    abandoned, provider = _in_store(session_store_case, scenario)
    # Each abandoned step stays charged at its 0.15 USD reservation, so the
    # 0.30 cap admits two of them and refuses the third before the provider.
    assert abandoned[:2] == [True, True]
    assert abandoned[2] is not True
    assert len(provider.requests) == 3  # the trunk and two fork steps


def test_auxiliary_inference_runs_under_the_cap(session_store_case):
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
        # A provider that supports managed auxiliary inference.
        provider = ScriptedModelProvider(
            [
                _completed("closed"),
                _calls("summarize", {}),
                _completed("summary"),
                _completed("ok"),
            ],
            name="fake",
        )
        app, _journal = _host_app(
            store, provider, extra_tools=(Summarize(),), host_tools=("summarize",)
        )
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(
            controller.fork(
                ForkSessionRequest(
                    source_session_id="trunk", session_id=CHILD, overlay=_overlay(tools=())
                )
            )
        )
        resumed = await _drain(controller.resume(_invocation(CHILD)))
        return resumed, provider

    resumed, provider = _in_store(session_store_case, scenario)
    assert resumed[-1].type is EventType.SESSION_COMPLETED
    [result] = [e for e in resumed if e.type is EventType.TOOL_CALL_COMPLETED]
    assert result.payload.get("is_error") is not True
    assert len(provider.requests) == 4


def test_a_hosted_tool_agent_under_a_capped_fork_fails_cleanly(session_store_case):
    from tests.core.test_session_fork_overlays import HostedSearchProvider

    from cayu import AgentSpec
    from cayu.providers.hosted import OpenAIWebSearch

    async def scenario(store):
        toolset, _server = _controller_toolset()
        provider = HostedSearchProvider([_completed("closed"), _completed("never")])
        app, _journal = _host_app(store, provider, toolset=toolset)
        app.register_agent(
            AgentSpec(name="researcher", model="fake-model"), hosted_tools=[OpenAIWebSearch()]
        )
        await _run_trunk(app)
        await _drain(app.fork_controller_access(SUBJECT).fork(_fork_request(CHILD)))
        # The host starts a web-search agent under the fork, so it joins the
        # controller's lineage and its cap, which cannot bound hosted search.
        events = await _drain(
            app.run(
                RunRequest(
                    agent_name="researcher",
                    session_id="researcher-under-the-fork",
                    parent_session_id=CHILD,
                    messages=[Message.text("user", "Search.")],
                )
            )
        )
        return events, await store.load("researcher-under-the-fork"), provider

    events, child, provider = _in_store(session_store_case, scenario)
    assert events[-1].type is EventType.SESSION_FAILED
    assert child is not None and child.status.value == "failed"
    assert len(provider.requests) == 1  # only the host's trunk


def test_model_runs_on_a_durable_store_need_a_durable_budget_ledger(tmp_path) -> None:
    from cayu.sessions.memory import InMemorySessionStore
    from cayu.storage import SQLiteSessionStore
    from cayu.storage.budget_ledger import SQLiteBudgetLedger

    def host(**options):
        return _host_app(
            SQLiteSessionStore(tmp_path / "sessions.sqlite"), ScriptedProvider([]), **options
        )

    # Sessions would outlive the in-memory ledger's reservations across a restart.
    with pytest.raises(ValueError, match="needs a durable budget ledger"):
        host(budget_ledger=None)
    host(budget_ledger=SQLiteBudgetLedger(tmp_path / "budget.sqlite"))
    # Forks that never run models reserve nothing, and an in-memory store loses
    # its sessions with its ledger, so both keep the default ledger.
    host(allow_model_runs=False)
    _host_app(InMemorySessionStore(), ScriptedProvider([]))


def test_a_fork_reusing_a_deleted_sessions_id_is_still_gated_and_capped(session_store_case):
    async def scenario(store):
        toolset, _server = _controller_toolset()
        provider = ScriptedProvider(
            [_completed("closed"), _completed("host"), *([_completed("forked")] * 3)]
        )
        app, _journal = _host_app(
            store, provider, toolset=toolset, budget_limit=fork_budget_limit("0.15")
        )
        await _run_trunk(app)
        # A host session runs in this process, then the host deletes it.
        await _drain(
            app.run(
                RunRequest(
                    agent_name=HOST_AGENT,
                    session_id="reused",
                    messages=[Message.text("user", "Close October.")],
                )
            )
        )
        await store.delete_session("reused")
        # The controller forks into the freed id; the fork is its own and capped.
        controller = app.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request("reused")))
        runs = [await _drain(controller.resume(_invocation("reused"))) for _ in range(3)]
        return runs, provider

    runs, provider = _in_store(session_store_case, scenario)
    # One step fits the 0.15 cap; every later resume stops before the provider.
    assert all(_limit_reached(run) for run in runs)
    assert len(provider.requests) == 3  # the trunk, the host session, one fork step


def test_changing_the_cap_keeps_charging_unfinished_requests(session_store_case):
    async def abandon(stream) -> None:
        async with aclosing(stream) as events:
            async for event in events:
                if event.type is EventType.MODEL_TEXT_DELTA:
                    return

    async def scenario(store, ledger):
        toolset, _server = _controller_toolset()
        first, _journal = _host_app(
            store,
            ScriptedProvider([_completed("closed"), _completed("partial")]),
            toolset=toolset,
            budget_limit=fork_budget_limit(CAP),
            budget_ledger=ledger,
        )
        await _run_trunk(first)
        controller = first.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request(CHILD)))
        # The controller abandons a request, which stays charged at 0.15 USD.
        await abandon(controller.resume(_invocation(CHILD)))

        # The host restarts and lowers the cap to that charge.
        toolset, _server = _controller_toolset()
        provider = ScriptedProvider([_completed("never")])
        restarted, _journal = _host_app(
            store,
            provider,
            toolset=toolset,
            budget_limit=fork_budget_limit("0.15"),
            budget_ledger=ledger,
        )
        # The outcome checked is that no request reached the provider.
        with suppress(Exception):
            await _drain(restarted.fork_controller_access(SUBJECT).resume(_invocation(CHILD)))
        return provider

    provider = _with_ledger(session_store_case, scenario)
    assert provider.requests == []


def test_a_new_fork_in_a_deleted_forks_id_starts_with_a_fresh_cap(session_store_case):
    async def scenario(store):
        toolset, _server = _controller_toolset()
        provider = ScriptedProvider([_completed("closed"), *([_completed("forked")] * 4)])
        app, _journal = _host_app(
            store, provider, toolset=toolset, budget_limit=fork_budget_limit(CAP)
        )
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request(CHILD)))
        for _ in range(2):
            await _drain(controller.resume(_invocation(CHILD)))
        spent = len(provider.requests)
        # The host deletes the spent fork; the controller forks into its id again.
        await store.delete_session(CHILD)
        await _drain(controller.fork(_fork_request(CHILD)))
        fresh = await _drain(controller.resume(_invocation(CHILD)))
        return spent, fresh, provider

    spent, fresh, provider = _in_store(session_store_case, scenario)
    assert spent == 3  # the trunk and two fork steps at the 0.30 cap
    # The new incarnation has its own ledger entry, so its first step runs.
    assert fresh[-1].type is EventType.SESSION_COMPLETED
    assert len(provider.requests) == 4


def test_a_capped_fork_request_stays_inside_its_reservation(session_store_case):
    async def scenario(store):
        toolset, _server = _controller_toolset()
        provider = ScriptedProvider([_completed("closed"), _completed("bounded")])
        app, _journal = _host_app(store, provider, toolset=toolset)
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request(CHILD)))
        bounded = await _drain(controller.resume(_invocation(CHILD)))
        # A prompt bigger than the 50,000 reserved input tokens never dispatches.
        huge = ResumeRequest(session_id=CHILD, messages=[Message.text("user", "x" * 60_000)])
        oversized = await _drain(controller.resume(huge))
        return bounded, oversized, provider

    bounded, oversized, provider = _in_store(session_store_case, scenario)
    assert bounded[-1].type is EventType.SESSION_COMPLETED
    # The provider was told to stop at the reserved 500 output tokens.
    assert provider.requests[-1].options["fake"]["max_output_tokens"] == 500
    assert oversized[-1].type is EventType.SESSION_FAILED
    assert len(provider.requests) == 2


def test_a_provider_that_cannot_cap_output_cannot_run_a_capped_fork(session_store_case):
    class UncappedProvider(ScriptedProvider):
        def output_token_option_path(self) -> None:
            return None

    async def scenario(store):
        toolset, _server = _controller_toolset()
        provider = UncappedProvider([_completed("closed"), _completed("never")])
        app, _journal = _host_app(store, provider, toolset=toolset)
        await _run_trunk(app)
        controller = app.fork_controller_access(SUBJECT)
        await _drain(controller.fork(_fork_request(CHILD)))
        refused = await _drain(controller.resume(_invocation(CHILD)))
        return refused, provider

    refused, provider = _in_store(session_store_case, scenario)
    assert refused[-1].type is EventType.SESSION_FAILED
    assert len(provider.requests) == 1  # only the host's own trunk, which has no cap
