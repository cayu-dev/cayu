"""Counts retain authoritative state/accounting without presentation joins."""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_cloud_reporting import _app, _delegating_app, _drain, _start

from cayu.budgets.pricing import default_price_book
from cayu.server import ServerConfig, create_server
from cayu.server.config import DashboardConfig
from cayu.sessions import ForkSessionRequest
from cayu.sessions._run_identity import eval_trial_scope
from cayu.sessions._supervision_projection import RELEASE_MARKER_EVENT_TYPES
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.fork_overlays import FORK_LINEAGE_METADATA_KEY
from cayu.sessions.pending_action_contracts import PendingActionIssue, PendingActionSession
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest

__all__ = ["conformance_postgres_dsn", "session_store_case"]
PATH = "/api/sessions/overview/counts"


def _client(app, *, pricing=None):
    config = ServerConfig.local_development(
        dashboard=DashboardConfig(runtime_config={} if pricing is None else {"priceBook": pricing})
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_server(app, config=config)),
        base_url="http://test",
    )


@pytest.mark.store_semantics
def test_counts_skip_presentation_reads_and_preserve_old_usage(session_store_case, monkeypatch):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await store.create(
                RunRequest(session_id="older-running", agent_name="writer", messages=[]),
                identity=SessionIdentity(provider_name="test", model="test"),
            )
            await _start(app, "older-used", "writer")
            older = await store.load("older-used")
            since = older.created_at + timedelta(microseconds=1)
            for name, agent in (
                ("done", "writer"),
                ("question", "asker"),
                ("approval", "approver"),
            ):
                await _start(app, name, agent)
            async with eval_trial_scope():
                await _start(app, "trial", "writer")
            await _drain(
                app.fork_session(ForkSessionRequest(source_session_id="done", session_id="fork"))
            )
            until = datetime.now(UTC) + timedelta(seconds=1)
            params = {"since": since.isoformat(), "until": until.isoformat()}
            pricing = default_price_book()
            calls = Counter()
            outcome = store.summarize_outcome
            events = store.query_events
            pending = store.query_pending_actions_for_sessions

            async def observed_outcome(*args, **kwargs):
                calls["outcome"] += 1
                return await outcome(*args, **kwargs)

            async def observed_events(query=None, **kwargs):
                if query is not None and query.event_types == RELEASE_MARKER_EVENT_TYPES:
                    calls["release"] += 1
                return await events(query, **kwargs)

            # Overviews and counts read pending actions a page of sessions at a time.
            async def observed_pending(session_ids, query=None, **kwargs):
                for session_id in session_ids:
                    calls[f"pending:{session_id}"] += 1
                return await pending(session_ids, query, **kwargs)

            outcomes = store.summarize_outcomes
            latest = store.query_latest_events

            async def observed_outcomes(*args, **kwargs):
                calls["outcome"] += 1
                return await outcomes(*args, **kwargs)

            async def observed_latest(session_ids, event_types):
                if tuple(event_types) == RELEASE_MARKER_EVENT_TYPES:
                    calls["release"] += 1
                return await latest(session_ids, event_types)

            monkeypatch.setattr(store, "summarize_outcome", observed_outcome)
            monkeypatch.setattr(store, "query_events", observed_events)
            monkeypatch.setattr(store, "query_pending_actions_for_sessions", observed_pending)
            # The overview page reads outcomes and releases a page at a time.
            monkeypatch.setattr(store, "summarize_outcomes", observed_outcomes)
            monkeypatch.setattr(store, "query_latest_events", observed_latest)
            async with _client(app, pricing=pricing) as client:
                # The sibling full overview still owns the detailed presentation.
                listing = await client.get("/api/sessions", params={"include": "overview"})
                assert listing.status_code == 200, listing.text
                states = {row["session_id"]: row["state"] for row in listing.json()["overviews"]}
                assert calls["outcome"] > 0 and calls["release"] > 0
                for include_eval, include_forks in (
                    (False, False),
                    (True, False),
                    (False, True),
                    (True, True),
                ):
                    selected = ["older-running", "older-used", "done", "question", "approval"]
                    if include_eval:
                        selected.append("trial")
                    if include_forks:
                        selected.append("fork")
                    query = EventQuery(session_ids=tuple(selected), since=since, until=until)
                    usage = await store.read_usage_accounting(query)
                    cost = await store.read_cost_accounting(query, pricing)
                    calls.clear()
                    response = await client.get(
                        PATH,
                        params={
                            **params,
                            "include_eval": str(include_eval).lower(),
                            "include_forks": str(include_forks).lower(),
                        },
                    )
                    assert response.status_code == 200, response.text
                    body = response.json()
                    assert response.headers["cache-control"] == "private, no-store"
                    assert body["total_count"] == str(len(selected) - 2)
                    assert body["running_now"] == "1"
                    expected_states = Counter(states[name] for name in selected[2:])
                    assert body["counts_by_state"] == {
                        state: str(expected_states[state]) for state in body["counts_by_state"]
                    }
                    assert body["tokens"] == {
                        "input": str(usage.summary.usage.input_tokens),
                        "output": str(usage.summary.usage.output_tokens),
                        "cached": str(usage.summary.usage.cache.read_tokens),
                    }
                    assert (
                        int(body["tokens"]["input"]) >= 4000
                    )  # Includes older-used's recent call.
                    assert Decimal(body["cost"]["amount"]) == cost.totals.total_cost > 0
                    assert body["cost"]["complete"] is True
                    assert calls["outcome"] == calls["release"] == 0, calls
                    assert calls["pending:trial"] == int(include_eval)
                    assert calls["pending:fork"] == int(include_forks)
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_counts_follow_exact_nested_waits_and_refuse_incomplete_leaf(
    session_store_case, monkeypatch
):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _delegating_app(store)
            await _start(app, "root", "root")
            pending = store.query_pending_actions_for_sessions

            async def incomplete(session_ids, query=None, **kwargs):
                page = await pending(session_ids, query, **kwargs)
                leaves = [
                    session
                    for session in (await store.load_sessions(session_ids)).values()
                    if session.agent_name == "leaf"
                ]
                if leaves:
                    return page.model_copy(
                        update={
                            "issues": [
                                PendingActionIssue.source_invalid(
                                    PendingActionSession.from_session(leaf)
                                )
                                for leaf in leaves
                            ]
                        }
                    )
                return page

            async with _client(app) as client:
                listing = await client.get("/api/sessions", params={"include": "overview"})
                assert listing.status_code == 200, listing.text
                assert [row["state"] for row in listing.json()["overviews"]] == [
                    "waiting_approval"
                ] * 3
                counts = await client.get(PATH)
                assert counts.status_code == 200, counts.text
                assert counts.json()["total_count"] == "3"
                assert counts.json()["counts_by_state"]["waiting_approval"] == "3"
                monkeypatch.setattr(store, "query_pending_actions_for_sessions", incomplete)
                rejected = await client.get(PATH)
                assert rejected.status_code == 503, rejected.text
                assert "total_count" not in rejected.json()
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("phase", ["selection", "accounting"])
def test_counts_fence_replacement_across_actual_await(session_store_case, monkeypatch, phase):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "reused", "writer")
            method = (
                "query_pending_actions_for_sessions"
                if phase == "selection"
                else "read_session_usage_totals"
            )
            original = getattr(store, method)
            reached, release = asyncio.Event(), asyncio.Event()

            async def paused(*args, **kwargs):
                result = await original(*args, **kwargs)
                reached.set()
                await release.wait()
                return result

            monkeypatch.setattr(store, method, paused)
            async with _client(app) as client:
                request = asyncio.create_task(client.get(PATH))
                await asyncio.wait_for(reached.wait(), 5)
                monkeypatch.setattr(store, method, original)
                await store.delete_session("reused")
                await _start(app, "reused", "writer")
                release.set()
                response = await asyncio.wait_for(request, 5)
                if phase == "selection":
                    assert response.status_code == 200, response.text
                    assert response.json()["total_count"] == "0"
                    assert response.json()["tokens"] == {"input": "0", "output": "0", "cached": "0"}
                else:
                    assert response.status_code == 503, response.text
                    assert "tokens" not in response.json()
                fresh = await client.get(PATH)
                assert fresh.status_code == 200 and fresh.json()["total_count"] == "1"
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
def test_cancelled_counts_read_does_not_publish_or_poison_next_request(
    session_store_case, monkeypatch
):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _start(app, "safe", "writer")
            original = store.read_session_usage_totals
            reached, release = asyncio.Event(), asyncio.Event()

            async def paused(*args, **kwargs):
                result = await original(*args, **kwargs)
                reached.set()
                await release.wait()
                return result

            monkeypatch.setattr(store, "read_session_usage_totals", paused)
            async with _client(app) as client:
                request = asyncio.create_task(client.get(PATH))
                await asyncio.wait_for(reached.wait(), 5)
                request.cancel()
                assert request.cancelling() == 1
                with pytest.raises(asyncio.CancelledError):
                    await request
                assert request.cancelled() and request.cancelling() == 1
                release.set()
                response = await client.get(PATH)
                assert response.status_code == 200, response.text
                assert response.json()["total_count"] == "1"
                assert response.json()["tokens"]["input"] == "1000"
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("names", [("known", "unknown"), ("unknown", "known"), ("unknown",)])
def test_counts_require_positive_unforked_lineage(session_store_case, names, monkeypatch):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            for name in names:
                await store.create(
                    RunRequest(
                        session_id=name,
                        agent_name="writer",
                        messages=[],
                    ),
                    identity=SessionIdentity(provider_name="test", model="test"),
                )
            original = store.load
            original_page = store.load_sessions

            def without_ancestry(session):
                if session is not None and session.id == "unknown":
                    # Model a store read with missing ancestry evidence. The real
                    # resolver must attempt the parent and fail closed, regardless
                    # of the trusted session immediately before or after this one.
                    metadata = dict(session.metadata)
                    metadata.pop(FORK_LINEAGE_METADATA_KEY, None)
                    return session.model_copy(
                        update={"metadata": metadata, "parent_session_id": "missing-parent"}
                    )
                return session

            async def missing_ancestry(session_id):
                return without_ancestry(await original(session_id))

            async def missing_page_ancestry(session_ids):
                loaded = await original_page(session_ids)
                return {key: without_ancestry(session) for key, session in loaded.items()}

            original_list = store.list_sessions

            async def missing_listed_ancestry(query=None):
                listed = await original_list(query)
                return listed.model_copy(
                    update={"sessions": [without_ancestry(row) for row in listed.sessions]}
                )

            monkeypatch.setattr(store, "load", missing_ancestry)
            # Counts load a store page of sessions with one read, and count the
            # window's sessions as the creation listing returns them.
            monkeypatch.setattr(store, "load_sessions", missing_page_ancestry)
            monkeypatch.setattr(store, "list_sessions", missing_listed_ancestry)
            async with _client(app) as client:
                response = await client.get(PATH)
                assert response.status_code == 200, response.text
                expected = str(int("known" in names))
                assert response.json()["total_count"] == expected
                assert response.json()["running_now"] == expected
                included = await client.get(PATH, params={"include_forks": "true"})
                assert included.status_code == 200, included.text
                assert included.json()["total_count"] == str(len(names))
                assert included.json()["running_now"] == str(len(names))
        finally:
            await _close_store(store)

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("consumer", ["counts", "overview", "cloud"])
@pytest.mark.parametrize("change", ["same", "replace_before"])
def test_delegated_pending_evidence_stays_bound_to_its_incarnation(
    session_store_case, monkeypatch, depth, consumer, change
):
    asyncio.run(
        _delegated_replacement_scenario(session_store_case, monkeypatch, depth, consumer, change)
    )


@pytest.mark.store_semantics
@pytest.mark.parametrize("depth", [1, 2])
def test_delegated_pending_read_is_fenced_after_materialization(
    session_store_case, monkeypatch, depth
):
    asyncio.run(
        _delegated_replacement_scenario(
            session_store_case, monkeypatch, depth, "counts", "replace_after"
        )
    )


@pytest.mark.store_semantics
@pytest.mark.parametrize("depth", [1, 2])
def test_concurrent_overview_reads_never_adopt_a_replacement_childs_wait(
    session_store_case, monkeypatch, depth
):
    asyncio.run(
        _delegated_replacement_scenario(
            session_store_case, monkeypatch, depth, "cloud", "replace_before", ordered=False
        )
    )


async def _delegated_replacement_scenario(
    case, monkeypatch, depth, consumer, change, *, ordered=True
):
    from tests.core.test_cloud_reporting import ANSWER, REVIEWER, _pending
    from tests.server.test_cloud_reader_http import TOKEN, _bearer, _digest

    from cayu.approvals import ToolApprovalDecision, ToolApprovalRequest, UserInputResponse
    from cayu.server.cloud_reporting import CLOUD_READER_TOKEN_SHA256_VARIABLE
    from cayu.sessions.overview import SessionOverviewReader

    monkeypatch.setenv(CLOUD_READER_TOKEN_SHA256_VARIABLE, _digest(TOKEN))
    store = await _open_store(case)
    try:
        app = _delegating_app(store)
        await _start(app, "root", "root")
        # Discover actual runtime-created identities, never forge the retained wait.
        async with _client(app) as client:
            listing = await client.get("/api/sessions", params={"include": "overview"})
            assert listing.status_code == 200, listing.text
            by_agent = {row["agent_name"]: row["id"] for row in listing.json()["sessions"]}
            target = by_agent["middle" if depth == 1 else "leaf"]
            old = await store.load(target)
            native_page = store.query_pending_actions_for_sessions
            delegated_page = SessionOverviewReader._delegated_children
            reached, release = asyncio.Event(), asyncio.Event()
            verified = False

            # Counts and overview pages walk one level of delegation for the
            # whole page at once. Catch the target's read after it is verified.
            async def observed_page_delegation(self, delegations):
                nonlocal verified
                children = await delegated_page(self, delegations)
                for child in children.values():
                    if child is not None and child.id == target:
                        assert child.instance_id == old.instance_id
                        verified = True
                return children

            async def hold(read, wanted):
                if not wanted or not verified or reached.is_set():
                    return await read()
                page = await read() if change == "replace_after" else None
                reached.set()
                await release.wait()
                return page if page is not None else await read()

            async def paused_page(session_ids, query=None, **kwargs):
                return await hold(
                    lambda: native_page(session_ids, query, **kwargs), target in session_ids
                )

            monkeypatch.setattr(
                SessionOverviewReader, "_delegated_children", observed_page_delegation
            )
            monkeypatch.setattr(store, "query_pending_actions_for_sessions", paused_page)
            path, params, headers = {
                "counts": (PATH, {}, {}),
                "overview": ("/api/sessions", {"ids": "root", "include": "overview"}, {}),
                "cloud": ("/api/cloud/v1/activity", {}, _bearer(TOKEN)),
            }[consumer]
            if consumer == "overview" and case[0] != "memory":
                # The listing above stored every overview; a write to the root makes
                # its summary row unsettled, so this read derives it again.
                await store.update_metadata("root", {"unsettle": True})
            request = asyncio.create_task(client.get(path, params=params, headers=headers))
            await asyncio.wait_for(reached.wait(), 5)
            # Do not hold the replacement writer inside the reader's test barrier.
            monkeypatch.setattr(store, "query_pending_actions_for_sessions", native_page)
            monkeypatch.setattr(SessionOverviewReader, "_delegated_children", delegated_page)
            if change != "same":
                leaf = by_agent["leaf"]
                approval = await _pending(app, leaf)
                await _drain(
                    app.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id=leaf,
                            approval_id=approval.approval_id,
                            tool_round_id=approval.round_id,
                            tool_call_id=approval.tool_call_id,
                            decision=ToolApprovalDecision.APPROVE,
                            resolved_by=REVIEWER,
                        )
                    )
                )
                question = await _pending(app, leaf)
                await _drain(
                    app.resolve_user_input(
                        UserInputResponse(
                            session_id=leaf,
                            input_id=question.input_id,
                            answer=ANSWER,
                        )
                    )
                )
                # Real completion and handoff acknowledgement make deletion legal.
                assert (await store.load("root")).status.value == "completed"
                await store.delete_session(target)
                replacement_app = _app(store)
                await _start(replacement_app, target, "asker")
                assert (await store.load(target)).instance_id != old.instance_id
            release.set()
            response = await asyncio.wait_for(request, 5)
            if consumer == "cloud":
                assert response.status_code == 200, response.text
                root = next(row for row in response.json()["runs"] if row["session_id"] == "root")
                if ordered:
                    assert root["status"] == ("waiting_approval" if change == "same" else "running")
                else:
                    # Concurrent reads may see the old child before or after its
                    # replacement, but never its replacement's question.
                    assert root["status"] in {"waiting_approval", "running"}
            elif change != "same":
                assert response.status_code == 503, response.text
                assert "total_count" not in response.json() and "overviews" not in response.json()
            else:
                assert response.status_code == 200, response.text
                if consumer == "counts":
                    assert response.json()["counts_by_state"]["waiting_approval"] == "3"
                else:
                    assert response.json()["overviews"][0]["state"] == "waiting_approval"
    finally:
        await _close_store(store)
