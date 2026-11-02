"""Label, parent-session and causal-budget filters on start counts and their series.

The Usage chart reads ``GET /api/sessions/overview/counts`` while its headline reads
``POST /api/usage/rollup`` with a ``session_filter``. Both must select the same
current sessions for these filters in every store.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from tests.core.session_store_conformance.support import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,
    session_store_case,
)
from tests.core.test_cloud_reporting import _app, _drain

from cayu.messages import Message
from cayu.server import ServerConfig, create_server
from cayu.sessions import ForkSessionRequest
from cayu.sessions._run_identity import eval_trial_scope
from cayu.sessions.queries import (
    LabelSelectorOperator,
    LabelSelectorRequirement,
    SessionAggregateFilter,
    SessionOrder,
    session_query_from_aggregate_filter,
)
from cayu.sessions.requests import RunRequest
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage.sqlite import SQLiteSessionStore

__all__ = ["conformance_postgres_dsn", "session_store_case"]
PATH = "/api/sessions/overview/counts"

# (query params, the same filter as a rollup session_filter, default started sessions)
CASES = [
    ({"label": "team=red"}, {"labels": {"team": "red"}}, {"root", "other"}),
    (
        {"label": ["team=red", "tier=gold"]},
        {"labels": {"team": "red", "tier": "gold"}},
        {"root"},
    ),
    (
        {"label_selector": "team in (red,blue)"},
        {"label_selectors": [{"key": "team", "operator": "in", "values": ["red", "blue"]}]},
        {"root", "child", "other"},
    ),
    (
        {"label_selector": "team notin (red)"},
        {"label_selectors": [{"key": "team", "operator": "not_in", "values": ["red"]}]},
        {"child", "plain"},
    ),
    (
        {"label_selector": "!team"},
        {"label_selectors": [{"key": "team", "operator": "not_exists"}]},
        {"plain"},
    ),
    (
        {"label_selector": ["tier", "team=red"]},
        {
            "label_selectors": [
                {"key": "tier", "operator": "exists"},
                {"key": "team", "operator": "in", "values": ["red"]},
            ]
        },
        {"root"},
    ),
    ({"parent_session_id": "root"}, {"parent_session_id": "root"}, {"child"}),
    ({"causal_budget_id": "root"}, {"causal_budget_id": "root"}, {"root", "child"}),
    (
        {"causal_budget_id": "root", "label": "team=blue"},
        {"causal_budget_id": "root", "labels": {"team": "blue"}},
        {"child"},
    ),
]


async def _run(app, session_id, labels=None, **request):
    await _drain(
        app.run(
            RunRequest(
                agent_name="writer",
                session_id=session_id,
                messages=[Message.text("user", "go")],
                labels=labels or {},
                **request,
            )
        )
    )


async def _seed(app):
    await _run(app, "root", {"team": "red", "tier": "gold"})
    await _run(app, "child", {"team": "blue"}, parent_session_id="root", causal_budget_id="root")
    await _run(app, "other", {"team": "red"})
    await _run(app, "plain")
    async with eval_trial_scope():
        await _run(app, "trial", {"team": "red"}, parent_session_id="root", causal_budget_id="root")
    await _drain(app.fork_session(ForkSessionRequest(source_session_id="root", session_id="fork")))


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=create_server(app, config=ServerConfig.local_development())
        ),
        base_url="http://test",
    )


@pytest.mark.store_semantics
def test_relation_filters_match_rollup_and_keep_start_time_rules(session_store_case):
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app = _app(store)
            await _seed(app)
            sessions = {}
            for name in ("root", "child", "other", "plain", "trial", "fork"):
                session = await store.load(name)
                assert session is not None
                sessions[name] = session
            # Forks keep the source's labels and causal budget, so only the default
            # exclusions keep them out of the started population.
            assert sessions["fork"].labels == sessions["root"].labels
            assert sessions["fork"].causal_budget_id == "root"
            assert sessions["trial"].parent_session_id == "root"
            now = datetime.now(UTC)
            window = {
                "since": (now - timedelta(days=2)).isoformat(),
                "until": (now + timedelta(minutes=1)).isoformat(),
            }
            async with _client(app) as client:
                for params, session_filter, expected in CASES:
                    query = {
                        **window,
                        **params,
                        "bucket_unit": "day",
                        "timezone": "America/Los_Angeles",
                    }
                    response = await client.get(PATH, params=query)
                    assert response.status_code == 200, (params, response.text)
                    body = response.json()
                    assert body["total_count"] == str(len(expected)), params
                    assert body["series"]["timezone"] == "America/Los_Angeles"
                    assert sum(int(b["total_count"]) for b in body["series"]["buckets"]) == len(
                        expected
                    )
                    for name in expected:
                        start = sessions[name].created_at
                        (bucket,) = [
                            b
                            for b in body["series"]["buckets"]
                            if datetime.fromisoformat(b["start_at"])
                            <= start
                            < datetime.fromisoformat(b["end_at"])
                        ]
                        assert int(bucket["total_count"]) >= 1

                    # With evals and forks included, the started population is exactly
                    # the population the rollup headline filters, since every session
                    # started inside the window.
                    everything = await client.get(
                        PATH,
                        params={**query, "include_eval": "true", "include_forks": "true"},
                    )
                    assert everything.status_code == 200, everything.text
                    rollup = await client.post(
                        "/api/usage/rollup",
                        json={
                            "start_at": window["since"],
                            "end_at": window["until"],
                            "session_filter": session_filter,
                        },
                    )
                    assert rollup.status_code == 200, rollup.text
                    matching = {
                        name
                        for name, session in sessions.items()
                        if _matches(session, session_filter)
                    }
                    assert rollup.json()["matching_session_count"] == str(len(matching))
                    assert everything.json()["total_count"] == str(len(matching)), params
                    assert expected == matching - {"trial", "fork"}
        finally:
            await _close_store(store)

    asyncio.run(scenario())


def _matches(session, session_filter) -> bool:
    from cayu.sessions.queries import _session_matches

    query = session_query_from_aggregate_filter(SessionAggregateFilter(**session_filter))
    return _session_matches(session, query)


def test_relation_filters_are_validated_without_echo_and_advertised():
    app = _app()

    async def scenario():
        async with _client(app) as client:
            contract = (await client.get("/api/contract")).json()
            surface = contract["capabilities"]["surfaces"]["session_overview_count_filters"]
            assert surface["read"]["enabled"] is True
            assert surface["mutate"]["enabled"] is False
            for params in (
                {"label": "canary-without-separator"},
                {"label": ["canary=one", "canary=two"]},
                {"label_selector": "canary in one"},
                {"label_selector": "canary,,"},
                {"parent_session_id": " canary"},
                {"causal_budget_id": ""},
                {"label": [f"canary-{index}=v" for index in range(64)]},
            ):
                response = await client.get(PATH, params=params)
                assert response.status_code == 422, (params, response.text)
                assert "canary" not in response.text

    asyncio.run(scenario())


def test_sqlite_label_filters_read_from_the_label_index(tmp_path):
    """Positive label filters drive the read from the label index instead of a session scan."""

    path = tmp_path / "plans.sqlite"

    async def create():
        await SQLiteSessionStore(path).close()

    asyncio.run(create())
    from cayu.storage._sqlite_session_queries import SQL_DIALECT

    connection = sqlite3.connect(path)
    try:
        for session_filter in (
            SessionAggregateFilter(labels={"team": "red"}),
            SessionAggregateFilter(
                label_selectors=(
                    LabelSelectorRequirement(
                        key="team", operator=LabelSelectorOperator.IN, values=("red", "blue")
                    ),
                )
            ),
            SessionAggregateFilter(
                label_selectors=(
                    LabelSelectorRequirement(key="team", operator=LabelSelectorOperator.EXISTS),
                )
            ),
        ):
            query = session_query_from_aggregate_filter(session_filter).model_copy(
                update={"limit": 200, "order_by": SessionOrder.CREATED_AT_ASC}
            )
            plan = session_store_sql.build_session_query_sql(query, dialect=SQL_DIALECT)
            details = [
                row[3]
                for row in connection.execute(
                    "EXPLAIN QUERY PLAN SELECT id FROM cayu_sessions "
                    f"{plan.page_where_sql} ORDER BY {plan.order_sql} {plan.pagination_sql}",
                    plan.page_params,
                )
            ]
            assert any("idx_cayu_session_labels_key_value_session" in d for d in details), details
            assert not any(d.startswith("SCAN cayu_sessions") for d in details), details
    finally:
        connection.close()
