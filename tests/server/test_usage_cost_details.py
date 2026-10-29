"""Cost reporting uses complete event-window candidates through the HTTP boundary."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
from fastapi.testclient import TestClient
from tests.server.server_test_support import _LOCAL_SERVER_CONFIG, _price_book_payload

from cayu.applications import CayuApp
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.server import create_server
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore

START = datetime(2026, 10, 1, tzinfo=UTC)
WINDOW = {"start_at": START.isoformat(), "end_at": "2026-10-02T00:00:00Z"}


def prices():
    result = _price_book_payload(model="cheap", input_per_million="1")
    result["prices"].extend(
        _price_book_payload(model="expensive", input_per_million="1000")["prices"]
    )
    return result


async def seed(store, *, auxiliary=False, scope=None):
    for session_id, model, tokens in (
        ("token-winner", "cheap", 1000),
        ("cost-winner", "expensive", 10),
    ):
        session_id = f"{scope}-{session_id}" if scope else session_id
        await store.create(
            RunRequest(
                labels={"test_scope": scope} if scope else {},
                session_id=session_id,
                agent_name="writer",
                title=f"Title {session_id}",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="configured", model="configured"),
        )
        await store.append_event(
            session_id,
            Event(
                session_id=session_id,
                timestamp=START,
                type=EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
                if auxiliary
                else EventType.MODEL_COMPLETED,
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": model,
                        "input_tokens": tokens,
                        "total_tokens": tokens,
                    }
                },
            ),
        )


@pytest.fixture(params=["memory", "sqlite"])
def usage_store(request, tmp_path):
    store = (
        InMemorySessionStore()
        if request.param == "memory"
        else SQLiteSessionStore(tmp_path / "costs.sqlite")
    )
    asyncio.run(seed(store))
    yield store
    if isinstance(store, SQLiteSessionStore):
        asyncio.run(store.close())


@pytest.fixture
def usage_client(usage_store):
    with TestClient(
        create_server(
            CayuApp(session_store=usage_store, enable_logging=False), config=_LOCAL_SERVER_CONFIG
        )
    ) as client:
        yield client


def test_cost_ranking_prices_all_candidates_before_top_n(usage_client, usage_store, monkeypatch):
    async def no_row_reads(*args, **kwargs):
        raise AssertionError("Cost rows must use the native snapshot, not per-session reads.")

    monkeypatch.setattr(usage_store, "load", no_row_reads)
    response = usage_client.post(
        "/api/usage/rollup",
        json={**WINDOW, "session_group_limit": 1, "session_order": "cost", "pricing": prices()},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_ranking"] == {
        "requested_order": "cost",
        "effective_order": "cost",
        "unavailable_reason": None,
        "currency": "USD",
    }
    winner = body["session_breakdown"]["groups"][0]
    assert (winner["session_id"], winner["agent_name"], winner["title"]) == (
        "cost-winner",
        "writer",
        "Title cost-winner",
    )
    assert Decimal(
        body["session_cost_breakdown"]["groups"][0]["cost"]["currencies"][0]["total_cost"]
    ) == Decimal(".01")
    model_costs = {
        group["model"]: group["cost"] for group in body["model_cost_breakdown"]["groups"]
    }
    assert Decimal(model_costs["cheap"]["currencies"][0]["total_cost"]) == Decimal(".001")
    assert Decimal(model_costs["expensive"]["currencies"][0]["total_cost"]) == Decimal(".01")
    assert Decimal(
        body["provider_cost_breakdown"]["groups"][0]["cost"]["currencies"][0]["total_cost"]
    ) == Decimal(".011")


@pytest.mark.parametrize("pricing", [None, _price_book_payload(model="cheap")])
def test_unavailable_cost_ranking_is_explicit_token_fallback(usage_client, pricing):
    response = usage_client.post(
        "/api/usage/rollup",
        json={**WINDOW, "session_group_limit": 1, "session_order": "cost", "pricing": pricing},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_ranking"]["requested_order"] == "cost"
    assert body["session_ranking"]["effective_order"] == "tokens"
    assert body["session_ranking"]["unavailable_reason"]
    assert body["session_breakdown"]["groups"][0]["session_id"] == "token-winner"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_auxiliary_only_usage_is_valid_session_activity(backend, tmp_path):
    store = (
        InMemorySessionStore()
        if backend == "memory"
        else SQLiteSessionStore(tmp_path / "aux.sqlite")
    )
    asyncio.run(seed(store, auxiliary=True))
    try:
        with TestClient(
            create_server(
                CayuApp(session_store=store, enable_logging=False), config=_LOCAL_SERVER_CONFIG
            ),
            raise_server_exceptions=False,
        ) as client:
            response = client.post(
                "/api/usage/rollup", json={**WINDOW, "session_group_limit": 1, "pricing": prices()}
            )
            assert response.status_code == 200, response.text
            assert (
                response.json()["session_breakdown"]["groups"][0]["totals"]["session_count"] == "1"
            )
    finally:
        if isinstance(store, SQLiteSessionStore):
            asyncio.run(store.close())


@pytest.mark.parametrize("mode", ["mixed", "zero", "bounded_inputs"])
def test_cost_order_comparison_conditions(usage_client, mode):
    pricing = prices()
    body = {**WINDOW, "session_group_limit": 1, "session_order": "cost", "pricing": pricing}
    if mode == "mixed":
        pricing["prices"][1]["schedules"][0]["pricing"]["currency"] = "EUR"
    elif mode == "zero":
        for price in pricing["prices"]:
            price["schedules"][0]["pricing"]["standard"][0]["input_per_million"] = "0"
    else:
        body["pricing_input_limit"] = 1
    response = usage_client.post("/api/usage/rollup", json=body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["session_breakdown"]["groups"][0]["session_id"] == "token-winner"
    assert data["session_ranking"]["effective_order"] == ("cost" if mode == "zero" else "tokens")
    assert (
        data["session_ranking"]["unavailable_reason"]
        == {
            "mixed": "mixed_currencies",
            "zero": None,
            "bounded_inputs": "pricing_projection_incomplete",
        }[mode]
    )
    if mode == "zero":
        assert (
            Decimal(
                data["session_cost_breakdown"]["groups"][0]["cost"]["currencies"][0]["total_cost"]
            )
            == 0
        )
    elif mode == "mixed":
        assert {row["currency"] for row in data["cost"]["currencies"]} == {"USD", "EUR"}


def test_cost_groups_preserve_exact_other_remainder(usage_client):
    response = usage_client.post(
        "/api/usage/rollup", json={**WINDOW, "group_limit": 1, "pricing": prices()}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    groups = data["model_cost_breakdown"]
    assert groups["groups"][0]["model"] == "cheap"
    assert Decimal(groups["remainder"]["cost"]["currencies"][0]["total_cost"]) == Decimal(".01")
    assert groups["remainder"]["group_count"] == "1"
    assert data["session_breakdown"] is None


@pytest.mark.parametrize("count", [1000, 1001])
def test_global_candidate_limit_is_truthful_and_wire_rows_stay_bounded(
    usage_client, usage_store, count
):
    async def add():
        for i in range(count - 2):
            await usage_store.create(
                RunRequest(
                    session_id=f"empty-{i}",
                    agent_name="writer",
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="fake", model="cheap"),
            )

    asyncio.run(add())
    response = usage_client.post(
        "/api/usage/rollup",
        json={**WINDOW, "session_group_limit": 1, "session_order": "cost", "pricing": prices()},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    ranking = data["session_ranking"]
    assert ranking["effective_order"] == ("cost" if count == 1000 else "tokens")
    assert ranking["unavailable_reason"] == (None if count == 1000 else "candidate_limit")
    assert len(data["session_breakdown"]["groups"]) == 1
    assert int(data["session_breakdown"]["remainder"]["group_count"]) == count - 1


def test_oversized_nonwinner_candidate_retains_valid_token_fallback(usage_client, usage_store):
    asyncio.run(
        usage_store.create(
            RunRequest(
                session_id="x" * 1025, agent_name="writer", messages=[Message.text("user", "hello")]
            ),
            identity=SessionIdentity(provider_name="fake", model="cheap"),
        )
    )
    response = usage_client.post(
        "/api/usage/rollup",
        json={**WINDOW, "session_group_limit": 1, "session_order": "cost", "pricing": prices()},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["session_ranking"]["unavailable_reason"] == "candidate_byte_limit"
    assert data["session_ranking"]["effective_order"] == "tokens"
    assert data["session_breakdown"]["groups"][0]["session_id"] == "token-winner"


@pytest.mark.parametrize("status", ["completed", "outcome_unknown", "in_progress"])
def test_hosted_only_cost_candidates_and_nonterminal_exclusion(usage_client, usage_store, status):
    async def add():
        await usage_store.create(
            RunRequest(
                session_id="hosted",
                agent_name="searcher",
                title="Search",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="configured", model="configured"),
        )
        await usage_store.append_event(
            "hosted",
            Event(
                session_id="hosted",
                timestamp=START,
                type=EventType.MODEL_HOSTED_TOOL_CALL,
                payload={
                    "tool_type": "web_search",
                    "call_id": "search-call",
                    "status": status,
                    "provider_name": "fake",
                    "model": "expensive",
                },
            ),
        )

    asyncio.run(add())
    pricing = prices()
    pricing["prices"][1]["schedules"][0]["pricing"]["web_search_per_thousand"] = "100"
    response = usage_client.post(
        "/api/usage/rollup",
        json={**WINDOW, "session_group_limit": 1, "session_order": "cost", "pricing": pricing},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    if status == "outcome_unknown":
        assert data["session_ranking"]["effective_order"] == "tokens"
        assert data["session_ranking"]["unavailable_reason"] == "prices_incomplete"
    else:
        assert data["session_ranking"]["effective_order"] == "cost"
        assert data["session_breakdown"]["groups"][0]["session_id"] == (
            "hosted" if status == "completed" else "cost-winner"
        )


@pytest.mark.parametrize("support", [False, None, 1, "true"])
def test_cost_order_requires_positive_capability(support):
    store = InMemorySessionStore()
    # Discovery rejects malformed declarations; emulate an extension changing its
    # flag after server wiring to exercise the request guard independently.
    with TestClient(
        create_server(
            CayuApp(session_store=store, enable_logging=False), config=_LOCAL_SERVER_CONFIG
        )
    ) as client:
        store.supports_usage_cost_details = support
        response = client.post(
            "/api/usage/rollup", json={**WINDOW, "session_group_limit": 1, "session_order": "cost"}
        )
        assert response.status_code == 501


def test_empty_cost_order_and_request_validation():
    with TestClient(
        create_server(CayuApp(enable_logging=False), config=_LOCAL_SERVER_CONFIG)
    ) as client:
        response = client.post(
            "/api/usage/rollup",
            json={**WINDOW, "session_group_limit": 1, "session_order": "cost", "pricing": prices()},
        )
        assert response.status_code == 200
        assert response.json()["session_ranking"]["effective_order"] == "cost"
        assert response.json()["session_breakdown"]["groups"] == []
        for extra in (
            {"session_order": "cost"},
            {"session_order": True},
            {"session_group_limit": 101},
            {"include_cost_candidates": True},
        ):
            assert client.post("/api/usage/rollup", json={**WINDOW, **extra}).status_code == 422


def test_all_cost_sections_reuse_canonical_price_resolution(usage_client, monkeypatch):
    from cayu.budgets import pricing as pricing_module

    original = pricing_module.estimate_model_step_cost
    calls = []

    def count(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(pricing_module, "estimate_model_step_cost", count)
    response = usage_client.post(
        "/api/usage/rollup",
        json={**WINDOW, "session_group_limit": 1, "session_order": "cost", "pricing": prices()},
    )
    assert response.status_code == 200, response.text
    assert len(calls) == 2


def test_snapshot_survives_sqlite_restart_and_uses_event_date(tmp_path):
    from copy import deepcopy
    from datetime import timedelta

    path = tmp_path / "restart.sqlite"

    async def setup():
        store = SQLiteSessionStore(path)
        await seed(store)
        await store.append_event(
            "cost-winner",
            Event(
                session_id="cost-winner",
                type=EventType.MODEL_COMPLETED,
                timestamp=START + timedelta(days=1),
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "expensive",
                        "input_tokens": 10,
                        "total_tokens": 10,
                    }
                },
            ),
        )
        await store.close()

    asyncio.run(setup())
    store = SQLiteSessionStore(path)
    pricing = prices()
    schedule = pricing["prices"][1]["schedules"][0]
    schedule["effective_through"] = "2026-10-01"
    later = deepcopy(schedule)
    later.pop("effective_through")
    later["effective_from"] = "2026-10-02"
    later["pricing"]["standard"][0]["input_per_million"] = "2000"
    pricing["prices"][1]["schedules"].append(later)
    try:
        with TestClient(
            create_server(
                CayuApp(session_store=store, enable_logging=False), config=_LOCAL_SERVER_CONFIG
            )
        ) as client:
            response = client.post(
                "/api/usage/rollup",
                json={
                    **WINDOW,
                    "end_at": "2026-10-03T00:00:00Z",
                    "session_order": "cost",
                    "session_group_limit": 1,
                    "pricing": pricing,
                },
            )
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["session_breakdown"]["groups"][0]["title"] == "Title cost-winner"
            assert Decimal(
                data["session_cost_breakdown"]["groups"][0]["cost"]["currencies"][0]["total_cost"]
            ) == Decimal(".03")
            assert data["session_breakdown"]["groups"][0]["totals"]["usage"]["total_tokens"] == "20"
    finally:
        asyncio.run(store.close())


@pytest.mark.parametrize("mode", ["cost", "tokens", "rejected"])
def test_postgres_cost_candidates_match_memory(postgres_dsn, mode):
    from uuid import uuid4

    from cayu.budgets.pricing import PriceBook
    from cayu.budgets.usage_costs import build_usage_cost_report
    from cayu.sessions.queries import SessionAggregateFilter
    from cayu.sessions.usage import UsageRollupQuery
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def run():
        scope = f"cost-parity-{uuid4().hex}"
        memory = InMemorySessionStore()
        postgres = PostgresSessionStore(
            postgres_dsn, min_size=1, max_size=2, schema_mode=SchemaMode.CREATE
        )
        query = UsageRollupQuery(
            **WINDOW,
            session_group_limit=1,
            include_pricing_inputs=True,
            include_cost_candidates=mode != "tokens",
            sessions=SessionAggregateFilter(labels={"test_scope": scope}),
        )
        try:
            reports = []
            for store in (memory, postgres):
                await seed(store, scope=scope)
                if mode == "rejected":
                    await seed_rejected_pricing(store, auxiliary=True, scope=scope)
                for i in range(101):
                    await store.create(
                        RunRequest(
                            session_id=f"{scope}-pg-empty-{i}",
                            labels={"test_scope": scope},
                            agent_name="writer",
                            messages=[Message.text("user", "hello")],
                        ),
                        identity=SessionIdentity(provider_name="fake", model="cheap"),
                    )
                result = (await store.aggregate_usage(query)).validate_for_query(query)
                report = build_usage_cost_report(
                    result,
                    PriceBook.model_validate(prices()),
                    session_order="tokens" if mode == "tokens" else "cost",
                    session_limit=1,
                    group_limit=20,
                )
                reports.append(report)
            expected, actual = reports
            assert actual.result.session_breakdown == expected.result.session_breakdown
            assert actual.session_ranking == expected.session_ranking
            assert actual.model_cost_breakdown == expected.model_cost_breakdown
            assert actual.provider_cost_breakdown == expected.provider_cost_breakdown
            assert actual.session_cost_breakdown == expected.session_cost_breakdown
        finally:
            await postgres.close()

    asyncio.run(run())


def test_oversized_candidate_retry_discards_the_entire_first_snapshot():
    from cayu.budgets.aggregates import UsageRollupResultTooLarge

    class ChangingStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        calls = 0

        async def aggregate_usage(self, query):
            self.calls += 1
            result = await super().aggregate_usage(query)
            if query.include_cost_candidates:
                await self.create(
                    RunRequest(
                        session_id="arrived",
                        agent_name="new-writer",
                        title="Arrived during query",
                        messages=[Message.text("user", "hello")],
                    ),
                    identity=SessionIdentity(provider_name="fake", model="cheap"),
                )
                await self.append_event(
                    "arrived",
                    Event(
                        session_id="arrived",
                        timestamp=START,
                        type=EventType.MODEL_COMPLETED,
                        payload={
                            "usage_metrics": {
                                "provider_name": "fake",
                                "model": "cheap",
                                "input_tokens": 2000,
                                "total_tokens": 2000,
                            }
                        },
                    ),
                )
                raise UsageRollupResultTooLarge("candidate projection")
            return result

    store = ChangingStore()
    asyncio.run(seed(store))
    with TestClient(
        create_server(
            CayuApp(session_store=store, enable_logging=False), config=_LOCAL_SERVER_CONFIG
        )
    ) as client:
        response = client.post(
            "/api/usage/rollup",
            json={**WINDOW, "session_group_limit": 1, "session_order": "cost", "pricing": prices()},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert store.calls == 2
        assert data["matching_session_count"] == "3"
        assert data["totals"]["usage"]["total_tokens"] == "3010"
        assert data["session_breakdown"]["groups"][0]["title"] == "Arrived during query"
        assert data["session_ranking"]["unavailable_reason"] == "candidate_byte_limit"
        assert Decimal(data["cost"]["currencies"][0]["total_cost"]) == Decimal(".013")


def test_custom_store_cannot_return_token_limited_candidates_as_global_cost():
    class IgnoringStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def aggregate_usage(self, query):
            return await super().aggregate_usage(
                query.model_copy(update={"include_cost_candidates": False})
            )

    store = IgnoringStore()
    asyncio.run(seed(store))
    with TestClient(
        create_server(
            CayuApp(session_store=store, enable_logging=False), config=_LOCAL_SERVER_CONFIG
        )
    ) as client:
        response = client.post(
            "/api/usage/rollup",
            json={**WINDOW, "session_group_limit": 1, "session_order": "cost", "pricing": prices()},
        )
        assert response.status_code == 500
        assert "inconsistent usage projection" in response.json()["detail"]


def test_snapshot_display_metadata_obeys_redaction():
    from cayu.vaults.redaction import SecretRedactor

    store = InMemorySessionStore()
    asyncio.run(seed(store))
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        secret_redactor=SecretRedactor(["Title cost-winner", "writer"]),
    )
    with TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)) as client:
        response = client.post(
            "/api/usage/rollup",
            json={**WINDOW, "session_group_limit": 1, "session_order": "cost", "pricing": prices()},
        )
        assert response.status_code == 200, response.text
        assert "Title cost-winner" not in response.text
        assert '"writer"' not in response.text


def test_dimension_cost_models_detach_nested_financial_values(usage_client):
    from cayu.budgets.usage_costs import UsageDimensionCostBreakdown, UsageDimensionCostGroup

    response = usage_client.post("/api/usage/rollup", json={**WINDOW, "pricing": prices()})
    original = UsageDimensionCostBreakdown.model_validate_json(
        __import__("json").dumps(response.json()["model_cost_breakdown"])
    )
    group = UsageDimensionCostGroup(
        provider_name=original.groups[0].provider_name,
        model=original.groups[0].model,
        cost=original.groups[0].cost,
    )
    snapshot = UsageDimensionCostBreakdown(
        price_book_version=original.price_book_version,
        price_book_generated_at=original.price_book_generated_at,
        groups=(group,),
        accuracy=original.accuracy,
    )
    before = snapshot.model_dump(mode="json")
    group.cost.currencies[0].total_cost = Decimal("999")
    original.accuracy.reason = "mutated"
    assert snapshot.model_dump(mode="json") == before


def test_replaced_session_uses_current_incarnation_display_and_events(usage_client, usage_store):
    async def replace():
        old = await usage_store.load("cost-winner")
        await usage_store.delete_session("cost-winner")
        new = await usage_store.create(
            RunRequest(
                session_id="cost-winner",
                agent_name="replacement",
                title="Replacement session",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="cheap"),
        )
        assert old.instance_id != new.instance_id

    asyncio.run(replace())
    response = usage_client.post(
        "/api/usage/rollup",
        json={**WINDOW, "session_group_limit": 2, "session_order": "cost", "pricing": prices()},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["session_breakdown"]["groups"][0]["session_id"] == "token-winner"
    replaced = data["session_breakdown"]["groups"][1]
    assert (replaced["agent_name"], replaced["title"]) == ("replacement", "Replacement session")
    assert replaced["totals"]["usage"]["total_tokens"] == "0"
    assert Decimal(data["cost"]["currencies"][0]["total_cost"]) == Decimal(".001")


def test_expanding_redaction_does_not_break_usage_report():
    from cayu.vaults.redaction import SecretRedactor

    store = InMemorySessionStore()
    asyncio.run(
        store.create(
            RunRequest(
                session_id="safe",
                agent_name="writer",
                title="tag" * 80,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="cheap"),
        )
    )
    app = CayuApp(session_store=store, enable_logging=False, secret_redactor=SecretRedactor("tag"))
    with TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG)) as client:
        response = client.post("/api/usage/rollup", json={**WINDOW, "session_group_limit": 1})
        assert response.status_code == 200, response.text
        assert response.json()["session_breakdown"]["groups"][0]["title"] is None
        assert "tag" * 80 not in response.text


async def seed_rejected_pricing(store, *, auxiliary=False, unknown=False, scope=None):
    session_id = f"{scope}-token-winner" if scope else "token-winner"
    await store.append_event(
        session_id,
        Event(
            session_id=session_id,
            timestamp=START,
            type=EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            if auxiliary
            else EventType.MODEL_COMPLETED,
            payload={
                "usage_metrics": {
                    "provider_name": None if unknown else "fake",
                    "model": None if unknown else "cheap",
                    "input_tokens": 7,
                    "total_tokens": 7,
                    "billing_identity": "not-an-object",
                }
            },
        ),
    )


@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("group_limit", [1, 20])
def test_rejected_pricing_never_becomes_exact_dimension_zero(
    usage_client, usage_store, auxiliary, unknown, group_limit
):
    asyncio.run(seed_rejected_pricing(usage_store, auxiliary=auxiliary, unknown=unknown))
    response = usage_client.post(
        "/api/usage/rollup",
        json={
            **WINDOW,
            "pricing": prices(),
            "session_group_limit": 1,
            "session_order": "cost",
            "group_limit": group_limit,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    counter = "unpriced_auxiliary_attempts" if auxiliary else "unpriced_model_steps"
    assert body["cost"][counter] == "1"
    assert body["session_ranking"]["unavailable_reason"] == "prices_incomplete"
    session_costs = body["session_cost_breakdown"]
    session_summaries = [item["cost"] for item in session_costs["groups"]]
    if session_costs["remainder"]:
        session_summaries.append(session_costs["remainder"]["cost"])
    assert sum(int(item[counter]) for item in session_summaries) == 1
    for dimension in ("provider", "model"):
        costs = body[f"{dimension}_cost_breakdown"]
        assert costs["accuracy"]["kind"] == "truncated"
        assert "attribution" in costs["accuracy"]["reason"]
        summaries = [item["cost"] for item in costs["groups"]]
        if costs["remainder"]:
            summaries.append(costs["remainder"]["cost"])
        assert all(item["accuracy"]["kind"] == "truncated" for item in summaries)
        assert all(not item["currencies"] for item in summaries)
        assert sum(int(item["unevaluated_model_steps"]) for item in summaries) == int(
            body["totals"]["model_steps"]
        )


def test_postgres_native_session_decoder_preserves_metadata():
    """Exercise the native read entrance with the database's positional row shape."""
    from cayu.sessions.usage import UsageRollupQuery
    from cayu.storage import _postgres_aggregates as native
    from cayu.storage._session_store_sql import SessionQuerySqlPlan

    overall = {name: 0 for name in native._ROW_COLUMNS}
    overall.update(section="overall", matching_session_count=1)
    # PostgreSQL SELECT order: identifiers/state, usage counters, remainder counts,
    # then the two display columns. This is a database row, not a model/helper call.
    session_row = (
        "session",
        "empty",
        False,
        "completed",
        False,
        *([0] * 19),
        "writer",
        "Empty title",
    )

    class Cursor:
        def __init__(self, rows):
            self.rows = rows

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def execute(self, sql, params):
            assert sql and params

        async def fetchall(self):
            return self.rows

    class Connection:
        def __init__(self):
            self.rows = iter([[tuple(overall.values())], [session_row], []])

        def cursor(self):
            return Cursor(next(self.rows))

    async def run():
        query = UsageRollupQuery(**WINDOW, session_group_limit=1)
        result = await native.aggregate_session_usage(
            Connection(),
            session_plan=SessionQuerySqlPlan(
                filter_where_sql="TRUE",
                filter_params=(),
                page_where_sql="TRUE",
                page_params=(),
                order_sql="",
                pagination_sql="",
            ),
            query=query,
            as_of=START,
        )
        result.validate_for_query(query)
        assert result.session_breakdown.groups[0].agent_name == "writer"
        assert result.session_breakdown.groups[0].title == "Empty title"

    asyncio.run(run())


def test_cost_parity_cases_are_isolated_in_shared_database(monkeypatch, tmp_path):
    """Run the same parity-case setup against one persistent database without Docker."""
    from cayu.storage import postgres as postgres_module

    monkeypatch.setattr(
        postgres_module,
        "PostgresSessionStore",
        lambda *args, **kwargs: SQLiteSessionStore(tmp_path / "shared-parity.sqlite"),
    )
    for mode in ("cost", "tokens", "rejected"):
        test_postgres_cost_candidates_match_memory(None, mode)


@pytest.mark.parametrize("pricing", [False, True])
@pytest.mark.parametrize("order", ["tokens", "cost"])
@pytest.mark.parametrize("limit", [1, 10])
@pytest.mark.parametrize("sample_totals", [False, True])
def test_session_selection_preserves_sampled_accuracy(pricing, order, limit, sample_totals):
    from cayu.budgets.aggregates import (
        AggregateAccuracy,
        AggregateAccuracyKind,
        _combined_aggregate_accuracy,
    )

    source_reason = "Store samples session event counters."
    sampled = AggregateAccuracy(kind=AggregateAccuracyKind.SAMPLED, reason=source_reason)

    class SampledStore(InMemorySessionStore):
        async def aggregate_usage(self, query):
            result = await super().aggregate_usage(query)
            result.session_breakdown.accuracy = _combined_aggregate_accuracy(
                result.session_breakdown.accuracy, sampled
            )
            if sample_totals:
                result.totals_accuracy = sampled
                result.provider_breakdown.accuracy = _combined_aggregate_accuracy(
                    result.provider_breakdown.accuracy, sampled
                )
                result.model_breakdown.accuracy = _combined_aggregate_accuracy(
                    result.model_breakdown.accuracy, sampled
                )
            # The real server still validates this store-owned result against
            # the request before any reporting/selection transformation.
            return result

    store = SampledStore()
    asyncio.run(seed(store))
    with TestClient(
        create_server(
            CayuApp(session_store=store, enable_logging=False), config=_LOCAL_SERVER_CONFIG
        )
    ) as client:
        response = client.post(
            "/api/usage/rollup",
            json={
                **WINDOW,
                "session_group_limit": limit,
                "session_order": order,
                "pricing": prices() if pricing else None,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        accuracy = body["session_breakdown"]["accuracy"]
        assert accuracy["kind"] == ("truncated" if limit == 1 else "sampled")
        assert source_reason in accuracy["reason"]
        if limit == 1:
            assert "session_group_limit" in accuracy["reason"]
        if pricing:
            assert source_reason in body["session_cost_breakdown"]["accuracy"]["reason"]
        if order == "cost":
            assert body["session_ranking"]["effective_order"] == "tokens"
            expected_reason = (
                "pricing_projection_incomplete" if pricing else "pricing_not_configured"
            )
            assert body["session_ranking"]["unavailable_reason"] == expected_reason
