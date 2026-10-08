from __future__ import annotations

import asyncio
import json
from datetime import date
from decimal import Decimal, localcontext
from uuid import uuid4

import httpx
import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    ExecutionProfileBehaviorIdentity,
    GatewayProvider,
    Message,
    ResumeRequest,
    RunRequest,
)
from cayu.budgets import BudgetLimit, BudgetPolicy, BudgetReservation
from cayu.budgets.base import InMemoryBudgetLedger
from cayu.budgets.pricing import PriceBook, estimate_model_step_cost
from cayu.budgets.usage import UsageMetrics
from cayu.providers.gateway import HttpxGatewayTransport
from cayu.runtime.execution_profiles import ExecutionProfileMismatchError
from cayu.sessions import InMemorySessionStore
from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.storage.budget_postgres import PostgresBudgetLedger
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore


def component(dimension, nano_usd, per_units=1):
    return dict(schema_version=1, dimension=dimension, nano_usd=nano_usd, per_units=per_units)


def model(identity="example/model", *, extra=()):
    return {
        "schema_version": 1,
        "id": identity,
        "object": "model",
        "owned_by": "example",
        "cayu": {
            "schema_version": 1,
            "protocol": "chat_completions",
            "price_id": "price-v1",
            "currency": "USD",
            "components": [
                component("input_tokens", 1000),
                component("output_tokens", 2000),
                *extra,
            ],
        },
    }


def provider_for(handler):
    transport = HttpxGatewayTransport()
    transport._client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GatewayProvider(
        base_url="https://gateway.example/v1", api_key="test-key", transport=transport
    )


async def book_for(rows):
    provider = provider_for(lambda _: catalog_response(rows))
    try:
        return await provider.price_book()
    finally:
        await provider.aclose()


def catalog_response(rows):
    return httpx.Response(
        200,
        headers={"content-type": "application/json"},
        stream=httpx.ByteStream(json.dumps({"data": rows}).encode()),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("reasoning,expected", [(1000, "2"), (2000, "2"), (3000, "3")])
async def test_catalog_maps_dimensions_and_retains_exact_provenance(reasoning, expected):
    row = model(
        extra=[
            component("cached_input_tokens", 500),
            component("cache_write_tokens", 1500),
            component("reasoning_tokens", reasoning),
        ]
    )
    row["cayu"]["components"][0] = component("input_tokens", 3000, 2)
    book = await book_for([row])
    price = book.prices[0]
    assert price.provider_name == "cayu_gateway"
    assert price.match == "exact"
    tier = price.schedules[0].pricing.base()
    assert tier.input_per_million == Decimal("1.5")
    assert tier.output_per_million == Decimal(expected)
    assert tier.cache_read_input_per_million == Decimal("0.5")
    assert tier.cache_write_input_per_million == Decimal("1.5")
    provenance = price.schedules[0].provenance
    assert provenance.source == "Gateway catalog price_id=price-v1"
    assert provenance.url == "https://gateway.example/v1/models"
    assert provenance.as_of == book.generated_at
    assert PriceBook.model_validate_json(book.model_dump_json()) == book
    estimate = estimate_model_step_cost(
        metrics=UsageMetrics(
            provider_name="cayu_gateway", model="example/model-unknown", input_tokens=1
        ),
        pricing=book,
        effective_on=date.today(),
    )
    assert not estimate.priced


@pytest.mark.anyio
@pytest.mark.parametrize("dimension", ["requests", "provider_tool_calls"])
@pytest.mark.parametrize("amount", [0, 1])
async def test_unsupported_charges_are_not_free(dimension, amount):
    book = await book_for([model("other"), model(extra=[component(dimension, amount)])])
    assert ("example/model" in [p.model for p in book.prices]) is (amount == 0)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "change",
    [
        {"nano_usd": True},
        {"nano_usd": -1},
        {"nano_usd": 1.0},
        {"nano_usd": "1"},
        {"nano_usd": 1 << 53},
        {"per_units": 0},
        {"per_units": False},
        {"per_units": 1 << 53},
        {"dimension": "future_charge"},
        {"schema_version": True},
        {"unrecognized_modifier": 2},
    ],
)
async def test_malformed_component_leaves_model_unpriced(change):
    row = model()
    row["cayu"]["components"][0].update(change)
    book = await book_for([row, model("other")])
    assert [p.model for p in book.prices] == ["other"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case", ["currency", "missing-output", "duplicate-component", "no-price-id", "future-version"]
)
async def test_incomplete_or_ambiguous_prices_are_unpriced(case):
    row = model()
    metadata = row["cayu"]
    if case == "currency":
        metadata["currency"] = "EUR"
    elif case == "missing-output":
        metadata["components"].pop()
    elif case == "duplicate-component":
        metadata["components"].append(component("input_tokens", 999))
    elif case == "no-price-id":
        del metadata["price_id"]
    else:
        metadata["schema_version"] = 2
    book = await book_for([model("other"), row])
    assert [p.model for p in book.prices] == ["other"]


@pytest.mark.anyio
async def test_catalog_bounds_empty_duplicate_and_numeric_precision():
    for rows in ([], [model(), model()], [model(str(i)) for i in range(257)]):
        with pytest.raises(ValueError, match="Gateway catalog"):
            await book_for(rows)
    row = model(extra=[component("requests", 1)])
    with pytest.raises(ValueError, match="no supported"):
        await book_for([row])
    row = model()
    row["cayu"]["components"][0] = component("input_tokens", 1, 3)
    with localcontext() as context:
        context.prec = 3
        book = await book_for([row])
    rate = book.prices[0].schedules[0].pricing.base().input_per_million
    with localcontext() as context:
        context.prec = 80
        assert rate * 3000 >= 1
    assert len((await book_for([model(str(i)) for i in range(256)])).prices) == 256


@pytest.mark.anyio
async def test_refresh_is_explicit_detached_and_failure_preserves_previous_book():
    rows = [model()]
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET"
        assert request.url.path == "/v1/models"
        return catalog_response(rows)

    provider = provider_for(handle)
    try:
        original = await provider.price_book()
        app = CayuApp(
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=original,
                        reservation=BudgetReservation(max_input_tokens=1000, max_output_tokens=32),
                    ),
                )
            )
        )
        rows[0]["cayu"]["price_id"] = "price-v2"
        rows[0]["cayu"]["components"][0]["nano_usd"] = 5000
        refreshed = await provider.price_book()
        assert original.prices[0].schedules[0].pricing.base().input_per_million == 1
        assert refreshed.prices[0].schedules[0].pricing.base().input_per_million == 5
        assert refreshed.price_book_version != original.price_book_version
        assert app.budget_policy.limits[0].pricing == original
        app.budget_policy = BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=refreshed,
                    reservation=BudgetReservation(max_input_tokens=1000, max_output_tokens=32),
                ),
            )
        )
        assert app.budget_policy.limits[0].pricing == refreshed
        rows.clear()
        with pytest.raises(ValueError):
            await provider.price_book()
        assert refreshed.prices[0].schedules[0].pricing.base().input_per_million == 5
        assert app.budget_policy.limits[0].pricing == refreshed
        assert len(calls) == 3
    finally:
        await provider.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "backend", ["memory", "sqlite", pytest.param("postgres", marks=pytest.mark.postgres)]
)
async def test_unchanged_catalog_preserves_resume_identity_after_reconstruction(
    tmp_path, request, backend
):
    identity = "refresh-" + uuid4().hex
    rows = [model(), model("other")]
    posts = []

    class DeclaredGatewayProvider(GatewayProvider):
        @property
        def execution_profile_identity(self):
            # Declare the deterministic test transport from the initial run;
            # opaque provider instances intentionally cannot survive restart.
            return ExecutionProfileBehaviorIdentity(
                name="tests:catalog-refresh-gateway",
                behavior_version="1",
                implementation_version="1",
            )

    def open_provider():
        transport = HttpxGatewayTransport()
        transport._client._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        return DeclaredGatewayProvider(
            base_url="https://gateway.example/v1", api_key="test-key", transport=transport
        )

    def handle(http_request):
        if http_request.method == "GET":
            return catalog_response(rows)
        posts.append(json.loads(http_request.content))
        chunks = [
            {
                "id": "response",
                "model": "example/model",
                "choices": [{"index": 0, "delta": {"content": "OK"}, "finish_reason": None}],
            },
            {
                "id": "response",
                "model": "example/model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        ]
        body = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=httpx.ByteStream((body + "data: [DONE]\n\n").encode()),
        )

    def open_store():
        if backend == "sqlite":
            return SQLiteSessionStore(tmp_path / "refresh.db")
        if backend == "postgres":
            return PostgresSessionStore(
                request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
            )
        return InMemorySessionStore()

    def application(store, provider, book):
        app = CayuApp(
            session_store=store,
            budget_ledger=ledger,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="agent",
                        key=identity,
                        max_estimated_cost=Decimal("1"),
                        pricing=book,
                        reservation=BudgetReservation(max_input_tokens=1000, max_output_tokens=32),
                    ),
                )
            ),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(
                name=identity,
                model="example/model",
                provider_options={"cayu_gateway": {"max_completion_tokens": 32}},
            )
        )
        return app

    def open_ledger():
        if backend == "sqlite":
            return SQLiteBudgetLedger(tmp_path / "budgets.db")
        if backend == "postgres":
            return PostgresBudgetLedger(
                request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
            )
        return InMemoryBudgetLedger()

    store = open_store()
    ledger = open_ledger()
    provider = open_provider()
    try:
        original = await provider.price_book()
        app = application(store, provider, original)
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name=identity, session_id=identity, messages=[Message.text("user", "Hi")]
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        await provider.aclose()
        if backend != "memory":
            await store.close()
            await ledger.close()
            store = open_store()
            ledger = open_ledger()
        provider = open_provider()
        rows.reverse()
        refreshed = await provider.price_book()
        assert refreshed == original
        assert refreshed.generated_at == "unspecified"
        app = application(store, provider, refreshed)
        events = [
            event
            async for event in app.resume(
                ResumeRequest(session_id=identity, messages=[Message.text("user", "Again")])
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(posts) == 2
        selected = next(row for row in rows if row["id"] == "example/model")
        selected["cayu"]["price_id"] = "price-v2"
        selected["cayu"]["components"][0]["nano_usd"] = 5000
        changed = await provider.price_book()
        assert changed.price_book_version != original.price_book_version
        app = application(store, provider, changed)
        with pytest.raises(ExecutionProfileMismatchError):
            _ = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=identity, messages=[Message.text("user", "Changed prices")]
                    )
                )
            ]
        assert len(posts) == 2
    finally:
        await provider.aclose()
        if backend != "memory":
            await store.close()
            await ledger.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "backend", ["memory", "sqlite", pytest.param("postgres", marks=pytest.mark.postgres)]
)
@pytest.mark.parametrize("case", ["affordable", "at-budget", "over-budget", "unpriced"])
async def test_public_run_uses_ordinary_reservations(tmp_path, request, backend, case):
    identity = "budgeted-" + uuid4().hex
    posts = []
    selected = model(extra=[component("requests", 1)] if case == "unpriced" else [])

    def handle(request):
        if request.method == "GET":
            return catalog_response([selected, model("other")])
        posts.append(json.loads(request.content))
        chunks = [
            {
                "id": "generation",
                "model": "example/model",
                "choices": [{"index": 0, "delta": {"content": "OK"}, "finish_reason": None}],
            },
            {
                "id": "generation",
                "model": "example/model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        ]
        body = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=httpx.ByteStream(body.encode()),
        )

    provider = provider_for(handle)
    store = SQLiteSessionStore(tmp_path / "runtime.db") if backend == "sqlite" else None
    if backend == "postgres":
        store = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    try:
        book = await provider.price_book()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="agent",
                        key=identity,
                        max_estimated_cost=Decimal(
                            {
                                "affordable": "0.001065",
                                "at-budget": "0.001064",
                                "over-budget": "0.001063",
                                "unpriced": "1",
                            }[case]
                        ),
                        pricing=book,
                        reservation=BudgetReservation(max_input_tokens=1000, max_output_tokens=32),
                    ),
                )
            ),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(
                name=identity,
                model="example/model",
                provider_options={"cayu_gateway": {"max_completion_tokens": 32}},
            )
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name=identity, session_id=identity, messages=[Message.text("user", "Hi")]
                )
            )
        ]
        if case in {"affordable", "at-budget"}:
            assert len(posts) == 1
            assert "cayu" not in posts[0]
            assert any(e.type is EventType.BUDGET_RESERVED for e in events)
            assert events[-1].type is EventType.SESSION_COMPLETED
            persisted = await app.session_store.load_events(identity)
            assert any(e.type is EventType.BUDGET_RESERVED for e in persisted)
            reserved = next(e for e in persisted if e.type is EventType.BUDGET_RESERVED)
            record = await app.budget_ledger.load_reservation(reserved.payload["reservation_id"])
            assert record is not None and record.status == "reconciled"
            assert record.reserved_amount == Decimal("0.001064")
            assert record.actual_amount == Decimal("0.000014")
            if store is not None:
                await store.close()
                store = (
                    SQLiteSessionStore(tmp_path / "runtime.db")
                    if backend == "sqlite"
                    else PostgresSessionStore(request.getfixturevalue("postgres_dsn"))
                )
                reconstructed = await store.load_events(identity)
                assert any(e.type is EventType.BUDGET_RECONCILED for e in reconstructed)
                assert len(posts) == 1
        else:
            assert posts == []
            assert not any(e.type is EventType.MODEL_COMPLETED for e in events)
    finally:
        await provider.aclose()
        if store is not None:
            await store.close()


@pytest.mark.anyio
async def test_price_lookup_preserves_real_cancellation_and_does_not_retry():
    entered = asyncio.Event()
    calls = 0

    async def handle(request):
        nonlocal calls
        calls += 1
        entered.set()
        await asyncio.Event().wait()

    provider = provider_for(handle)
    handled = []

    async def lookup():
        try:
            await provider.price_book()
        except asyncio.CancelledError:
            handled.append(True)
            raise

    task = asyncio.create_task(lookup())
    try:
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 1
        assert handled == [True] and calls == 1
    finally:
        await provider.aclose()


@pytest.mark.anyio
async def test_explicit_free_prices_and_absent_cache_prices_remain_distinct():
    row = model()
    row["cayu"]["components"] = [component("input_tokens", 0), component("output_tokens", 0)]
    book = await book_for([row])
    tier = book.prices[0].schedules[0].pricing.base()
    assert tier.input_per_million == tier.output_per_million == 0
    assert tier.cache_read_input_per_million is None
    assert tier.cache_write_input_per_million is None


@pytest.mark.anyio
@pytest.mark.parametrize("status", [401, 429, 503])
async def test_catalog_failure_never_returns_free_book_or_retries(status, capsys, caplog):
    from cayu.providers.base import ModelProviderError

    calls = []
    canary = "private-catalog-response-canary"

    def handle(request):
        calls.append(request)
        return httpx.Response(status, stream=httpx.ByteStream(canary.encode()))

    provider = provider_for(handle)
    try:
        with pytest.raises(ModelProviderError) as caught:
            await provider.price_book()
        assert caught.value.status_code == status
        assert not caught.value.retryable
        assert len(calls) == 1
        captured = capsys.readouterr()
        assert (
            canary
            not in str(caught.value)
            + repr(caught.value)
            + caplog.text
            + captured.out
            + captured.err
        )
    finally:
        await provider.aclose()
