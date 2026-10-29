"""`cayu session usage` and `rollup` price model calls from a price book."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from cayu.budgets.pricing import (
    ModelPrice,
    PriceBook,
    default_price_book,
    dump_price_book,
    estimate_session_cost,
)
from cayu.cli import main
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions._call_inspection import CallPricer
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.storage import SQLiteSessionStore

_AT = datetime(2026, 10, 27, 12, tzinfo=UTC)


def _completed(provider: str, model: str, input_tokens: int, output_tokens: int) -> Event:
    return Event(
        type=EventType.MODEL_COMPLETED,
        session_id="sess_cost",
        timestamp=_AT,
        payload={
            "usage_metrics": {
                "provider_name": provider,
                "requested_model": model,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        },
    )


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "cayu.db"

    async def seed() -> None:
        store = SQLiteSessionStore(path)
        try:
            await store.create(
                RunRequest(
                    agent_name="analyst",
                    session_id="sess_cost",
                    messages=[Message.text("user", "price me")],
                ),
                identity=SessionIdentity(provider_name="anthropic", model="claude-sonnet-5"),
            )
            await store.append_event(
                "sess_cost", _completed("anthropic", "claude-sonnet-5", 1_000_000, 100_000)
            )
            await store.append_event("sess_cost", _completed("fake", "unknown-model", 10, 1))
        finally:
            await store.close()

    asyncio.run(seed())
    return path


def _custom_book(tmp_path: Path) -> Path:
    book = PriceBook(
        price_book_version="team-2026-10",
        generated_at="2026-10-28",
        prices=(
            ModelPrice.fixed(
                provider_name="anthropic",
                model="claude-sonnet-5",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("10"),
                match="exact",
            ),
        ),
    )
    path = tmp_path / "prices.json"
    path.write_text(dump_price_book(book))
    return path


def _usage(capsys, database: Path, *argv: str) -> dict:
    assert main(["session", "usage", "sess_cost", "--sqlite", str(database), "--json", *argv]) == 0
    return json.loads(capsys.readouterr().out)


def test_usage_prices_calls_from_the_bundled_price_book_by_default(database: Path, capsys) -> None:
    payload = _usage(capsys, database)

    priced, unpriced = payload["calls"]
    expected = CallPricer(default_price_book()).observe(
        _completed("anthropic", "claude-sonnet-5", 1_000_000, 100_000)
    )
    assert expected is not None
    assert expected.priced and expected.amount > 0
    assert (priced["cost_state"], priced["cost_currency"]) == ("priced", expected.currency)
    assert Decimal(priced["cost"]) == expected.amount
    assert (unpriced["cost_state"], unpriced["cost"]) == ("unpriced", None)
    assert payload["aggregate"]["estimated_cost"] == {expected.currency: priced["cost"]}
    assert payload["aggregate"]["unpriced_calls"] == 1
    book = default_price_book()
    assert payload["price_book"] == {
        "source": "bundled",
        "version": book.price_book_version,
        "generated_at": book.generated_at,
    }


def test_usage_and_rollup_take_a_custom_price_table(database: Path, capsys, tmp_path) -> None:
    prices = _custom_book(tmp_path)

    payload = _usage(capsys, database, "--price-table", str(prices))
    # 1M input tokens at 2 per million plus 100k output tokens at 10 per million.
    assert payload["calls"][0]["cost"] == "3"
    assert payload["aggregate"]["estimated_cost"] == {"USD": "3"}
    assert payload["price_book"]["source"] == "custom"
    assert payload["price_book"]["version"] == "team-2026-10"

    assert (
        main(
            ["session", "rollup", "--sqlite", str(database), "--json", "--price-book", str(prices)]
        )
        == 0
    )
    rollup = json.loads(capsys.readouterr().out)
    (group,) = rollup["groups"]
    assert (group["estimated_cost"], group["unpriced_model_calls"]) == ({"USD": "3"}, 1)
    assert rollup["price_book"]["version"] == "team-2026-10"

    assert (
        main(
            ["session", "rollup", "--sqlite", str(database), "--table", "--price-book", str(prices)]
        )
        == 0
    )
    table = capsys.readouterr().out
    assert " cost " in table.splitlines()[0] + " " and "3 USD" in table

    assert main(["session", "usage", "sess_cost", "--sqlite", str(database), "--table"]) == 0
    table = capsys.readouterr().out
    assert "unpriced" in table and "Priced with the bundled price book" in table


def test_unreadable_price_tables_are_named(database: Path, capsys, tmp_path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{}")

    assert (
        main(
            [
                "session",
                "usage",
                "sess_cost",
                "--sqlite",
                str(database),
                "--price-table",
                str(broken),
            ]
        )
        == 1
    )
    assert "is not a valid Cayu price book" in capsys.readouterr().out
    missing = tmp_path / "missing.json"
    assert (
        main(
            [
                "session",
                "usage",
                "sess_cost",
                "--sqlite",
                str(database),
                "--price-table",
                str(missing),
            ]
        )
        == 1
    )
    assert "Could not read the price book" in capsys.readouterr().out


def test_cost_matches_cost_accounting_with_hosted_web_search(tmp_path: Path, capsys) -> None:
    """Hosted tool usage on the completion and on separate events is priced as cost accounting does."""

    path = tmp_path / "hosted.db"
    book = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="openai",
                model="gpt-test",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("10"),
                web_search_per_thousand=Decimal("10"),
                match="exact",
            ),
        ),
    )
    prices = tmp_path / "prices.json"
    prices.write_text(dump_price_book(book))
    usage = {
        "provider_name": "openai",
        "requested_model": "gpt-test",
        "model": "gpt-test",
        "input_tokens": 1_000_000,
        "output_tokens": 100_000,
        "total_tokens": 1_100_000,
    }
    events = [
        *(
            Event(
                type=EventType.MODEL_HOSTED_TOOL_CALL,
                session_id="hosted",
                timestamp=_AT,
                payload={
                    "model_attempt_id": "attempt_2",
                    "tool_type": "web_search",
                    "status": "completed",
                    "provider_name": "openai",
                    "model": "gpt-test",
                },
            )
            for _ in range(2)
        ),
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="hosted",
            timestamp=_AT,
            payload={
                "model_attempt_id": "attempt_1",
                "usage_metrics": usage,
                "hosted_tool_usage": {"web_search_calls": 1000, "web_search_outcome_unknown": 0},
            },
        ),
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="hosted",
            timestamp=_AT,
            payload={"model_attempt_id": "attempt_2", "usage_metrics": usage},
        ),
    ]

    async def seed() -> None:
        store = SQLiteSessionStore(path)
        try:
            await store.create(
                RunRequest(
                    agent_name="a", session_id="hosted", messages=[Message.text("user", "x")]
                ),
                identity=SessionIdentity(provider_name="openai", model="gpt-test"),
            )
            for event in events:
                await store.append_event("hosted", event)
        finally:
            await store.close()

    asyncio.run(seed())
    expected = estimate_session_cost(session_id="hosted", events=events, pricing=book)

    assert (
        main(
            [
                "session",
                "usage",
                "hosted",
                "--sqlite",
                str(path),
                "--json",
                "--price-table",
                str(prices),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert Decimal(payload["aggregate"]["estimated_cost"]["USD"]) == expected.total_cost
    # 3 for tokens, plus 1000 searches on the first call and 2 recorded separately on the
    # second, at 10 per thousand.
    assert [call["cost"] for call in payload["calls"]] == ["13", "3.02"]
    assert expected.total_cost == Decimal("16.02")

    assert (
        main(["session", "rollup", "--sqlite", str(path), "--json", "--price-table", str(prices)])
        == 0
    )
    (group,) = json.loads(capsys.readouterr().out)["groups"]
    assert Decimal(group["estimated_cost"]["USD"]) == expected.total_cost


def test_rollup_table_never_shows_unpriced_calls_as_free() -> None:
    from cayu.cli.session import _cost_total_text

    assert _cost_total_text({}, 2) == "unknown"
    assert _cost_total_text({"USD": "3"}, 1) == "3 USD + unpriced"
    assert _cost_total_text({"USD": "3"}, 0) == "3 USD"
    assert _cost_total_text({}, 0) == "0"


def test_posted_price_books_are_bounded() -> None:
    from cayu.sessions._call_inspection import MAX_POSTED_PRICES
    from cayu.sessions._rollup import SessionRollupQuery

    huge = PriceBook(
        prices=tuple(
            ModelPrice.fixed(
                provider_name="p",
                model=f"m{index}",
                input_per_million=Decimal(1),
                output_per_million=Decimal(1),
                match="exact",
            )
            for index in range(MAX_POSTED_PRICES + 1)
        )
    )
    with pytest.raises(ValueError, match="at most"):
        SessionRollupQuery(pricing=huge)


def test_hosted_calls_the_book_cannot_price_mark_the_cost_partial() -> None:
    from cayu.sessions._call_inspection import CallPricer

    book = default_price_book()
    priced_model = next(price for price in book.prices if price.provider_name == "openai")
    usage = {
        "provider_name": "openai",
        "requested_model": priced_model.model,
        "model": priced_model.model,
        "input_tokens": 100,
        "output_tokens": 10,
        "total_tokens": 110,
    }

    def hosted(attempt: str | None) -> Event:
        payload = {
            "tool_type": "web_search",
            "status": "outcome_unknown",
            "provider_name": "openai",
        }
        if attempt is not None:
            payload["model_attempt_id"] = attempt
        return Event(
            type=EventType.MODEL_HOSTED_TOOL_CALL, session_id="s", timestamp=_AT, payload=payload
        )

    completion = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="s",
        timestamp=_AT,
        payload={"model_attempt_id": "a1", "usage_metrics": usage},
    )
    events = [hosted("a1"), completion, hosted(None)]
    expected = estimate_session_cost(session_id="s", events=events, pricing=book)

    pricer = CallPricer(book)
    costs = [cost for event in events if (cost := pricer.observe(event)) is not None]
    costs.extend(pricer.unclaimed())

    # The search's unknown outcome makes its model call unpriced, as cost accounting does,
    # and the unclaimed search is unpriced too: nothing is silently dropped.
    assert [cost.priced for cost in costs] == [item.priced for item in expected.line_items]
    assert not any(cost.priced for cost in costs)


def test_usage_counts_unpriced_auxiliary_attempts_and_skips_malformed_hosted_calls(
    tmp_path: Path, capsys
) -> None:
    path = tmp_path / "aux.db"

    async def seed() -> None:
        store = SQLiteSessionStore(path)
        try:
            await store.create(
                RunRequest(agent_name="a", session_id="aux", messages=[Message.text("user", "x")]),
                identity=SessionIdentity(provider_name="anthropic", model="claude-sonnet-5"),
            )
            for event in (
                _completed("anthropic", "claude-sonnet-5", 1_000_000, 100_000),
                Event(
                    type=EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
                    session_id="aux",
                    timestamp=_AT,
                    payload={
                        "usage_metrics": {
                            "provider_name": "fake",
                            "model": "unknown-model",
                            "input_tokens": 10,
                            "output_tokens": 1,
                            "total_tokens": 11,
                        }
                    },
                ),
                Event(
                    type=EventType.MODEL_HOSTED_TOOL_CALL,
                    session_id="aux",
                    timestamp=_AT,
                    payload={"tool_type": "web_search", "status": []},
                ),
            ):
                await store.append_event("aux", event.model_copy(update={"session_id": "aux"}))
        finally:
            await store.close()

    asyncio.run(seed())
    prices = _custom_book(tmp_path)

    assert (
        main(
            [
                "session",
                "usage",
                "aux",
                "--sqlite",
                str(path),
                "--json",
                "--price-table",
                str(prices),
            ]
        )
        == 0
    )
    aggregate = json.loads(capsys.readouterr().out)["aggregate"]
    assert (aggregate["estimated_cost"], aggregate["unpriced_calls"]) == ({"USD": "3"}, 1)

    assert (
        main(["session", "rollup", "--sqlite", str(path), "--json", "--price-table", str(prices)])
        == 0
    )
    (group,) = json.loads(capsys.readouterr().out)["groups"]
    assert (group["estimated_cost"], group["unpriced_model_calls"]) == ({"USD": "3"}, 1)
