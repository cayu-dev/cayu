"""The rate-limit retry time budget paces and ends overload retries."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelProviderError, ModelStreamEvent
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime.retry_policy import retry_decision
from cayu.sessions.requests import RunRequest


def _budgeted(**updates: object) -> RetryPolicy:
    fields: dict[str, object] = {
        "max_attempts": 5,
        "initial_delay_s": 0.5,
        "jitter_s": 0.0,
        "max_delay_s": 60.0,
        "max_retry_elapsed_s": 300.0,
    }
    fields.update(updates)
    return RetryPolicy.model_validate(fields)


def test_budget_and_delay_cap_are_on_by_default_and_can_be_turned_off() -> None:
    policy = RetryPolicy()

    assert policy.max_retry_elapsed_s == 300.0
    assert policy.max_delay_s == 60.0
    assert policy.model_dump(mode="json")["max_retry_elapsed_s"] == 300.0
    off = RetryPolicy(max_retry_elapsed_s=None)
    assert off.model_dump(mode="json")["max_retry_elapsed_s"] is None
    assert RetryPolicy.model_validate_json(off.model_dump_json()) == off
    budgeted = _budgeted()
    assert RetryPolicy.model_validate_json(budgeted.model_dump_json()) == budgeted


@pytest.mark.parametrize("value", [0.0, -1.0, 86400.5, True])
def test_budget_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        RetryPolicy.model_validate({"max_retry_elapsed_s": value})


@pytest.mark.parametrize(
    "fields", [{"status_code": 429}, {"status_code": 503}, {"status_code": 529}]
)
def test_overload_retries_spread_across_the_budget(fields: dict[str, int]) -> None:
    first = retry_decision(
        policy=_budgeted(), attempt=1, error="provider failed", elapsed_s=0.0, **fields
    )

    # The first retry keeps ordinary backoff, so a brief limit costs no more.
    assert first.retry is True
    assert first.delay_seconds == 0.5

    second = retry_decision(
        policy=_budgeted(), attempt=2, error="provider failed", elapsed_s=1.0, **fields
    )
    # 299 s left over three retries growing by 2x: 299 / 7 s first.
    assert second.delay_seconds == pytest.approx(299.0 / 7.0)

    later = retry_decision(
        policy=_budgeted(), attempt=3, error="provider failed", elapsed_s=200.0, **fields
    )
    # 100 s left over two retries growing by 2x.
    assert later.delay_seconds == pytest.approx(100.0 / 3.0)

    last = retry_decision(
        policy=_budgeted(), attempt=4, error="provider failed", elapsed_s=200.0, **fields
    )
    # The last retry may use all that is left, up to max_delay_s.
    assert last.delay_seconds == 60.0


def test_zero_initial_delay_is_never_paced() -> None:
    decision = retry_decision(
        policy=_budgeted(initial_delay_s=0.0),
        attempt=3,
        error="x",
        status_code=429,
        elapsed_s=10.0,
    )

    assert decision.retry is True
    assert decision.delay_seconds == 0.0


def test_rate_limit_message_is_paced_like_a_429() -> None:
    decision = retry_decision(
        policy=_budgeted(max_delay_s=1000.0), attempt=4, error="rate limit", elapsed_s=0.0
    )

    assert decision.reason is not None and decision.reason.value == "rate_limit"
    assert decision.delay_seconds == 300.0


def test_budget_never_waits_past_its_end_and_ends_retries_once_spent() -> None:
    near_end = retry_decision(
        policy=_budgeted(max_attempts=50), attempt=2, error="x", status_code=429, elapsed_s=299.0
    )
    assert near_end.retry is True
    assert near_end.delay_seconds == pytest.approx(1.0)

    spent = retry_decision(
        policy=_budgeted(max_attempts=50), attempt=2, error="x", status_code=429, elapsed_s=300.0
    )
    assert spent.retry is False
    assert spent.disposition.value == "configured_attempt_exhaustion"
    assert spent.effective_max_attempts == 2
    assert spent.delay_seconds == 0.0


def test_retry_after_longer_than_the_pace_is_still_honored() -> None:
    decision = retry_decision(
        policy=_budgeted(),
        attempt=2,
        error="x",
        status_code=429,
        retry_after_s=58.0,
        elapsed_s=250.0,
    )

    # The pace is 50 s / 7; the provider asked for 58 s but only 50 s of the
    # budget is left.
    assert decision.delay_seconds == 50.0


def test_budget_leaves_other_failures_and_unbudgeted_calls_alone() -> None:
    policy = _budgeted()

    server_error = retry_decision(
        policy=policy, attempt=1, error="x", status_code=500, elapsed_s=0.0
    )
    assert server_error.delay_seconds == 0.5
    no_elapsed = retry_decision(policy=policy, attempt=1, error="x", status_code=429)
    assert no_elapsed.delay_seconds == 0.5
    exhausted = retry_decision(policy=policy, attempt=5, error="x", status_code=429, elapsed_s=1.0)
    assert exhausted.retry is False
    assert exhausted.effective_max_attempts == 5


@pytest.mark.parametrize("elapsed", [-0.1, float("nan"), float("inf"), True])
def test_elapsed_must_be_finite_and_non_negative(elapsed: object) -> None:
    with pytest.raises(ValueError, match="elapsed_s"):
        retry_decision(
            policy=_budgeted(),
            attempt=1,
            error="x",
            status_code=429,
            elapsed_s=elapsed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(("attempt_elapsed_s", "expected_calls"), [(0.1, 3), (0.3, 1)])
def test_model_step_stops_rate_limit_retries_when_the_budget_is_spent(
    monkeypatch, attempt_elapsed_s: float, expected_calls: int
) -> None:
    from cayu.runtime import _model_step_executor

    elapsed_s = 0.0
    waits: list[float] = []

    async def advance_retry_clock(delay: float) -> None:
        nonlocal elapsed_s
        waits.append(delay)
        elapsed_s += delay
        await asyncio.sleep(0)

    # Control only the executor's retry clock and sleeps, leaving the real event
    # loop and other runtime timeouts alone. A slow first attempt may legitimately
    # spend the whole budget before any retry, regardless of machine load.
    monkeypatch.setattr(
        _model_step_executor,
        "asyncio",
        SimpleNamespace(
            **(
                vars(asyncio)
                | {
                    "get_running_loop": lambda: SimpleNamespace(time=lambda: elapsed_s),
                    "sleep": advance_retry_clock,
                }
            )
        ),
    )

    class RateLimitedProvider(ModelProvider):
        name = "fake"
        calls = 0

        async def stream(self, request):
            del request
            nonlocal elapsed_s
            self.calls += 1
            elapsed_s += attempt_elapsed_s
            raise ModelProviderError("rate limited", provider="fake", status_code=429)
            yield ModelStreamEvent.completed({})  # pragma: no cover

    provider = RateLimitedProvider()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                    retry_policy=RetryPolicy(
                        max_attempts=50,
                        initial_delay_s=0.01,
                        jitter_s=0.0,
                        max_delay_s=0.1,
                        max_retry_elapsed_s=0.3,
                    ),
                )
            )
        ]

    events = asyncio.run(run())
    errors = [event for event in events if event.type is EventType.MODEL_ERROR]
    retries = [event for event in events if event.type is EventType.MODEL_RETRY]

    # Without the elapsed-time budget the step would make all 50 attempts.
    assert provider.calls == expected_calls
    assert len(retries) == expected_calls - 1
    assert waits == [retry.payload["delay_seconds"] for retry in retries]
    assert errors[-1].payload["retry_disposition"] == "configured_attempt_exhaustion"
    assert errors[-1].payload["effective_max_attempts"] == provider.calls
    assert all(retry.payload["delay_seconds"] > 0 for retry in retries)
    assert sum(retry.payload["delay_seconds"] for retry in retries) <= 0.3 + 1e-9
