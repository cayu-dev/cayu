"""One growing conversation, with and without a per-turn trailing system message.

Variant ``static`` sends only a fixed leading system prompt. Variant
``late_system`` also appends a system message after the history on every call
that changes every turn and is not kept in history, the way an agent sends
transient controller or memory state. The conversation drives the provider
directly with ``ModelRequest``, as a custom agent bridge does.

Each turn records the provider-reported input, cache-read, cache-write and
uncached input tokens, and a cost estimate from ``cayu.default_price_book()``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any
from uuid import uuid4

from cayu import (
    Message,
    default_price_book,
    estimate_model_step_cost,
    normalize_usage_metrics,
)
from cayu.providers import ModelProvider, ModelRequest

SCENARIO = "late_system_message_caching"
VARIANTS = ("static", "late_system")
TURNS = 6

SYSTEM_PROMPT = (
    "You are a careful banking support assistant. Answer each question in one short "
    "sentence using only the policy notes in the conversation."
)
# About 11k tokens of stable early context.
POLICY_NOTES = "\n".join(
    f"Policy note {index}: accounts of tier {index % 7} may transfer up to "
    f"{1000 + 37 * index} units per day after identity check {index % 5}; disputes for "
    f"category {index % 11} close within {3 + index % 9} business days unless flagged by "
    f"review queue {index % 13}."
    for index in range(1, 260)
)
QUESTIONS = (
    "What is the daily transfer limit in policy note 12?",
    "How long do disputes take in policy note 40?",
    "Which identity check applies in policy note 77?",
    "What review queue flags policy note 101?",
    "What tier does policy note 150 cover?",
    "How many business days for disputes in policy note 199?",
)
# Turns 2+ of the late-system variant must read at least this share of their
# input from the provider cache in live mode.
LIVE_MIN_CACHED_SHARE = 0.5
# Releases before the cache_affinity_key field still run this scenario, so the
# same code measures the "before" state.
HAS_CACHE_AFFINITY_FIELD = "cache_affinity_key" in ModelRequest.model_fields


@dataclass
class TurnMeasurement:
    variant: str
    turn: int
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    uncached_input_tokens: int
    output_tokens: int
    cost_usd: str | None

    @property
    def cached_share(self) -> float:
        return self.cache_read_tokens / self.input_tokens if self.input_tokens else 0.0

    def as_json(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "turn": self.turn,
            "input_tokens": self.input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "output_tokens": self.output_tokens,
            "cached_share": round(self.cached_share, 4),
            "cost_usd": self.cost_usd,
        }


@dataclass
class ConversationRun:
    variant: str
    requests: list[ModelRequest] = field(default_factory=list)
    turns: list[TurnMeasurement] = field(default_factory=list)


def controller_state(turn: int, nonce: str) -> str:
    return (
        f"Controller state: turn={turn}, open_items={turn % 3}, last_tool=none, "
        f"checklist={nonce[:4]}{turn:02d}."
    )


def turn_request(
    *,
    model: str,
    history: list[Message],
    variant: str,
    turn: int,
    nonce: str,
    options: dict[str, Any],
) -> ModelRequest:
    messages = list(history)
    if variant == "late_system":
        messages.append(Message.text("system", controller_state(turn, nonce)))
    fields: dict[str, Any] = {"model": model, "messages": messages, "options": options}
    if HAS_CACHE_AFFINITY_FIELD:
        # The Cayu runtime sets one key per conversation lineage; a bridge that
        # builds its own requests sets it the same way.
        fields["cache_affinity_key"] = f"late-system-example-{nonce}-{variant}"
    return ModelRequest(**fields)


def _measure(
    *,
    provider: ModelProvider,
    provider_name: str,
    model: str,
    variant: str,
    turn: int,
    raw_usage: Any,
) -> TurnMeasurement:
    metrics = normalize_usage_metrics(
        provider_name=provider_name,
        model=model,
        requested_model=model,
        raw_usage=raw_usage,
        usage_dialect=provider.usage_dialect,
    )
    if metrics is None:
        raise RuntimeError(f"{provider_name} reported no usage for {variant} turn {turn}.")
    estimate = estimate_model_step_cost(
        metrics=metrics,
        pricing=default_price_book(),
        effective_on=date.today(),
    )
    return TurnMeasurement(
        variant=variant,
        turn=turn,
        input_tokens=metrics.input_tokens,
        cache_read_tokens=metrics.cache.read_tokens,
        cache_write_tokens=metrics.cache.write_tokens,
        uncached_input_tokens=metrics.cache.uncached_input_tokens,
        output_tokens=metrics.output_tokens,
        cost_usd=str(estimate.total_cost) if estimate.priced else None,
    )


async def run_conversation(
    provider: ModelProvider,
    *,
    provider_name: str,
    model: str,
    variant: str,
    options: dict[str, Any],
    nonce: str | None = None,
    on_request: Callable[[ModelRequest], None] | None = None,
) -> ConversationRun:
    """Run one six-turn conversation and measure every turn."""

    nonce = nonce or uuid4().hex[:12]
    # The nonce keeps each conversation's prefix unique, so turn 1 cannot read
    # a cache written by an earlier trial or by the other variant.
    history = [
        Message.text("system", f"Conversation {nonce}. {SYSTEM_PROMPT}"),
        Message.text("user", "Here are the policy notes:\n" + POLICY_NOTES),
        Message.text("assistant", "Understood. I will answer from these notes."),
    ]
    run = ConversationRun(variant=variant)
    for turn in range(1, TURNS + 1):
        history.append(Message.text("user", QUESTIONS[(turn - 1) % len(QUESTIONS)]))
        request = turn_request(
            model=model,
            history=history,
            variant=variant,
            turn=turn,
            nonce=nonce,
            options=options,
        )
        run.requests.append(request)
        if on_request is not None:
            on_request(request)
        text = ""
        usage: Any = None
        async for event in provider.stream(request):
            if event.type.value == "text_delta":
                text += event.delta
            elif event.type.value == "completed":
                usage = event.payload.get("usage")
            elif event.type.value == "error":
                raise RuntimeError(f"{provider_name} {variant} turn {turn} failed.")
        run.turns.append(
            _measure(
                provider=provider,
                provider_name=provider_name,
                model=model,
                variant=variant,
                turn=turn,
                raw_usage=usage,
            )
        )
        history.append(Message.text("assistant", text.strip() or "(no answer)"))
    return run


def _cached_share(turns: list[TurnMeasurement]) -> float:
    total = sum(turn.input_tokens for turn in turns)
    return round(sum(turn.cache_read_tokens for turn in turns) / total, 4) if total else 0


def summarize(runs: list[ConversationRun]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for run in runs:
        costs = [turn.cost_usd for turn in run.turns]
        summary[run.variant] = {
            "turns": [turn.as_json() for turn in run.turns],
            "input_tokens": sum(turn.input_tokens for turn in run.turns),
            "cache_read_tokens": sum(turn.cache_read_tokens for turn in run.turns),
            "cache_write_tokens": sum(turn.cache_write_tokens for turn in run.turns),
            "uncached_input_tokens": sum(turn.uncached_input_tokens for turn in run.turns),
            "turns_2_plus_cached_share": _cached_share([t for t in run.turns if t.turn >= 2]),
            "turns_3_plus_cached_share": _cached_share([t for t in run.turns if t.turn >= 3]),
            "cost_usd": (
                str(sum((Decimal(cost) for cost in costs if cost is not None), Decimal(0)))
                if all(cost is not None for cost in costs)
                else None
            ),
        }
    return summary
