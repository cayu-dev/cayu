"""Bounded, runtime-owned classifications for unavailable model costs."""

from typing import Literal

UnpricedReason = Literal["missing_usage", "missing_pricing", "unsupported_pricing"]


def _unpriced_reason(reason: str) -> UnpricedReason:
    # Only internal, fixed reasons enter this classifier. Never project provider text.
    if reason in {
        "model.completed event has no valid normalized usage metrics",
        "model.completed event has no token usage metrics",
    }:
        return "missing_usage"
    if reason in {
        "no matching model pricing",
        "pricing schedule expired",
        "no applicable pricing schedule",
        "hosted web-search pricing is unavailable",
    }:
        return "missing_pricing"
    return "unsupported_pricing"
