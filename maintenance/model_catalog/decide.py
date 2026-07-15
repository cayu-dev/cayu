"""The agent's per-model judgment: accept the current record, or verify the official page.

This is the decision that makes it an agent rather than a cron job. It VERIFYs when the
record is stale, when seed sources disagree, when capabilities are incomplete, or when there
is no price — otherwise it ACCEPTs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from cayu import ModelInfo, ModelPrice
from maintenance.model_catalog.policy import VERIFY_MAX_AGE_DAYS


class Action(Enum):
    ACCEPT = "accept"
    VERIFY = "verify"


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str


def decide(
    model: ModelInfo,
    *,
    now: str,
    price: ModelPrice | None = None,
    sources_agree: bool = True,
    max_age_days: int = VERIFY_MAX_AGE_DAYS,
) -> Decision:
    if not sources_agree:
        return Decision(Action.VERIFY, "seed sources disagree")
    if model.context_window is None or not model.modalities_in:
        return Decision(Action.VERIFY, "incomplete capabilities")
    try:
        today = date.fromisoformat(now)
        age = (today - date.fromisoformat(model.provenance.as_of)).days
    except ValueError:
        return Decision(Action.VERIFY, "unparseable model as_of")
    if age > max_age_days:
        return Decision(Action.VERIFY, f"stale model facts ({age}d)")
    if price is None:
        return Decision(Action.VERIFY, "missing price")
    schedule = price.schedule_on(today)
    if schedule is None:
        return Decision(Action.VERIFY, "no applicable price schedule")
    try:
        price_age = (today - date.fromisoformat(schedule.provenance.as_of)).days
    except ValueError:
        return Decision(Action.VERIFY, "unparseable pricing as_of")
    if price_age > max_age_days:
        return Decision(Action.VERIFY, f"stale pricing ({price_age}d)")
    return Decision(Action.ACCEPT, "fresh, agreed, complete")
