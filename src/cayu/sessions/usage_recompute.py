"""Compare a store's stored usage and tool totals with a recompute from its events.

Native stores serve usage rollups and tool totals from trigger-kept stored
totals. The event-level rollup they replaced stays as the reference, and this
check reads both for one window and names every difference. ``POST
/usage/rollup/recompute`` and ``cayu usage rollup --recompute`` run it.

The two reads are separate snapshots, so events written between them show up as
differences; check a window that has ended, or run it again. A difference
carries leaf values only (lists and objects by kind and length), and callers
that serve the report pass it through their redaction like any rollup value.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cayu.sessions.store import SessionStore
from cayu.sessions.usage import UsageRollupQuery

#: Differences a report lists; ``difference_count`` still counts every one.
MAX_RECOMPUTE_DIFFERENCES = 50


class UsageRecomputeDifference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Dotted path of the value that differs.")
    stored: Any = Field(description="The value from the stored totals.")
    recomputed: Any = Field(description="The value recomputed from events.")


class UsageRecomputeReport(BaseModel):
    """Whether stored usage and tool totals equal a recompute from events."""

    model_config = ConfigDict(extra="forbid")

    matches: bool
    rollup_matches: bool
    tool_usage_matches: bool
    difference_count: int
    differences: list[UsageRecomputeDifference]


def _leaf(value: Any) -> Any:
    """A value a difference may carry: scalars as they are, containers by kind only."""

    if isinstance(value, dict):
        return "<object>"
    if isinstance(value, list):
        return f"<list of {len(value)}>"
    return value


def _differences(
    stored: Any, recomputed: Any, path: str, found: list[UsageRecomputeDifference]
) -> int:
    """Count differing leaves, listing up to ``MAX_RECOMPUTE_DIFFERENCES``.

    A difference carries leaf values only, never a whole subtree: lists of
    different lengths differ in ``length`` and then element by element.
    """

    if isinstance(stored, dict) and isinstance(recomputed, dict):
        return sum(
            _differences(
                stored.get(key), recomputed.get(key), f"{path}.{key}" if path else key, found
            )
            for key in sorted(set(stored) | set(recomputed))
        )
    if isinstance(stored, list) and isinstance(recomputed, list):
        count = 0
        if len(stored) != len(recomputed):
            count += _differences(len(stored), len(recomputed), f"{path}.length", found)
        return count + sum(
            _differences(left, right, f"{path}[{index}]", found)
            for index, (left, right) in enumerate(zip(stored, recomputed, strict=False))
        )
    if stored == recomputed:
        return 0
    if len(found) < MAX_RECOMPUTE_DIFFERENCES:
        found.append(
            UsageRecomputeDifference(path=path, stored=_leaf(stored), recomputed=_leaf(recomputed))
        )
    return 1


async def recompute_usage(store: SessionStore, query: UsageRollupQuery) -> UsageRecomputeReport:
    """Read ``query`` from stored totals and from events, and compare them.

    Tool totals have no window, so the whole retained history is compared.
    """

    recomputed = await store.aggregate_usage(query, from_events=True)
    found: list[UsageRecomputeDifference] = []
    try:
        stored = (await store.aggregate_usage(query)).model_dump(mode="json", exclude={"as_of"})
    except ValueError as exc:
        # Stored totals that disagree with each other fail the result's own
        # consistency checks; that is a difference too, not a failed check.
        stored = {"error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"}
    rollup = _differences(
        stored,
        recomputed.model_dump(mode="json", exclude={"as_of"}),
        "rollup",
        found,
    )
    stored_tools, _ = await store.tool_usage_totals()
    recomputed_tools, _ = await store.tool_usage_totals(from_events=True)
    tools = _differences(
        {name: _tally(tally) for name, tally in stored_tools.items()},
        {name: _tally(tally) for name, tally in recomputed_tools.items()},
        "tool_usage",
        found,
    )
    return UsageRecomputeReport(
        matches=rollup == 0 and tools == 0,
        rollup_matches=rollup == 0,
        tool_usage_matches=tools == 0,
        difference_count=rollup + tools,
        differences=found,
    )


def _tally(tally: Any) -> dict[str, Any]:
    return {
        "calls": tally.calls,
        "failures": tally.failures,
        "approved": tally.approved,
        "rejected": tally.rejected,
        "last_used_at": None if tally.last_used_at is None else tally.last_used_at.isoformat(),
    }
