"""Stored session summary rows: what ``cayu_session_summaries`` holds for one session.

The trigger-maintained columns are exact for the current session row and its
events. ``derived`` is the overview the reference reader computed for
``derived_generation``; it is current only while ``settled`` is true and every
stamp it recorded still matches. See ``cayu.storage._session_summary_schema``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cayu.budgets.usage import SessionUsageSummary

#: A session's freshness stamp: its incarnation and source generation, or
#: ``None`` when the session has no summary row.
SessionSummaryStamp = tuple[str, int] | None


@dataclass(frozen=True, slots=True)
class SessionSummaryRow:
    session_id: str
    instance_id: str
    parent_session_id: str | None
    agent_name: str
    environment_name: str | None
    provider_name: str
    model: str
    causal_budget_id: str
    status: str
    eval: bool
    #: The raw ``cayu:title`` metadata text; callers validate and redact it.
    title: str | None
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    #: Advances when one of the session's events is changed or deleted.
    usage_epoch: int
    #: Advances with every write that can change the session's overview.
    source_generation: int
    #: The generation ``derived`` was computed from, ``None`` before the first.
    derived_generation: int | None
    state: str | None
    forked: bool | None
    artifact_count: int
    derived: dict[str, Any] | None
    #: Whether every session ``derived`` read still has the stamp it had then.
    dependencies_current: bool = True

    @property
    def stamp(self) -> tuple[str, int]:
        return (self.instance_id, self.source_generation)

    @property
    def settled(self) -> bool:
        """Whether ``derived`` was computed from the session's current generation and reads."""

        return (
            self.derived is not None
            and self.derived_generation == self.source_generation
            and self.dependencies_current
        )


@dataclass(frozen=True, slots=True)
class SummarySettlement:
    """A derived overview to store if its row and dependencies are unchanged.

    ``dependencies`` maps every other session the derivation read to the stamp
    it had before that read (``None`` for a session without a row). They are
    stored with the overview, which stays current only while they hold.
    """

    session_id: str
    instance_id: str
    generation: int
    state: str
    forked: bool
    derived: dict[str, Any]
    dependencies: Mapping[str, SessionSummaryStamp] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SessionUsageDeltas:
    """Usage of each session's events after a per-session sequence.

    ``through_sequences`` gives each session's newest event the read covered;
    any later event of the session has a higher sequence, so a later read after
    it misses none. Sessions without new usage are absent from ``summaries``.
    """

    through_sequences: dict[str, int]
    summaries: dict[str, SessionUsageSummary]


__all__ = [
    "SessionSummaryRow",
    "SessionSummaryStamp",
    "SessionUsageDeltas",
    "SummarySettlement",
]
