"""Data moves: copy or reshape records, item by item, safely enough to run again.

Schema changes belong to migrations (`cayu storage migrate` for Cayu's tables, Alembic for an
application's). Moving data, such as from an older store into new tables or from one row
shape to another, is a data move: a paged source and one idempotent step per item.

A `DataMove` is four plain callables, each usable on its own:

- `read(cursor, limit)` returns a `SourcePage`: up to `limit` items after `cursor`, and the
  cursor of the next page (`None` after the last).
- `key(item)` names the item in the report, such as its id in the old store.
- `apply(item, dry_run=...)` moves one item and says what happened with a `StepResult`. It
  must be idempotent: run again, it finds its earlier write and answers `UNCHANGED`. When the
  target row exists and the application has changed it since, it answers `CONFLICT` and
  leaves the row alone. With `dry_run=True` it decides the same way and writes nothing.
- `verify(dry_run=...)`, optional, runs after the last page and returns the problems it
  finds, such as counts that don't match.

`run_data_move` drives them and returns a `DataMoveReport`; `cayu data-move run
MODULE:NAME` does the same from the command line against `CAYU_DATABASE_URL`, so it also
runs wherever a deploy step runs. An item whose step raises is reported as `FAILED` and the
run goes on, unless `stop_on_failure`. A run that stops early, or ends at `max_items`,
reports the cursor to resume from (`after=`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeVar

ItemT = TypeVar("ItemT")

_MAX_DETAIL = 500


class StepOutcome(StrEnum):
    """What one item's step did, or would do on a dry run."""

    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    SKIPPED = "skipped"
    CONFLICT = "conflict"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StepResult:
    """One item's outcome, with an optional short reason (shown for conflicts and skips)."""

    outcome: StepOutcome
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class SourcePage(Generic[ItemT]):
    """A page of source items and the cursor of the next page, `None` after the last."""

    items: Sequence[ItemT]
    next_cursor: str | None = None


@dataclass(frozen=True, slots=True)
class DataMove(Generic[ItemT]):
    """A named data move: a paged source, a key per item, an idempotent step, a check."""

    name: str
    read: Callable[[str | None, int], SourcePage[ItemT]]
    key: Callable[[ItemT], str]
    apply: Callable[..., StepResult]
    verify: Callable[..., Sequence[str]] | None = None


@dataclass(frozen=True, slots=True)
class ItemReport:
    """An item the report names: a conflict, a failure or a skip."""

    key: str
    outcome: StepOutcome
    detail: str | None = None


@dataclass(slots=True)
class DataMoveReport:
    """What a run did: counts per outcome, the items worth a look, and where to resume."""

    name: str
    dry_run: bool
    started_after: str | None = None
    pages: int = 0
    counts: dict[StepOutcome, int] = field(default_factory=lambda: dict.fromkeys(StepOutcome, 0))
    items: list[ItemReport] = field(default_factory=list)
    verify_problems: list[str] = field(default_factory=list)
    complete: bool = False
    resume_after: str | None = None

    @property
    def clean(self) -> bool:
        """Every page was read, and nothing conflicted, failed or failed verification."""

        return (
            self.complete
            and not self.counts[StepOutcome.CONFLICT]
            and not self.counts[StepOutcome.FAILED]
            and not self.verify_problems
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dry_run": self.dry_run,
            "started_after": self.started_after,
            "pages": self.pages,
            "counts": {outcome.value: count for outcome, count in self.counts.items()},
            "items": [
                {"key": item.key, "outcome": item.outcome.value, "detail": item.detail}
                for item in self.items
            ],
            "verify_problems": list(self.verify_problems),
            "complete": self.complete,
            "resume_after": self.resume_after,
            "clean": self.clean,
        }


def run_data_move(
    move: DataMove[ItemT],
    *,
    dry_run: bool = False,
    page_size: int = 100,
    after: str | None = None,
    max_items: int | None = None,
    stop_on_failure: bool = False,
    max_reported_items: int = 1000,
) -> DataMoveReport:
    """Run `move` page by page from `after` and report what each item's step did.

    `max_items` stops after that many items, at a page boundary, so `resume_after` is
    always a cursor `read` gave. Conflicts, failures and skips are listed in `items`, at most
    `max_reported_items` of them; the counts are always complete. `verify` runs only when
    every page was read.
    """

    if page_size < 1:
        raise ValueError("page_size must be at least 1")
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be at least 1")
    report = DataMoveReport(name=move.name, dry_run=dry_run, started_after=after)
    cursor = after
    seen = 0
    while True:
        limit = page_size if max_items is None else min(page_size, max_items - seen)
        page = move.read(cursor, limit)
        report.pages += 1
        for item in page.items:
            seen += 1
            key = move.key(item)
            try:
                result = move.apply(item, dry_run=dry_run)
            except Exception as exc:  # One item's failure is reported, not raised.
                result = StepResult(StepOutcome.FAILED, f"{type(exc).__name__}: {exc}")
            report.counts[result.outcome] += 1
            reported = (StepOutcome.CONFLICT, StepOutcome.FAILED, StepOutcome.SKIPPED)
            if result.outcome in reported and len(report.items) < max_reported_items:
                detail = None if result.detail is None else result.detail[:_MAX_DETAIL]
                report.items.append(ItemReport(key, result.outcome, detail))
            if result.outcome is StepOutcome.FAILED and stop_on_failure:
                # Resume at this page: earlier items are idempotent and answer UNCHANGED.
                report.resume_after = cursor
                return report
        cursor = page.next_cursor
        if cursor is None:
            break
        if max_items is not None and seen >= max_items:
            report.resume_after = cursor
            return report
    report.complete = True
    if move.verify is not None:
        report.verify_problems = list(move.verify(dry_run=dry_run))
    return report
