# Data moves

Migrations change schemas: `cayu storage migrate` for Cayu's tables, Alembic for an
application's own (`cayu guide applications#application-tables`). Moving data, such as
copying records from an older store into new tables or reshaping rows, is a data move:
`cayu.data_moves`.

## A move

A `DataMove` is a name and four plain callables. Each is ordinary application code you can
call and test on its own.

```python
from cayu.data_moves import DataMove, SourcePage, StepOutcome, StepResult

from database.locks import lock
from database.models import Invoice
from database.session import session


def read(cursor: str | None, limit: int) -> SourcePage[dict]:
    rows = old_store.invoices(after=cursor, limit=limit)
    return SourcePage(rows, rows[-1]["id"] if len(rows) == limit else None)


def apply(row: dict, *, dry_run: bool) -> StepResult:
    with session() as db, db.begin():
        lock(db, f"invoice:{row['id']}")
        existing = db.get(Invoice, row["id"])
        if existing is None:
            if not dry_run:
                # The app bumps `revision` on its own writes; `moved_revision` is the
                # revision this move left the row at.
                db.add(Invoice(id=row["id"], total=row["total"], revision=1, moved_revision=1))
            return StepResult(StepOutcome.CREATED)
        if existing.revision != existing.moved_revision:
            return StepResult(StepOutcome.CONFLICT, "changed in the app since it was moved")
        if existing.total == row["total"]:
            return StepResult(StepOutcome.UNCHANGED)
        if not dry_run:
            existing.total = row["total"]
            existing.revision = existing.moved_revision = existing.revision + 1
        return StepResult(StepOutcome.UPDATED)


def verify(*, dry_run: bool) -> list[str]:
    return [] if dry_run or counts_match() else ["invoice counts differ"]


INVOICES = DataMove("invoices", read, key=lambda row: row["id"], apply=apply, verify=verify)
```

- `read(cursor, limit)` returns up to `limit` items after `cursor`, and the cursor of the next
  page, or `None` after the last.
- `key(item)` names the item in the report.
- `apply(item, dry_run=...)` moves one item and returns its `StepOutcome`: `CREATED`,
  `UPDATED`, `UNCHANGED`, `SKIPPED` or `CONFLICT`. It must be idempotent, so a rerun answers
  `UNCHANGED` for what an earlier run moved. When the target row exists and the application
  has changed it since, it answers `CONFLICT` and leaves the row alone. With `dry_run=True` it
  decides the same way and writes nothing. A step that raises is reported as `FAILED`.
- `verify(dry_run=...)`, optional, runs after the last page and returns the problems it finds.

## Running it

```sh
uv run cayu data-move run moves.invoices:INVOICES --dry-run
uv run cayu data-move run moves.invoices:INVOICES --json
```

The target is a `DataMove`, or a function without arguments that returns one. The report
counts each outcome and lists the conflicts, failures and skips. It exits 0 when every page
was read with no conflict, failure or verify problem, and 3 otherwise. A failing item is
reported and the run goes on; `--stop-on-failure` stops at the first one instead.
`--max-items` stops at a page boundary. A run that stops reports `resume_after`; pass it as
`--after` to go on. Because every step is idempotent, a rerun from the start is also safe.

The move connects wherever its own code does, usually `CAYU_DATABASE_URL`. So the same
command runs on a laptop, against a scratch database in CI, or as part of the deploy step
that migrates the application's tables. Python code can call
`run_data_move(move, dry_run=..., page_size=..., after=..., max_items=...)` directly and
read the `DataMoveReport` it returns.
