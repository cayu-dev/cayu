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

Declare the project's moves by name in `pyproject.toml`:

```toml
[tool.cayu.data_moves]
invoices = "moves.invoices:INVOICES"
```

```sh
uv run cayu data-move run invoices --dry-run
uv run cayu data-move run invoices --json
uv run cayu data-move run moves.other:OTHER   # a module:name works too
```

The target is a declared name, or the `module:name` of a `DataMove` or of a function
without arguments that returns one. Nothing is imported until a move is used. The report
counts each outcome and lists the conflicts, failures and skips. It exits 0 when every page
was read with no conflict, failure or verify problem, and 3 otherwise. A failing item is
reported and the run goes on; `--stop-on-failure` stops at the first one instead. The report
then names it as `stopped_by` and leaves its page uncounted, since going on moves that page
again.
`--max-items` stops at a page boundary. A run that stops reports `resume_after`; pass it as
`--after` to go on. Because every step is idempotent, a rerun from the start is also safe.

## Run records

Each `cayu data-move run` is recorded as it goes: once before the first page, after every page
and at the end. The record holds the run's options, its report so far and the cursor to go
on from. When the process stops mid-run (a deploy, a crash, a host that went to sleep), or
when `--max-items` or `--stop-on-failure` stopped the run, the record says where it got to:

```sh
uv run cayu data-move runs                       # newest first
uv run cayu data-move runs --run-id RUN --json   # one whole record
uv run cayu data-move resume moves.invoices:INVOICES            # the newest open run
uv run cayu data-move resume moves.invoices:INVOICES --run-id RUN
```

`resume` goes on from the record with the options the run started with, adding to the same
counts. A page that was being moved when the process stopped is moved again. Its steps are
idempotent, so they answer `UNCHANGED`, and the counts include that page twice. Run one move
at a time: the record doesn't stop two processes from running the same move at once.
`--no-record` runs without one.

Records are one JSON file per run in `CAYU_DATA_VOLUME/data-moves` when the host has a data
volume, so `cayu cloud data ls data-moves` and `cayu cloud data cat data-moves/RUN.json`
read them on a Cayu Cloud Agent. Without a data volume they go in the project's
`data/data-moves`; `--runs-dir` picks another directory. Code that keeps records elsewhere
passes its own `DataMoveRunStore` (`save`, `get` and `list_runs`) to
`cayu.data_move_runs.run_recorded` and `resume_recorded`.

## Where it runs

The move connects wherever its own code does, usually `CAYU_DATABASE_URL`. So the same
command runs on a laptop, against a scratch database in CI, or as part of the deploy step
that migrates the application's tables. Python code can call
`run_data_move(move, dry_run=..., page_size=..., after=..., max_items=...)` directly and
read the `DataMoveReport` it returns.

## On a deployed Agent

A Cayu Cloud Agent runs its declared moves itself, with its own database and secrets:

```sh
cayu cloud data-move run invoices --application acme-orders --dry-run
cayu cloud data-move runs --application acme-orders
cayu cloud data-move runs --run-id RUN --application acme-orders
cayu cloud data-move resume invoices --application acme-orders
```

`cayu serve` mounts the routes for `[tool.cayu.data_moves]` under its API path
(`/api/data-moves`, `cayu.server.data_moves`), behind the server's own authentication; a
server without authentication (`--dev`) refuses them. The CLI reaches them with the Agent's
Developer dashboard login, as `cayu cloud data` does. `run` and `resume` start the run in the
Agent's web process, then read its record every two seconds until it stops (`--no-wait`
returns at once; `--wait-seconds` bounds the wait, and the run goes on after it). The
command exits 3, as `cayu data-move run` does, when the run stopped, wasn't clean or was
still running when the wait ended, and 0 when it was clean; `--no-wait` exits 0 once the run has
started. `run` and `resume` change the Agent, so like other changing `cayu cloud` commands
they check that the Agent belongs to the organization they act in (`--organization`). One run of a move goes at a time.

The run lives in the web process. When the process stops (a deploy, a restart, an Agent that
sleeps when idle), the run's record in the Agent's data volume keeps the cursor of its last
page, and `resume` goes on from it. Such a run reads as `stopped` once no process runs it,
so the routes assume the Agent has one web process. Keep the Agent awake for a long move, or resume it after.
An Agent whose release was built with a Cayu without these routes answers `unsupported`.
A maintained public service (`service_factory`) owns its routes, so it opts in itself: pass
the declared moves to `create_agent_service`, and they are served at
`{control_plane_path}/api/data-moves` behind the service's `operator_access`:

```python
from cayu.server import create_agent_service
from cayu.server.data_moves import declared_data_move_sources

service = create_agent_service(
    app,
    ...,  # the service's other arguments
    data_moves=declared_data_move_sources(),
)
```

Without it, `cayu serve` says the declared moves aren't served and how to serve them.
`data_move_router` can also be included in any other FastAPI app; mount it under that app's
Cayu API path so `cayu cloud data-move` finds `/data-moves` there.
