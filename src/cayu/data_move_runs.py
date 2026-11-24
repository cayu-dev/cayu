"""A record of data-move runs, kept as each page finishes, so a run can be resumed from it.

`run_recorded(move, store, ...)` runs a `cayu.data_moves.DataMove` and saves a `DataMoveRun`
before the first page and after every page: its options, its report so far and the cursor to
go on from. When the process stops mid-run (a deploy, a crash, a sleeping host), the record
says where it got to, and `resume_recorded(move, store)` goes on from there with the same
options, adding to the same counts. A page that was being moved when the process stopped is
moved again; its steps are idempotent, so they answer `UNCHANGED`, and its counts include
that page twice.

A `DataMoveRunStore` is three methods, so a record can live wherever the application keeps
things. `FileDataMoveRunStore` keeps one JSON file per run in a directory;
`default_run_store()` puts it in the host's data volume (`CAYU_DATA_VOLUME/data-moves`), which
`cayu cloud data ls|cat data-moves` reads on a Cloud Agent, else in the project's
`data/data-moves`. `MemoryDataMoveRunStore` is for tests. Run one move at a time: the record
doesn't stop two processes from running the same move at once.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, get_args
from uuid import uuid4

from cayu.data_moves import DataMove, DataMoveReport, run_data_move

ItemT = TypeVar("ItemT")

RunStatus = Literal["running", "stopped", "finished"]
#: The data volume's subdirectory `default_run_store` keeps records in.
RUNS_DIRECTORY = "data-moves"
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
_MOVE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


@dataclass(frozen=True, slots=True)
class RunOptions:
    """How a run reads its source; a resumed run keeps them."""

    page_size: int = 100
    max_items: int | None = None
    stop_on_failure: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "page_size": self.page_size,
            "max_items": self.max_items,
            "stop_on_failure": self.stop_on_failure,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> RunOptions:
        max_items = value.get("max_items")
        return cls(
            page_size=int(value.get("page_size", 100)),
            max_items=None if max_items is None else int(max_items),
            stop_on_failure=bool(value.get("stop_on_failure", False)),
        )


@dataclass(frozen=True, slots=True)
class DataMoveRun:
    """One run of a move: `running` while it goes, then `stopped` or `finished`.

    A run left `running` by a process that stopped is resumable like a `stopped` one.
    """

    run_id: str
    move: str
    status: RunStatus
    started_at: datetime
    updated_at: datetime
    options: RunOptions
    report: DataMoveReport
    resumes: int = 0
    error: str | None = None

    @property
    def resumable(self) -> bool:
        return self.status != "finished" and not self.report.complete

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "move": self.move,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "options": self.options.to_json(),
            "resumes": self.resumes,
            "error": self.error,
            "report": self.report.to_json(),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> DataMoveRun:
        status = value["status"]
        if status not in get_args(RunStatus):
            raise ValueError(f"unknown data-move run status: {status!r}")
        return cls(
            run_id=_run_id(str(value["run_id"])),
            move=str(value["move"]),
            status=status,
            started_at=datetime.fromisoformat(value["started_at"]),
            updated_at=datetime.fromisoformat(value["updated_at"]),
            options=RunOptions.from_json(value.get("options", {})),
            report=DataMoveReport.from_json(value["report"]),
            resumes=int(value.get("resumes", 0)),
            error=value.get("error"),
        )


class DataMoveRunStore(Protocol):
    """Where run records live: save replaces the record with the same `run_id`."""

    def save(self, run: DataMoveRun) -> None: ...

    def get(self, run_id: str) -> DataMoveRun | None: ...

    def list_runs(self, move: str | None = None, *, limit: int = 20) -> list[DataMoveRun]:
        """The newest runs first, of one move or of all."""
        ...


@dataclass
class MemoryDataMoveRunStore:
    """Run records in memory, for tests."""

    runs: dict[str, DataMoveRun] = field(default_factory=dict)

    def save(self, run: DataMoveRun) -> None:
        self.runs[run.run_id] = run

    def get(self, run_id: str) -> DataMoveRun | None:
        return self.runs.get(run_id)

    def list_runs(self, move: str | None = None, *, limit: int = 20) -> list[DataMoveRun]:
        runs = [run for run in self.runs.values() if move is None or run.move == move]
        return sorted(runs, key=lambda run: run.started_at, reverse=True)[:limit]


class FileDataMoveRunStore:
    """One JSON file per run, `DIRECTORY/RUN_ID.json`, each replaced whole on save."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def save(self, run: DataMoveRun) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"{_run_id(run.run_id)}.json"
        descriptor, temporary = tempfile.mkstemp(dir=self.directory, prefix=".", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(run.to_json(), handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def get(self, run_id: str) -> DataMoveRun | None:
        path = self.directory / f"{_run_id(run_id)}.json"
        try:
            return DataMoveRun.from_json(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None

    def list_runs(self, move: str | None = None, *, limit: int = 20) -> list[DataMoveRun]:
        if not self.directory.is_dir():
            return []
        runs = []
        for path in self.directory.glob("*.json"):
            try:
                run = DataMoveRun.from_json(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, KeyError, TypeError):
                continue  # A record this version can't read is left alone, not listed.
            if move is None or run.move == move:
                runs.append(run)
        return sorted(runs, key=lambda run: run.started_at, reverse=True)[:limit]


def default_run_store(environ: Mapping[str, str] | None = None) -> FileDataMoveRunStore:
    """`CAYU_DATA_VOLUME/data-moves` when the host has a data volume, else `data/data-moves`."""

    volume = (os.environ if environ is None else environ).get("CAYU_DATA_VOLUME", "").strip()
    root = Path(volume) if volume else Path.cwd() / "data"
    return FileDataMoveRunStore(root / RUNS_DIRECTORY)


def run_recorded(
    move: DataMove[ItemT],
    store: DataMoveRunStore,
    *,
    dry_run: bool = False,
    after: str | None = None,
    options: RunOptions | None = None,
    run_id: str | None = None,
) -> DataMoveRun:
    """Run `move` as `run_data_move` does, saving its record before and after every page."""

    _move_name(move.name)
    now = _now()
    run = DataMoveRun(
        run_id=_run_id(run_id or uuid4().hex[:16]),
        move=move.name,
        status="running",
        started_at=now,
        updated_at=now,
        options=RunOptions() if options is None else options,
        report=DataMoveReport(
            name=move.name, dry_run=dry_run, started_after=after, resume_after=after
        ),
    )
    store.save(run)
    return _drive(move, store, run, after=after, continuing=None)


def resume_recorded(
    move: DataMove[ItemT], store: DataMoveRunStore, *, run_id: str | None = None
) -> DataMoveRun:
    """Go on with run `run_id`, or the newest resumable run of `move`, from its record."""

    if run_id is None:
        run = next((run for run in store.list_runs(move.name, limit=1000) if run.resumable), None)
        if run is None:
            raise LookupError(f"no run of the data move {move.name!r} is left to resume")
    else:
        run = store.get(run_id)
        if run is None or run.move != move.name:
            raise LookupError(f"no run {run_id!r} of the data move {move.name!r} is recorded")
        if not run.resumable:
            raise ValueError(f"run {run_id!r} finished; start a new run instead")
    run = replace(run, status="running", resumes=run.resumes + 1, error=None, updated_at=_now())
    store.save(run)
    return _drive(move, store, run, after=None, continuing=run.report)


def _drive(
    move: DataMove[ItemT],
    store: DataMoveRunStore,
    run: DataMoveRun,
    *,
    after: str | None,
    continuing: DataMoveReport | None,
) -> DataMoveRun:
    current = run

    def checkpoint(report: DataMoveReport) -> None:
        nonlocal current
        # A copy: the run goes on changing `report`, and a store may keep what it's given.
        current = replace(current, report=_copy(report), updated_at=_now())
        store.save(current)

    try:
        report = run_data_move(
            move,
            dry_run=run.report.dry_run,
            page_size=run.options.page_size,
            after=after,
            max_items=run.options.max_items,
            stop_on_failure=run.options.stop_on_failure,
            continuing=None if continuing is None else _copy(continuing),
            on_page=checkpoint,
        )
    except BaseException as exc:
        # The source or verify step raised: the record keeps the last page's cursor.
        store.save(replace(current, status="stopped", error=_error(exc), updated_at=_now()))
        raise
    finished = replace(
        current,
        report=report,
        status="finished" if report.complete else "stopped",
        updated_at=_now(),
    )
    store.save(finished)
    return finished


def _copy(report: DataMoveReport) -> DataMoveReport:
    return DataMoveReport.from_json(report.to_json())


def _now() -> datetime:
    return datetime.now(UTC)


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]


def _run_id(value: str) -> str:
    if not _RUN_ID.fullmatch(value):
        raise ValueError("a run id is lower-case letters, digits and hyphens, at most 64")
    return value


def _move_name(value: str) -> str:
    if not _MOVE_NAME.fullmatch(value):
        raise ValueError("a data move's name is letters, digits, '_', '.' and '-'")
    return value
