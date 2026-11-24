"""`cayu.data_move_runs`: a record of data-move runs, and resuming from it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.data_move_runs import (
    DataMoveRun,
    FileDataMoveRunStore,
    MemoryDataMoveRunStore,
    RunOptions,
    default_run_store,
    resume_recorded,
    run_recorded,
)
from cayu.data_moves import DataMove, SourcePage, StepOutcome, StepResult


class Source:
    """Ten records; reading the page at `fail_at` raises once, as a lost process would."""

    def __init__(self, fail_at: str | None = None) -> None:
        self.records = [f"r{n}" for n in range(10)]
        self.target: set[str] = set()
        self.fail_at = fail_at

    def read(self, cursor: str | None, limit: int) -> SourcePage[str]:
        if cursor is not None and cursor == self.fail_at:
            self.fail_at = None
            raise ConnectionError("the old store went away")
        start = int(cursor or 0)
        end = start + limit
        return SourcePage(self.records[start:end], str(end) if end < len(self.records) else None)

    def apply(self, item: str, *, dry_run: bool) -> StepResult:
        if item in self.target:
            return StepResult(StepOutcome.UNCHANGED)
        if not dry_run:
            self.target.add(item)
        return StepResult(StepOutcome.CREATED)

    def move(self) -> DataMove[str]:
        return DataMove("records", self.read, str, self.apply)


def test_a_run_is_recorded_as_it_goes_and_finished() -> None:
    store = MemoryDataMoveRunStore()
    saved: list[str | None] = []
    original = store.save
    store.save = lambda run: (saved.append(run.report.resume_after), original(run))[1]

    run = run_recorded(Source().move(), store, options=RunOptions(page_size=4))

    assert run.status == "finished" and run.report.clean
    assert store.get(run.run_id) == run
    # Saved when it started, after each page with another after it, and when it finished.
    assert saved == [None, "4", "8", None]


def test_a_run_that_stopped_mid_way_resumes_from_its_record() -> None:
    source = Source(fail_at="4")
    store = MemoryDataMoveRunStore()

    with pytest.raises(ConnectionError):
        run_recorded(source.move(), store, options=RunOptions(page_size=4))
    (stopped,) = store.list_runs("records")
    resumed = resume_recorded(source.move(), store)

    assert stopped.status == "stopped" and stopped.resumable
    assert stopped.error == "ConnectionError: the old store went away"
    assert stopped.report.resume_after == "4"
    assert resumed.run_id == stopped.run_id and resumed.resumes == 1
    assert resumed.status == "finished" and resumed.report.clean
    assert resumed.report.counts[StepOutcome.CREATED] == 10
    assert source.target == set(source.records)


def test_a_run_left_running_by_a_lost_process_is_resumable() -> None:
    source = Source()
    store = MemoryDataMoveRunStore()
    part = run_recorded(source.move(), store, options=RunOptions(page_size=3, max_items=3))
    # As if the process had died after its first page.
    store.save(DataMoveRun.from_json({**part.to_json(), "status": "running"}))

    resumed = resume_recorded(source.move(), store, run_id=part.run_id)

    assert resumed.status == "stopped"  # max_items applies to each attempt
    assert resumed.report.counts[StepOutcome.CREATED] == 6


def test_a_run_that_stopped_before_its_first_page_resumes_where_it_was_asked_to_start() -> None:
    source = Source(fail_at="5")
    reads: list[str | None] = []
    read = source.read
    source.read = lambda cursor, limit: (reads.append(cursor), read(cursor, limit))[1]
    store = MemoryDataMoveRunStore()

    with pytest.raises(ConnectionError):
        run_recorded(source.move(), store, after="5", options=RunOptions(page_size=4))
    resumed = resume_recorded(source.move(), store)

    assert reads == ["5", "5", "9"]
    assert resumed.report.counts[StepOutcome.CREATED] == 5
    assert source.target == set(source.records[5:])


def test_a_check_that_failed_after_the_last_page_repeats_only_that_page() -> None:
    source = Source()
    failures = ["the database went away"]

    def verify(*, dry_run: bool) -> list[str]:
        if failures:
            raise ConnectionError(failures.pop())
        return []

    reads: list[str | None] = []
    read = source.read
    move = DataMove(
        "records",
        lambda cursor, limit: (reads.append(cursor), read(cursor, limit))[1],
        str,
        source.apply,
        verify,
    )
    store = MemoryDataMoveRunStore()

    with pytest.raises(ConnectionError):
        run_recorded(move, store, options=RunOptions(page_size=4))
    reads.clear()
    resumed = resume_recorded(move, store)

    assert reads == ["8"]
    assert resumed.status == "finished" and resumed.report.clean
    # The last checkpoint holds two pages; the repeated last page finds its rows moved.
    assert resumed.report.counts[StepOutcome.CREATED] == 8
    assert resumed.report.counts[StepOutcome.UNCHANGED] == 2
    assert source.target == set(source.records)


def test_a_run_stopped_on_a_failure_ends_clean_once_resumed() -> None:
    store = MemoryDataMoveRunStore()
    broken = {"r5"}
    source = Source()
    apply = source.apply

    def flaky(item: str, *, dry_run: bool) -> StepResult:
        if item in broken:
            raise RuntimeError("bad row")
        return apply(item, dry_run=dry_run)

    move = DataMove("records", source.read, str, flaky)
    options = RunOptions(page_size=4, stop_on_failure=True)

    stopped = run_recorded(move, store, options=options)
    broken.clear()
    resumed = resume_recorded(move, store)

    assert stopped.status == "stopped"
    assert stopped.report.stopped_by is not None
    assert stopped.report.stopped_by.detail == "RuntimeError: bad row"
    assert resumed.status == "finished" and resumed.report.clean
    assert resumed.report.stopped_by is None
    assert resumed.report.counts[StepOutcome.FAILED] == 0
    # r4 was written before the stop; moved again, it answers UNCHANGED.
    assert resumed.report.counts[StepOutcome.CREATED] == 9
    assert resumed.report.counts[StepOutcome.UNCHANGED] == 1
    assert source.target == set(source.records)
    assert DataMoveRun.from_json(stopped.to_json()) == stopped


def test_a_finished_run_is_not_resumed() -> None:
    store = MemoryDataMoveRunStore()
    run = run_recorded(Source().move(), store)

    with pytest.raises(ValueError, match="finished"):
        resume_recorded(Source().move(), store, run_id=run.run_id)
    with pytest.raises(LookupError, match="no run"):
        resume_recorded(Source().move(), store)


def test_file_records_round_trip_newest_first(tmp_path: Path) -> None:
    store = FileDataMoveRunStore(tmp_path / "runs")
    first = run_recorded(Source().move(), store, dry_run=True, run_id="first")
    second = run_recorded(Source().move(), store, run_id="second")
    (tmp_path / "runs" / "broken.json").write_text("{")

    assert [run.run_id for run in store.list_runs()] == ["second", "first"]
    assert store.get("first") == first
    assert store.get("second") == second
    assert store.get("missing") is None
    with pytest.raises(ValueError, match="run id"):
        store.get("../outside")


def test_the_default_store_is_in_the_data_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert default_run_store({"CAYU_DATA_VOLUME": "/data"}).directory == Path("/data/data-moves")
    assert default_run_store({}).directory == tmp_path / "data" / "data-moves"


_MODULE = """
from cayu.data_moves import DataMove, SourcePage, StepOutcome, StepResult

FAIL = {"at": "2"}
SOURCE = ["a", "b", "c", "d"]


def _read(cursor, limit):
    if cursor is not None and cursor == FAIL["at"]:
        FAIL["at"] = None
        raise ConnectionError("lost")
    start = int(cursor or 0)
    end = start + limit
    return SourcePage(SOURCE[start:end], str(end) if end < len(SOURCE) else None)


MOVE = DataMove("letters", _read, str, lambda item, *, dry_run: StepResult(StepOutcome.CREATED))
"""


def test_the_command_records_lists_and_resumes_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    (tmp_path / "letters_runs.py").write_text(_MODULE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CAYU_DATA_VOLUME", str(tmp_path / "volume"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        with pytest.raises(ConnectionError):
            main(["data-move", "run", "letters_runs:MOVE", "--page-size", "2"])
        capsys.readouterr()

        assert main(["data-move", "runs", "--json"]) == 0
        (listed,) = json.loads(capsys.readouterr().out)["runs"]
        assert listed["status"] == "stopped"
        assert (tmp_path / "volume" / "data-moves" / f"{listed['run_id']}.json").is_file()

        assert main(["data-move", "resume", "letters_runs:MOVE", "--json"]) == 0
        resumed = json.loads(capsys.readouterr().out)
        assert resumed["run_id"] == listed["run_id"]
        assert resumed["status"] == "finished"
        assert resumed["report"]["counts"]["created"] == 4

        assert main(["data-move", "runs"]) == 0
        assert f"{listed['run_id']}  letters  finished" in capsys.readouterr().out
    finally:
        sys.modules.pop("letters_runs", None)
