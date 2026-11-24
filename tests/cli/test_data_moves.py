"""`cayu.data_moves.run_data_move` and `cayu data-move run`."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.data_moves import DataMove, SourcePage, StepOutcome, StepResult, run_data_move


class Store:
    """An old store of ten records and a target table, both in memory."""

    def __init__(self, count: int = 10) -> None:
        self.source = [{"id": f"r{n:02d}", "total": n} for n in range(count)]
        self.target: dict[str, dict] = {}
        self.changed_by_app: set[str] = set()
        self.broken: set[str] = set()

    def read(self, cursor: str | None, limit: int) -> SourcePage[dict]:
        start = 0 if cursor is None else int(cursor)
        items = self.source[start : start + limit]
        end = start + len(items)
        return SourcePage(items, None if end >= len(self.source) else str(end))

    def apply(self, item: dict, *, dry_run: bool) -> StepResult:
        if item["id"] in self.broken:
            raise RuntimeError("source row is unreadable")
        existing = self.target.get(item["id"])
        if existing is None:
            if not dry_run:
                self.target[item["id"]] = dict(item)
            return StepResult(StepOutcome.CREATED)
        if item["id"] in self.changed_by_app:
            return StepResult(StepOutcome.CONFLICT, "changed in the app since the last move")
        return StepResult(StepOutcome.UNCHANGED)

    def verify(self, *, dry_run: bool) -> list[str]:
        if dry_run or len(self.target) == len(self.source):
            return []
        return [f"{len(self.target)} of {len(self.source)} records moved"]

    def move(self) -> DataMove[dict]:
        return DataMove("records", self.read, lambda item: item["id"], self.apply, self.verify)


def test_a_move_runs_every_page_and_a_rerun_changes_nothing() -> None:
    store = Store()

    first = run_data_move(store.move(), page_size=3)
    again = run_data_move(store.move(), page_size=3)

    assert first.clean and first.pages == 4
    assert first.counts[StepOutcome.CREATED] == 10
    assert again.clean
    assert again.counts[StepOutcome.UNCHANGED] == 10
    assert again.counts[StepOutcome.CREATED] == 0


def test_a_dry_run_reports_the_same_outcomes_and_writes_nothing() -> None:
    store = Store()

    report = run_data_move(store.move(), dry_run=True, page_size=4)

    assert report.dry_run and report.clean
    assert report.counts[StepOutcome.CREATED] == 10
    assert store.target == {}


def test_rows_the_app_changed_are_reported_as_conflicts_and_left_alone() -> None:
    store = Store()
    run_data_move(store.move())
    store.target["r03"]["total"] = 99
    store.changed_by_app.add("r03")

    report = run_data_move(store.move())

    assert not report.clean
    assert [(item.key, item.outcome) for item in report.items] == [("r03", StepOutcome.CONFLICT)]
    assert store.target["r03"]["total"] == 99


def test_a_failing_item_is_reported_and_the_move_goes_on() -> None:
    store = Store()
    store.broken.add("r04")

    report = run_data_move(store.move(), page_size=3)

    assert report.complete and not report.clean
    assert report.counts[StepOutcome.FAILED] == 1
    assert report.items[0].key == "r04"
    assert report.items[0].detail == "RuntimeError: source row is unreadable"
    assert report.verify_problems == ["9 of 10 records moved"]


def test_stop_on_failure_resumes_at_the_failing_page() -> None:
    store = Store()
    store.broken.add("r04")

    stopped = run_data_move(store.move(), page_size=3, stop_on_failure=True)
    store.broken.clear()
    resumed = run_data_move(store.move(), page_size=3, after=stopped.resume_after)

    assert not stopped.complete and stopped.resume_after == "3"
    assert stopped.verify_problems == []  # verify waits for the last page
    assert resumed.clean
    assert resumed.counts[StepOutcome.UNCHANGED] == 1  # r03, moved before the stop
    assert len(store.target) == 10


def test_max_items_stops_at_a_page_boundary_with_a_cursor_to_go_on_from() -> None:
    store = Store()

    part = run_data_move(store.move(), page_size=4, max_items=6)
    rest = run_data_move(store.move(), page_size=4, after=part.resume_after)

    assert not part.complete and part.resume_after == "6"
    assert part.counts[StepOutcome.CREATED] == 6
    assert rest.clean and rest.counts[StepOutcome.CREATED] == 4


def test_invalid_sizes_are_refused() -> None:
    with pytest.raises(ValueError, match="page_size"):
        run_data_move(Store().move(), page_size=0)


_MODULE = """
from cayu.data_moves import DataMove, SourcePage, StepOutcome, StepResult

TARGET = {}
SOURCE = [{"id": "a"}, {"id": "b"}, {"id": "c"}]


def _read(cursor, limit):
    start = int(cursor or 0)
    end = start + limit
    return SourcePage(SOURCE[start:end], str(end) if end < len(SOURCE) else None)


def _apply(item, *, dry_run):
    if item["id"] in TARGET:
        return StepResult(StepOutcome.UNCHANGED)
    if item["id"] == "c":
        return StepResult(StepOutcome.CONFLICT, "edited")
    if not dry_run:
        TARGET[item["id"]] = item
    return StepResult(StepOutcome.CREATED)


def build():
    return DataMove("letters", _read, lambda item: item["id"], _apply)


NOT_A_MOVE = 3
"""


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "letters_move.py").write_text(_MODULE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield tmp_path
    sys.modules.pop("letters_move", None)


def test_the_command_prints_the_report_and_exits_by_whether_it_was_clean(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["data-move", "run", "letters_move:build", "--page-size", "2", "--json"])

    report = json.loads(capsys.readouterr().out)
    assert code == 3
    assert report["counts"]["created"] == 2
    assert report["items"] == [{"key": "c", "outcome": "conflict", "detail": "edited"}]
    assert report["complete"] is True and report["clean"] is False


def test_the_command_says_when_it_stopped_and_where_to_go_on(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "data-move",
            "run",
            "letters_move:build",
            "--dry-run",
            "--page-size",
            "1",
            "--max-items",
            "1",
        ]
    )

    out = capsys.readouterr().out
    assert code == 3
    assert "dry run: nothing written" in out
    assert "--after=1" in out


def test_the_command_refuses_a_target_that_is_not_a_move(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["data-move", "run", "letters_move:NOT_A_MOVE"]) == 2
    assert "is not a DataMove" in capsys.readouterr().err
