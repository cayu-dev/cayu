"""`cayu storage migrate --progress`: one flushed line per committed revision."""

from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import sys
from datetime import UTC, datetime

import pytest

from cayu.cli import main
from cayu.storage import _postgres_schema_history as postgres_schema_history
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema

_ALL_REVISIONS = [revision.revision for revision in schema.REVISIONS]


class _FlushRecordingStdout(io.StringIO):
    """A stdout that remembers how much text had been flushed at each moment."""

    def __init__(self) -> None:
        super().__init__()
        self.flushed = ""

    def flush(self) -> None:
        super().flush()
        self.flushed = self.getvalue()

    def flushed_lines(self) -> list[str]:
        return [line for line in self.flushed.splitlines() if line]


def _json_lines(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _progress(lines: list[dict[str, object]]) -> list[dict[str, object]]:
    return [line for line in lines if line.get("event") == "revision_committed"]


def _assert_progress_lines(
    events: list[dict[str, object]], *, backend: str, revisions: list[int], staged: bool
) -> None:
    assert [event["revision"] for event in events] == revisions
    for event in events:
        rev = schema.revision(int(event["revision"]))  # type: ignore[arg-type]
        assert event == {
            "schema_version": "1",
            "event": "revision_committed",
            "backend": backend,
            "revision": rev.revision,
            "kind": str(rev.kind),
            "committed_at": event["committed_at"],
            "remaining": len(schema.pending(rev.revision)),
            "staged": staged,
        }
        committed_at = str(event["committed_at"])
        assert committed_at.endswith("Z")
        assert datetime.fromisoformat(committed_at).tzinfo is not None
    times = [datetime.fromisoformat(str(event["committed_at"])) for event in events]
    assert times == sorted(times)


def _sqlite_revision(path) -> int:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute("SELECT MAX(revision) FROM cayu_schema_migrations").fetchone()
    finally:
        connection.close()
    return int(row[0])


def test_observer_receives_each_reported_revision_and_is_scoped():
    seen: list[schema.RevisionCommitted] = []
    first, last = schema.REVISIONS[0], schema.REVISIONS[-1]

    schema.report_revision_committed(first)  # no observer: a no-op
    with schema.observe_migration_progress(seen.append):
        schema.report_revision_committed(first)
        inner: list[schema.RevisionCommitted] = []
        with schema.observe_migration_progress(inner.append):
            schema.report_revision_committed(last)
        schema.report_revision_committed(last)
    schema.report_revision_committed(last)

    assert [event.revision for event in seen] == [first.revision, last.revision]
    assert [event.revision for event in inner] == [last.revision]
    assert seen[0].remaining == len(schema.REVISIONS) - 1
    assert seen[1].remaining == 0
    assert seen[0].kind is first.kind
    assert seen[0].committed_at.tzinfo is UTC


def test_observer_follows_asyncio_run():
    seen: list[int] = []

    async def migrate() -> None:
        await asyncio.to_thread(schema.report_revision_committed, schema.REVISIONS[1])

    with schema.observe_migration_progress(lambda event: seen.append(event.revision)):
        asyncio.run(migrate())
    assert seen == [schema.REVISIONS[1].revision]


def test_sqlite_progress_lines_are_ordered_and_precede_the_unchanged_result(tmp_path, capsys):
    plain_db = tmp_path / "plain.sqlite"
    assert main(["storage", "migrate", "--sqlite", str(plain_db), "--waive-backup"]) == 0
    plain_out = capsys.readouterr().out
    plain = json.loads(plain_out)  # without --progress: exactly one JSON document
    assert "event" not in plain

    db = tmp_path / "progress.sqlite"
    assert main(["storage", "migrate", "--sqlite", str(db), "--waive-backup", "--progress"]) == 0
    lines = _json_lines(capsys.readouterr().out)
    events = _progress(lines)
    _assert_progress_lines(events, backend="sqlite", revisions=_ALL_REVISIONS, staged=True)
    assert lines[: len(events)] == events
    assert len(lines) == len(events) + 1

    result = lines[-1]
    assert "event" not in result
    assert set(result) == set(plain)
    assert set(result["migration_receipt"]) == set(plain["migration_receipt"])  # type: ignore[arg-type]
    for key in ("schema_version", "backend", "database", "application", "pending_migrations"):
        assert result[key] == plain[key]
    assert result["up_to_date"] is True
    assert _sqlite_revision(db) == schema.LATEST_REVISION


def test_sqlite_progress_line_is_flushed_before_the_next_revision_starts(tmp_path, monkeypatch):
    stdout = _FlushRecordingStdout()
    flushed_at_start: list[tuple[int, int]] = []
    apply_revision = sqlite_support._apply_revision

    def recording_apply(connection, rev):
        flushed_at_start.append((rev.revision, len(stdout.flushed_lines())))
        apply_revision(connection, rev)

    monkeypatch.setattr(sqlite_support, "_apply_revision", recording_apply)
    monkeypatch.setattr(sys, "stdout", stdout)
    db = tmp_path / "flushed.sqlite"
    assert main(["storage", "migrate", "--sqlite", str(db), "--waive-backup", "--progress"]) == 0
    monkeypatch.undo()

    # Revision N starts only after the lines for the baseline through N-1 were flushed.
    assert flushed_at_start == [
        (revision, position + 1) for position, revision in enumerate(_ALL_REVISIONS[1:])
    ]
    # Every progress line, the last revision's included, was flushed as it was written.
    flushed = _progress(_json_lines(stdout.flushed))
    assert [event["revision"] for event in flushed] == _ALL_REVISIONS


def test_sqlite_failure_mid_migration_keeps_the_lines_already_printed(
    tmp_path, monkeypatch, capsys
):
    failing = schema.REVISIONS[-3].revision
    apply_revision = sqlite_support._apply_revision

    def failing_apply(connection, rev):
        if rev.revision == failing:
            raise RuntimeError(f"injected failure at revision {failing}")
        apply_revision(connection, rev)

    monkeypatch.setattr(sqlite_support, "_apply_revision", failing_apply)
    db = tmp_path / "failing.sqlite"
    assert main(["storage", "migrate", "--sqlite", str(db), "--waive-backup", "--progress"]) == 1
    lines = _json_lines(capsys.readouterr().out)
    events = _progress(lines)
    committed = [revision for revision in _ALL_REVISIONS if revision < failing]
    _assert_progress_lines(events, backend="sqlite", revisions=committed, staged=True)
    assert len(lines) == len(events) + 1
    assert lines[-1]["error"] == {  # type: ignore[comparison-overlap]
        "code": "STORAGE_COMMAND_FAILED",
        "message": f"injected failure at revision {failing}",
    }
    # SQLite publishes only a complete staging copy, so the target was never created.
    assert not db.exists()


def test_sqlite_table_progress_prints_a_short_line_per_revision(tmp_path, capsys):
    db = tmp_path / "table.sqlite"
    args = ["storage", "migrate", "--sqlite", str(db), "--waive-backup", "--progress", "--table"]
    assert main(args) == 0
    out = capsys.readouterr().out.splitlines()
    expected = [
        f"staged revision {rev.revision} ({rev.kind}), {len(schema.pending(rev.revision))} remaining"
        for rev in schema.REVISIONS
    ]
    assert out[: len(expected)] == expected
    assert not any(line.startswith("staged revision") for line in out[len(expected) :])


def test_sqlite_progress_goes_to_stdout_when_the_result_goes_to_a_file(tmp_path, capsys):
    db = tmp_path / "output.sqlite"
    output = tmp_path / "result.json"
    args = ["storage", "migrate", "--sqlite", str(db), "--waive-backup", "--progress"]
    assert main([*args, "--output", str(output)]) == 0
    events = _json_lines(capsys.readouterr().out)
    assert [event["revision"] for event in events] == _ALL_REVISIONS
    result = json.loads(output.read_text(encoding="utf-8"))
    assert "event" not in result
    assert result["up_to_date"] is True


def test_status_does_not_accept_progress(capsys):
    with pytest.raises(SystemExit) as exited:
        main(["storage", "status", "--progress"])
    assert exited.value.code == 2
    assert "--progress" in capsys.readouterr().err


def test_up_to_date_sqlite_migration_prints_no_progress(tmp_path, capsys):
    db = tmp_path / "current.sqlite"
    assert main(["storage", "migrate", "--sqlite", str(db), "--waive-backup"]) == 0
    capsys.readouterr()
    assert main(["storage", "migrate", "--sqlite", str(db), "--waive-backup", "--progress"]) == 0
    lines = _json_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert "event" not in lines[0]


async def _reset_postgres_schema(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute("DROP SCHEMA public CASCADE")
        await conn.execute("CREATE SCHEMA public")


async def _postgres_revision(dsn: str) -> int:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        row = await (
            await conn.execute("SELECT MAX(revision) FROM cayu_schema_migrations")
        ).fetchone()
        assert row is not None
        return int(row[0])


def test_postgres_progress_lines_report_each_committed_revision(postgres_dsn, capsys):
    asyncio.run(_reset_postgres_schema(postgres_dsn))
    args = ["storage", "migrate", "--postgres", postgres_dsn, "--waive-backup", "--progress"]
    assert main(args) == 0
    lines = _json_lines(capsys.readouterr().out)
    events = _progress(lines)
    _assert_progress_lines(events, backend="postgres", revisions=_ALL_REVISIONS, staged=False)
    assert len(lines) == len(events) + 1
    result = lines[-1]
    assert "event" not in result
    assert result["up_to_date"] is True
    assert result["database"]["revision"] == schema.LATEST_REVISION  # type: ignore[index]
    assert "migration_receipt" in result


def test_postgres_failure_mid_migration_has_printed_the_committed_revisions(
    postgres_dsn, monkeypatch, capsys
):
    asyncio.run(_reset_postgres_schema(postgres_dsn))
    failing = next(
        revision.revision
        for revision in reversed(schema.REVISIONS)
        if revision.revision in postgres_schema_history._MIGRATION_STEPS
        and revision.revision not in postgres_schema_history._CONCURRENT_INDEX_MIGRATIONS
    )
    monkeypatch.setitem(postgres_schema_history._MIGRATION_STEPS, failing, ("SELECT 1 / 0",))
    args = ["storage", "migrate", "--postgres", postgres_dsn, "--waive-backup", "--progress"]
    assert main(args) == 1
    lines = _json_lines(capsys.readouterr().out)
    events = _progress(lines)
    committed = [revision for revision in _ALL_REVISIONS if revision < failing]
    _assert_progress_lines(events, backend="postgres", revisions=committed, staged=False)
    assert len(lines) == len(events) + 1
    assert lines[-1]["error"]["code"] == "STORAGE_COMMAND_FAILED"  # type: ignore[index]
    assert "division by zero" in lines[-1]["error"]["message"]  # type: ignore[index]
    # Each Postgres revision commits on its own, so the database stays where the lines say.
    assert asyncio.run(_postgres_revision(postgres_dsn)) == committed[-1]
    asyncio.run(_reset_postgres_schema(postgres_dsn))
