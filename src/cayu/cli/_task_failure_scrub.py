"""`cayu task scrub-error`: re-redact a task's stored failure text, or every task's.

Against a running server (``--server-url``) or a Cayu Cloud Agent (``--application``) it
calls ``POST /api/tasks/{task_id}/scrub-error`` or pages ``POST /api/tasks/scrub-errors``,
which apply the server's registered secrets and credential shapes. Against a local store
(``--sqlite``, ``--postgres`` or the project's store) it opens the store for writing and
calls the store directly; a bare store knows no registered secrets, so only credential
shapes are masked there. Output names tasks, fields and counts, never the text.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import quote

from cayu.cli._output import print_table, terminal_safe
from cayu.cli._read_targets import ReadTarget
from cayu.tasks._failure_scrub import (
    MAX_TASK_FAILURE_SCRUB_PAGE,
    TaskFailureScrub,
    failure_text_redaction,
    scrub_result_row,
)

CLI_SCHEMA_VERSION = "1"
#: Sweep pages a single command follows before it stops and says how to continue.
MAX_SWEEP_PAGES = 1000


def add_scrub_arguments(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("task_id", nargs="?", help="The task whose failure text to scrub.")
    selection.add_argument(
        "--all",
        action="store_true",
        help="Scrub every task that keeps failure text, a page at a time.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which tasks and fields would change without writing anything.",
    )
    parser.add_argument(
        "--after",
        metavar="TASK_ID",
        help="With --all: continue a sweep after this task id (a previous run's next_after).",
    )


class _Scrubber(Protocol):
    source: dict[str, Any]

    async def scrub(self, task_id: str, *, dry_run: bool) -> dict[str, Any]: ...

    async def sweep(
        self, *, dry_run: bool, after: str | None
    ) -> tuple[list[dict[str, Any]], int, str | None]: ...

    async def close(self) -> None: ...


class _ApiScrubber:
    def __init__(self, api: Any, *, source: dict[str, Any]) -> None:
        self._api = api
        self.source = source

    async def scrub(self, task_id: str, *, dry_run: bool) -> dict[str, Any]:
        result = self._api.post(
            f"/tasks/{quote(task_id, safe='')}/scrub-error",
            {"dry_run": dry_run},
            resource="task",
        )
        return _checked_row(result, task_id)

    async def sweep(
        self, *, dry_run: bool, after: str | None
    ) -> tuple[list[dict[str, Any]], int, str | None]:
        body: dict[str, Any] = {"dry_run": dry_run, "limit": MAX_TASK_FAILURE_SCRUB_PAGE}
        if after is not None:
            body["after_task_id"] = after
        page = self._api.post("/tasks/scrub-errors", body, resource="task")
        if (
            not isinstance(page, dict)
            or not isinstance(page.get("tasks"), list)
            or type(page.get("examined")) is not int
        ):
            raise ValueError("The agent returned an unusable scrub page.")
        following = page.get("next_after_task_id")
        if following is not None and not isinstance(following, str):
            raise ValueError("The agent returned an unusable scrub page.")
        rows = [_checked_row(row, None) for row in page["tasks"]]
        return rows, page["examined"], following

    async def close(self) -> None:
        self._api.close()


class _StoreScrubber:
    def __init__(self, store: Any, *, source: dict[str, Any]) -> None:
        self._store = store
        self.source = source

    def _scrub(self, dry_run: bool) -> TaskFailureScrub:
        return TaskFailureScrub(
            redact=failure_text_redaction(),
            scrubbed_at=datetime.now(UTC),
            dry_run=dry_run,
        )

    async def scrub(self, task_id: str, *, dry_run: bool) -> dict[str, Any]:
        try:
            result = await self._store.scrub_task_failure_text(task_id, self._scrub(dry_run))
        except KeyError:
            raise ValueError(f"Task not found: {task_id}") from None
        return scrub_result_row(result, dry_run)

    async def sweep(
        self, *, dry_run: bool, after: str | None
    ) -> tuple[list[dict[str, Any]], int, str | None]:
        task_ids = await self._store.list_task_ids_with_failure_text(
            after_task_id=after, limit=MAX_TASK_FAILURE_SCRUB_PAGE
        )
        rows = []
        for task_id in task_ids:
            try:
                result = await self._store.scrub_task_failure_text(task_id, self._scrub(dry_run))
            except KeyError:
                continue
            if result.fields or result.receipts or result.skipped_reason:
                rows.append(scrub_result_row(result, dry_run))
        following = task_ids[-1] if len(task_ids) == MAX_TASK_FAILURE_SCRUB_PAGE else None
        return rows, len(task_ids), following

    async def close(self) -> None:
        close = getattr(self._store, "close", None)
        if callable(close):
            await close()


def _checked_row(value: Any, task_id: str | None) -> dict[str, Any]:
    """A server's scrub row, refused unless it has exactly the fields the route returns."""

    if (
        not isinstance(value, dict)
        or not isinstance(value.get("task_id"), str)
        or (task_id is not None and value["task_id"] != task_id)
        or type(value.get("applied")) is not bool
        or type(value.get("dry_run")) is not bool
        or not isinstance(value.get("fields"), list)
        or not all(isinstance(field, str) for field in value["fields"])
        or type(value.get("masked_values")) is not int
        or type(value.get("receipts")) is not int
        or value.get("skipped_reason") not in (None, *_SKIPS)
    ):
        raise ValueError("The agent returned an unusable scrub result.")
    return {
        key: value[key]
        for key in (
            "task_id",
            "applied",
            "dry_run",
            "fields",
            "masked_values",
            "receipts",
            "skipped_reason",
        )
    }


def open_scrubber(target: ReadTarget) -> _Scrubber:
    if target.server_url is not None:
        from cayu.cli._cloud_agent_api import CayuApi
        from cayu.cli.diagnostics import read_authorization

        api = CayuApi(
            target.server_url,
            authorization=read_authorization(target.authorization_env),
            timeout_seconds=target.timeout_seconds,
        )
        return _ApiScrubber(api, source={"kind": "server", "url": target.server_url})
    if target.store is not None:
        return _open_store_scrubber(target)
    from cayu.cli._cloud_agent_api import open_cloud_agent_api

    assert target.application is not None
    api = open_cloud_agent_api(
        target.application,
        context=target.context,
        timeout_seconds=target.timeout_seconds,
        wake_timeout_seconds=target.wake_timeout_seconds,
        offers_wake_option=True,
    )
    return _ApiScrubber(api, source={"kind": "cayu_cloud", "application": target.application})


def _open_store_scrubber(target: ReadTarget) -> _Scrubber:
    """The task store in ``target``'s database, opened for writing (never migrated)."""

    from cayu.cli.store_targets import SessionStoreBackend
    from cayu.storage import migrations as schema

    store_target = target.store
    assert store_target is not None
    if store_target.backend is SessionStoreBackend.SQLITE:
        path = store_target.sqlite_path
        assert path is not None
        if not path.is_file():
            raise ValueError(f"No SQLite database exists at {path}.")
        from cayu.storage import SQLiteTaskStore

        return _StoreScrubber(
            SQLiteTaskStore(path, schema_mode=schema.SchemaMode.VALIDATE),
            source={"kind": "sqlite", "path": str(path)},
        )
    assert store_target.postgres_dsn is not None
    from cayu.storage import PostgresTaskStore

    return _StoreScrubber(
        PostgresTaskStore(store_target.postgres_dsn, schema_mode=schema.SchemaMode.VALIDATE),
        source={"kind": "postgres"},
    )


async def run_scrub(args: argparse.Namespace, target: ReadTarget) -> int:
    if args.after is not None and not args.all:
        raise ValueError("--after applies only with --all.")
    scrubber = open_scrubber(target)
    try:
        if not args.all:
            row = await scrubber.scrub(args.task_id, dry_run=args.dry_run)
            payload: dict[str, Any] = {
                "schema_version": CLI_SCHEMA_VERSION,
                "source": scrubber.source,
                "result": row,
            }
            if args.output_format == "json":
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                _print_rows([row])
                print(_single_summary(row))
            return 0
        rows: list[dict[str, Any]] = []
        examined = 0
        after = args.after
        for _ in range(MAX_SWEEP_PAGES):
            page, count, after = await scrubber.sweep(dry_run=args.dry_run, after=after)
            rows.extend(page)
            examined += count
            if after is None:
                break
        payload = {
            "schema_version": CLI_SCHEMA_VERSION,
            "source": scrubber.source,
            "dry_run": args.dry_run,
            "examined": examined,
            "tasks": rows,
            "next_after": after,
        }
        if args.output_format == "json":
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            return 0
        if rows:
            _print_rows(rows)
        changed = sum(1 for row in rows if row["fields"] or row["receipts"])
        skipped = sum(1 for row in rows if row["skipped_reason"])
        verb = "would change" if args.dry_run else "changed"
        print(
            f"Examined {examined} tasks with stored failure text: {verb} {changed}, "
            f"skipped {skipped} a worker holds."
        )
        if after is not None:
            print(f"More tasks follow; add --after {terminal_safe(after)}")
        return 0
    finally:
        await scrubber.close()


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print_table(
        ("task_id", "fields", "masked_values", "receipts", "outcome"),
        [
            {
                "task_id": row["task_id"],
                "fields": ",".join(row["fields"]) or None,
                "masked_values": row["masked_values"],
                "receipts": row["receipts"],
                "outcome": _outcome(row),
            }
            for row in rows
        ],
    )


# What each skip means, in the table and in one task's summary.
_SKIPS = {
    "leased": (
        "skipped: a worker holds the task",
        "is held by a worker; scrub it once the worker lets it go.",
    ),
    "session_failure": (
        "skipped: bound to its session's recorded failure",
        "keeps its session's recorded runtime failure, which the session's append-only "
        "events bind exactly, so it can't be rewritten.",
    ),
    "cancellation_reconciliation": (
        "skipped: bound to its cancellation reconciliation",
        "has a cancellation reconciliation receipt that binds its error, so it can't be rewritten.",
    ),
}


def _outcome(row: dict[str, Any]) -> str:
    if row["skipped_reason"] in _SKIPS:
        return _SKIPS[row["skipped_reason"]][0]
    if row["applied"]:
        return "scrubbed"
    if row["fields"] or row["receipts"]:
        return "would scrub"
    return "nothing to mask"


def _single_summary(row: dict[str, Any]) -> str:
    task = terminal_safe(row["task_id"])
    if row["skipped_reason"] in _SKIPS:
        return f"Task {task} {_SKIPS[row['skipped_reason']][1]}"
    if row["applied"]:
        return f"Scrubbed task {task}; `cayu task show {task}` shows its error."
    if row["fields"] or row["receipts"]:
        return f"Task {task} has failure text to mask; run again without --dry-run."
    return f"Task {task} has no failure text the current redaction masks."
