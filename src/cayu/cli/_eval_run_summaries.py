"""`cayu eval runs check|rebuild-summaries`: the Evals summary tables of a local store.

The dashboard's Evals pages, and `cayu eval runs list|show`, read run rows, result
summaries, case outcomes and trial checkpoint summaries (storage revision 153).
These commands open the project's store, or one named with ``--sqlite`` or
``--postgres``, and work on those tables directly:

- `runs check RUN` shows the run's stored summary and compares it with a
  recompute from its result document. It exits 1 when they differ.
- `runs rebuild-summaries` writes the summaries of results that a binary older
  than revision 153 published, which pages otherwise derive on every read.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from cayu.cli._output import add_output_options, output_destination, print_table

COMMANDS = frozenset({"check", "rebuild-summaries"})


def _add_store_options(parser: argparse.ArgumentParser) -> None:
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--sqlite", metavar="PATH", help="Read this SQLite database.")
    target.add_argument("--postgres", metavar="DSN", help="Read this Postgres database.")
    add_output_options(parser, formats=("json", "table"), default="table")


def add_run_summary_parsers(run_actions: Any) -> None:
    check = run_actions.add_parser(
        "check",
        help="Compare a run's stored summary with a recompute from its result.",
        description=(
            "Show one run's stored summary, case outcomes and live trial outcomes from the "
            "tables the Evals pages read, and compare the summary with a recompute from the "
            "run's result document. Exits 1 when they differ or the result has none. Reads "
            "the project's store, or the one --sqlite or --postgres names."
        ),
    )
    check.add_argument("run_id", metavar="RUN_ID")
    _add_store_options(check)
    rebuild = run_actions.add_parser(
        "rebuild-summaries",
        help="Write the summaries of results that have none.",
        description=(
            "Results that a binary older than storage revision 153 published have no stored "
            "summary, so the Evals pages derive one from the whole result on every read. "
            "This writes up to --limit of them, oldest first. Without --write it lists them."
        ),
    )
    rebuild.add_argument("--write", action="store_true", help="Write the missing summaries.")
    rebuild.add_argument(
        "--limit", type=int, default=100, metavar="N", help="Results per pass (1-100)."
    )
    _add_store_options(rebuild)


def _open_store(args: argparse.Namespace, *, write: bool):
    from cayu.cli.store_targets import resolve_session_store_target
    from cayu.storage import migrations as schema
    from cayu.storage.targets import SessionStoreBackend

    target = resolve_session_store_target(sqlite=args.sqlite, postgres=args.postgres)
    if target.backend is SessionStoreBackend.SQLITE:
        from cayu.storage.evals_sqlite import SQLiteEvalStore

        assert target.sqlite_path is not None
        return SQLiteEvalStore(
            target.sqlite_path, schema_mode=schema.SchemaMode.VALIDATE, read_only=not write
        )
    from cayu.storage.evals_postgres import PostgresEvalStore

    assert target.postgres_dsn is not None
    return PostgresEvalStore(target.postgres_dsn, schema_mode=schema.SchemaMode.VALIDATE)


async def _check(store, run_id: str) -> dict[str, Any]:
    from cayu.evals.view_summaries import EvalCaseOutcomeQuery
    from cayu.evals.view_summary_checks import check_result_view_summary

    snapshot = await store.load_run_view_snapshot(run_id)
    if snapshot is None:
        raise LookupError(f"Eval run {run_id} is not stored.")
    run = snapshot.run
    revision = None if run.result is None else run.result.revision
    body: dict[str, Any] = {
        "run_id": run.id,
        "target_key": run.spec.target_key,
        "suite_id": run.spec.suite_id,
        "status": str(run.status),
        "result_revision": revision,
        "terminal_trials": [item.model_dump(mode="json") for item in snapshot.terminal_trials],
        "summary": None,
        "cases": [],
        "check": None,
    }
    if revision is None:
        return body
    summary = (await store.load_result_view_summaries((revision,))).get(revision)
    page = await store.list_case_outcomes(
        EvalCaseOutcomeQuery(revision=revision, include_evidence=False)
    )
    body["summary"] = None if summary is None else summary.model_dump(mode="json")
    body["cases"] = [
        {
            "case_id": item.case_id,
            "status": str(item.presentation.status),
            "score": item.presentation.score,
            "passed_trials": item.presentation.reliability.passed_trials,
            "total_trials": item.presentation.reliability.total_trials,
            "duration_ms": item.duration_ms,
        }
        for item in ([] if page is None else page.items)
    ]
    body["check"] = (await check_result_view_summary(store, revision)).model_dump(mode="json")
    return body


def _render_check(body: dict[str, Any]) -> None:
    print(f"Run {body['run_id']}: {body['status']}, suite {body['suite_id']}")
    summary, check = body["summary"], body["check"]
    if body["result_revision"] is None:
        print(f"{len(body['terminal_trials'])} trials finished; no result yet.")
        return
    if summary is not None:
        cases = summary["cases"]
        print(
            f"Result {body['result_revision']}: score {summary['score']}, "
            f"{cases['passed']} passed, {cases['failed']} failed, "
            f"{cases['error'] + cases['unavailable']} error, {cases['flaky']} flaky; "
            f"trials {summary['completed_trials']}/{summary['total_trials']}"
        )
    if body["cases"]:
        print_table(
            ("case_id", "status", "score", "passed_trials", "total_trials", "duration_ms"),
            body["cases"],
        )
    if check["matches"]:
        print("The stored summary matches a recompute from the result.")
    elif not check["stored"]:
        print("The result has no stored summary: run `cayu eval runs rebuild-summaries --write`.")
    else:
        print("The stored summary differs from a recompute: " + ", ".join(check["differences"]))


async def _rebuild(store, args: argparse.Namespace) -> dict[str, Any]:
    from cayu.evals.view_summary_checks import rebuild_result_view_summaries

    if args.write:
        report = await rebuild_result_view_summaries(store, limit=args.limit)
        return {"written": report.written, "remaining": report.remaining}
    missing = await store.list_results_without_view_summaries(args.limit)
    return {"missing": list(missing), "has_more": len(missing) == args.limit}


def _render_rebuild(body: dict[str, Any]) -> None:
    if "written" in body:
        more = " More remain; run it again." if body["remaining"] else ""
        print(f"Wrote {body['written']} result summaries.{more}")
        return
    more = " (more remain)" if body["has_more"] else ""
    print(f"{len(body['missing'])} results have no stored summary{more}.")
    for revision in body["missing"]:
        print(f"  {revision}")


def run_run_summary_command(args: argparse.Namespace) -> int:
    rebuild = args.eval_runs_command == "rebuild-summaries"
    if rebuild and not 1 <= args.limit <= 100:
        raise SystemExit("--limit must be between 1 and 100.")

    async def run() -> dict[str, Any]:
        store = _open_store(args, write=rebuild and args.write)
        try:
            return await (_rebuild(store, args) if rebuild else _check(store, args.run_id))
        finally:
            await store.close()

    try:
        body = asyncio.run(run())
    except LookupError as exc:
        raise SystemExit(str(exc).strip("'\"")) from exc
    with output_destination(args.output):
        if args.output_format == "json":
            print(json.dumps(body, indent=2, sort_keys=True))
        elif rebuild:
            _render_rebuild(body)
        else:
            _render_check(body)
    check = body.get("check")
    return 1 if check is not None and not check["matches"] else 0


__all__ = ["COMMANDS", "add_run_summary_parsers", "run_run_summary_command"]
