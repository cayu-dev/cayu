"""``cayu storage check-trace``: compare stored trace events with a recompute.

The session trace and event-head pages read ``cayu_trace_events``, which
triggers keep beside every event. This command reads the named sessions' trace
events both ways, from the copy and by reducing their whole events, and reports
every event where they differ, so the copy can be checked without SQL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from cayu.cli._output import add_output_options


def add_check_trace_parser(inner: Any) -> None:
    check = inner.add_parser(
        "check-trace",
        help="Compare sessions' stored trace events with a recompute from their events.",
        description=(
            "Read each session's trace events from the copy its store keeps and again by "
            "reducing its whole events, and list the event sequences where they differ. "
            "Exits 0 when every session matches and 5 when one differs. Without --sqlite or "
            "--postgres, the target resolves from CAYU_DATABASE_URL, then "
            "[tool.cayu.session_store]."
        ),
    )
    target = check.add_mutually_exclusive_group()
    target.add_argument("--sqlite", metavar="PATH", help="Path to a SQLite database file.")
    target.add_argument(
        "--postgres",
        metavar="DSN",
        help="Postgres connection string. Prefer CAYU_DATABASE_URL.",
    )
    check.add_argument("session_ids", nargs="+", metavar="SESSION_ID", help="Sessions to check.")
    add_output_options(check)


#: Exit code when a stored copy differs from its recompute.
CHECK_EXIT_MISMATCH = 5


def run_check_trace(args: argparse.Namespace, open_store: Any) -> int:
    """Check each named session; ``open_store`` opens the resolved session store."""

    from cayu.sessions.trace_events import check_trace_events

    async def check() -> list[dict[str, Any]]:
        store = open_store(args)
        try:
            ensure_schema = getattr(store, "ensure_schema", None)
            if ensure_schema is not None:
                await ensure_schema()
            results = []
            for session_id in args.session_ids:
                if await store.load_state(session_id) is None:
                    results.append({"session_id": session_id, "found": False})
                    continue
                result = await check_trace_events(store, session_id)
                results.append(
                    {
                        "session_id": session_id,
                        "found": True,
                        "checked_events": result.checked_events,
                        "matches": not result.mismatched_sequences,
                        "mismatched_sequences": list(result.mismatched_sequences),
                    }
                )
            return results
        finally:
            await store.close()

    results = asyncio.run(check())
    if args.output_format == "json":
        print(json.dumps({"sessions": results}, indent=2, sort_keys=True))
    else:
        for item in results:
            if not item["found"]:
                print(f"{item['session_id']}: not found")
            elif item["matches"]:
                print(f"{item['session_id']}: {item['checked_events']} events match")
            else:
                shown = ", ".join(str(value) for value in item["mismatched_sequences"][:20])
                print(
                    f"{item['session_id']}: {len(item['mismatched_sequences'])} of "
                    f"{item['checked_events']} events differ (sequences {shown})"
                )
    if any(not item["found"] for item in results):
        print("error: a session was not found.", file=sys.stderr)
        return 1
    return CHECK_EXIT_MISMATCH if any(not item["matches"] for item in results) else 0
