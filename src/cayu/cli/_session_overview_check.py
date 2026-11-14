"""``cayu session overview``: a session's stored summary row, checked against its events.

The summary row (``cayu_session_summaries``) holds the overview the dashboard
serves. ``--recompute`` derives the overview again from the event log with the
reference reader and compares it with the one the store serves; ``--all`` does
that for every session. The command exits 1 when any differ, so a deployment can
be verified without SQL. A session that kept changing while it was checked, as
a running one can, is listed as ``changing`` instead of compared.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from typing import Any

from cayu.sessions.store import SessionStore

_PAGE = 200


def add_session_overview_parser(commands: Any, add_target_options, add_output_options) -> None:
    parser = commands.add_parser(
        "overview",
        help="Show a session's stored summary row, or check it against the event log.",
        description=(
            "Show the summary row a local store keeps for a session: the overview the "
            "dashboard serves, whether it is current, and the stamps it was derived from. "
            "With --recompute, derive the overview again from the event log and compare; "
            "with --all, check every session. Exits 1 when a stored overview differs from "
            "its recompute."
        ),
    )
    parser.add_argument("session_id", nargs="?", help="The session to show or check.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Derive the overview again from the event log and compare it with the stored one.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_sessions",
        help="Check every session's summary row (implies --recompute).",
    )
    add_target_options(parser, cloud=False)
    add_output_options(parser)


def _row_payload(row: Any) -> dict[str, Any]:
    return {
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in asdict(row).items()
        if key != "derived"
    }


def _summaries(store: SessionStore):
    from cayu.server._session_overview import uses_browser
    from cayu.server._session_summaries import SessionSummaries

    return SessionSummaries(store, uses_browser=uses_browser)


async def run_session_overview(args: argparse.Namespace, store: SessionStore) -> dict[str, Any]:
    """The payload to render and whether every checked overview matched."""

    from cayu.server._session_summaries import check_session_summary

    if args.all_sessions:
        if args.session_id is not None:
            raise ValueError("--all checks every session; don't name one.")
        return await _check_all(store)
    if args.session_id is None:
        raise ValueError("Name a session, or pass --all.")
    rows = await store.load_session_summaries((args.session_id,))
    row = rows.get(args.session_id)
    if row is None:
        raise ValueError(f"Session not found: {args.session_id}")
    payload: dict[str, Any] = {
        "session_id": args.session_id,
        "row": _row_payload(row),
    }
    check = await check_session_summary(
        _summaries(store), args.session_id, recompute=args.recompute
    )
    payload.update(current=check.current, stored=check.stored)
    if args.recompute:
        payload.update(
            recomputed=check.recomputed, matches=check.matches, changing=not check.stable
        )
    return payload


async def _check_all(store: SessionStore) -> dict[str, Any]:
    from cayu.server._session_summaries import check_session_summary

    summaries = _summaries(store)
    checked = current = 0
    mismatched: list[str] = []
    changing: list[str] = []
    after: str | None = None
    while True:
        page = await store.page_session_summary_ids(after=after, limit=_PAGE)
        if not page:
            break
        for session_id in page:
            check = await check_session_summary(summaries, session_id)
            checked += 1
            current += int(check.current)
            if not check.stable:
                changing.append(session_id)
            elif not check.matches:
                mismatched.append(session_id)
        after = page[-1]
    return {
        "checked": checked,
        "current": current,
        "mismatched": mismatched,
        "changing": changing,
        "matches": not mismatched,
    }
