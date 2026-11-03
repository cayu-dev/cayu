"""`cayu cloud compass findings list|dismiss`: narrow, count and dismiss findings in bulk.

`list` narrows the findings by rule, dimension, score and tool, keeps the strongest
`--limit`, and `--summary` counts them by rule and tool. Cloud filters them when it can
(it then answers `total`); on a Cloud from before that, the CLI filters the full listing
itself, so the answer is the same either way. Narrowed only by `--status` and `--kind`,
`--limit` and `--cursor` read Cloud's paged listing instead, in the same order, and the
answer's `next_cursor` goes on after the last finding listed.

`dismiss --rule` dismisses a rule's proposed findings in one go. It lists them first,
shows what it will dismiss and asks before doing it (`--yes` skips the question), then
sends them to Cloud's bulk route in batches under one idempotency key, which Cloud turns
into one key per finding. A Cloud without that route gets them one at a time. Findings
already reviewed are left as they are, so repeating the command dismisses nothing more.

These commands print a table by default; `--json` prints the JSON envelope the other
`cayu cloud` commands print.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, cast
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_auth import CloudAuthError
from cayu.cli._cloud_diagnostics import safe_text
from cayu.cli._cloud_settings import _cell, _table

DIMENSIONS = ("correctness", "cost", "time", "attention")
KINDS = ("agent_change", "autonomy_grant", "platform_issue")
# Compass ranks findings by dimension in this order, then by score, highest first.
_DIMENSION_ORDER = {dimension: index for index, dimension in enumerate(DIMENSIONS)}
# The ledger steps a finding's tool is read from: a tool call, or the approval gate on one.
_TOOL_STEPS = frozenset({"tool", "approval_gate"})
# Filters `list` and `dismiss --rule` share; Cloud reads each from the query parameter of the
# same name.
_FILTERS = ("agent", "rule", "dimension", "min_score", "tool")
# Cloud's paged listing of the Organization's findings, narrowed to one Agent by `agent`. It
# reads `status` and `kind` but none of `_FILTERS`, and pages in Compass's rank order.
_PAGED_FINDINGS = "/v1/compass/findings"
# Findings one request of the paged listing reads, as many as Cloud answers in one.
_FINDINGS_PAGE = 200
# Findings per bulk-dismiss request, as many as Cloud takes in one.
DISMISS_BATCH = 100
# Findings the confirmation lists one by one before it only counts the rest.
_PREVIEW_ROWS = 10
_MESSAGE_LIMIT = 500
# A cursor is printed whole: Cloud takes one of up to 1,024 characters.
_CURSOR_LIMIT = 1024


def _positive(value: str) -> int:
    number = _integer(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _non_negative(value: str) -> int:
    number = _integer(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return number


def _integer(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None


def add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    """The filters `findings list` and `findings dismiss --rule` share."""

    parser.add_argument("--agent", help="Only this watched agent's findings.")
    parser.add_argument("--rule", help="Only findings of this Compass rule, such as failing_tool.")
    parser.add_argument("--dimension", choices=DIMENSIONS)
    parser.add_argument(
        "--min-score", type=_non_negative, metavar="N", help="Only findings scoring N or more."
    )
    parser.add_argument(
        "--tool", help="Only findings about this tool (its calls, or the approval gate on it)."
    )


def add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the JSON envelope the other `cayu cloud` commands print.",
    )
    parser.set_defaults(_cloud_render=render_findings, _cloud_render_failure=render_failure)


def add_list_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--kind", choices=KINDS, help="Only findings of this kind (not with the filters above)."
    )
    parser.add_argument(
        "--limit",
        type=_positive,
        metavar="N",
        help=(
            "Only the N strongest matching findings. Narrowed by no more than --status and "
            "--kind, the answer's next_cursor goes on after them."
        ),
    )
    parser.add_argument(
        "--cursor",
        help=(
            "Go on after an earlier listing, from its next_cursor; give the same --status and "
            "--kind. Not with --agent, --rule, --dimension, --min-score or --tool."
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Count the matching findings by rule and tool instead of listing them.",
    )


def finding_tool(finding: Mapping[str, Any]) -> str | None:
    """The tool a finding is about: the step its ledger rows name, when that step is a tool
    call or the approval gate on one."""

    rows = finding.get("ledger_rows")
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("kind") in _TOOL_STEPS:
            name = row.get("name")
            if isinstance(name, str) and name:
                return name
    return None


def list_matching(
    client: CloudApiClient,
    application: str,
    path: str,
    arguments: argparse.Namespace,
    *,
    limit: int | None,
) -> dict[str, Any]:
    """`findings list`: from Cloud's paged listing when it can read what was asked, else from
    the Agent's listing (``list_findings``).

    Both rank the findings the same way, so ``limit`` keeps the strongest either way. The
    paged one also answers ``next_cursor``, but filters only by status and kind; and its
    findings carry no ``score``, so with neither ``limit`` nor anything only it reads, the
    Agent's listing answers, with their scores, as before.
    """

    narrowed = [
        f"--{option.replace('_', '-')}"
        for option in _FILTERS
        if getattr(arguments, option, None) is not None
    ]
    paging = [
        f"--{option}"
        for option in ("cursor", "kind")
        if getattr(arguments, option, None) is not None
    ]
    if narrowed and paging:
        raise ValueError(
            f"{' and '.join(paging)} can't be combined with {', '.join(narrowed)}: Cloud pages "
            "findings by --status and --kind only. Leave those out, or use --limit for the "
            "strongest that match."
        )
    if not narrowed and (limit is not None or paging):
        paged = page_findings(client, application, arguments, limit=limit)
        if paged is not None:
            return paged
    return list_findings(client, path, arguments, limit=limit)


def page_findings(
    client: CloudApiClient, application: str, arguments: argparse.Namespace, *, limit: int | None
) -> dict[str, Any] | None:
    """The Agent's findings of ``--status`` and ``--kind`` after ``--cursor``, strongest
    first, at most ``limit`` of them (every one without it), read a page at a time.

    Answers ``total`` matching and the ``next_cursor`` after the last one listed (None when
    none follow). None from a Cloud that doesn't page findings, when nothing asked needs it.
    """

    base = {"agent": application}
    for option in ("status", "kind"):
        value = getattr(arguments, option, None)
        if value is not None:
            base[option] = str(value)
    items: list[dict[str, Any]] = []
    cursor: str | None = arguments.cursor
    # Every cursor a page started from: a page that answers one of them again would loop.
    seen: set[str] = set() if cursor is None else {cursor}
    while True:
        size = _FINDINGS_PAGE if limit is None else min(_FINDINGS_PAGE, limit - len(items))
        query = {**base, "limit": str(size), **({} if cursor is None else {"cursor": cursor})}
        answer = client.request("GET", _PAGED_FINDINGS, query=query)
        total = answer.get("total")
        if "total" not in answer and not items:
            # A Cloud from before paging ignores `limit` and `cursor` and answers no total.
            if arguments.cursor is not None or arguments.kind is not None:
                raise CloudApiError(
                    "compass_paging_unavailable",
                    "This Cayu Cloud can't page Compass findings or list them by kind yet; "
                    "leave out --cursor and --kind.",
                )
            return None
        records = answer.get("findings")
        following = answer.get("next_cursor")
        if (
            type(total) is not int
            or not isinstance(records, list)
            or not (following is None or isinstance(following, str))
        ):
            raise CloudApiError("api_response_invalid", "Compass's page of findings is invalid.")
        items.extend(item for item in records if isinstance(item, dict))
        cursor = following or None
        if cursor is not None and cursor in seen:
            raise CloudApiError(
                "api_response_invalid", "Compass's pages of findings repeat a cursor."
            )
        if cursor is None or (limit is not None and len(items) >= limit):
            break
        seen.add(cursor)
    return {"items": items, "total": total, "next_cursor": cursor}


def list_findings(
    client: CloudApiClient, path: str, arguments: argparse.Namespace, *, limit: int | None
) -> dict[str, Any]:
    """The matching findings, strongest first under ``limit``, with ``total`` matching."""

    query = {
        option: str(value)
        for option in ("status", *_FILTERS)
        if (value := getattr(arguments, option, None)) is not None
    }
    if limit is not None:
        query["limit"] = str(limit)
    answer = client.request("GET", f"{path}/findings", query=query or None)
    listed = answer.get("items")
    if not isinstance(listed, list):
        raise CloudApiError("api_response_invalid", "Compass findings are missing their items.")
    items = [item for item in listed if isinstance(item, dict)]
    total = answer.get("total")
    if type(total) is int:
        # Cloud read the filters and the limit.
        return {**answer, "items": items}
    # A Cloud from before filtering answers every finding of the status and agent asked for.
    matched = [item for item in items if _matches(item, arguments)]
    total = len(matched)
    if limit is not None:
        matched = sorted(matched, key=_rank)[:limit]
    return {**answer, "items": matched, "total": total}


def _matches(finding: Mapping[str, Any], arguments: argparse.Namespace) -> bool:
    if arguments.rule is not None and finding.get("rule") != arguments.rule:
        return False
    if arguments.dimension is not None and finding.get("dimension") != arguments.dimension:
        return False
    if arguments.min_score is not None:
        score = finding.get("score")
        if type(score) is not int or score < arguments.min_score:
            return False
    return arguments.tool is None or finding_tool(finding) == arguments.tool


def _rank(finding: Mapping[str, Any]) -> tuple[int, int, str]:
    score = finding.get("score")
    return (
        _DIMENSION_ORDER.get(str(finding.get("dimension")), len(DIMENSIONS)),
        -score if type(score) is int else 0,
        str(finding.get("finding_id")),
    )


#: Sessions one request reads when listing a finding's sessions.
_SESSION_PAGE = 500


def with_sample(finding: Mapping[str, Any]) -> dict[str, Any]:
    """The finding, saying whether its cited ``sessions`` are all of them or a sample."""

    count = finding.get("session_count")
    if type(count) is not int:
        return dict(finding)
    cited = finding.get("sessions")
    shown = len(cited) if isinstance(cited, list) else 0
    return {
        **finding,
        "sessions_cited": f"{shown} of {count}"
        + ("" if shown >= count else "; `findings sessions` lists every one"),
    }


def finding_sessions(
    client: CloudApiClient, finding_path: str, *, limit: int | None
) -> dict[str, Any]:
    """Every session behind a finding (or the latest ``limit``), following Cloud's pages."""

    sessions: list[dict[str, Any]] = []
    cursor: str | None = None
    answer: dict[str, Any] = {}
    try:
        while limit is None or len(sessions) < limit:
            size = _SESSION_PAGE if limit is None else min(_SESSION_PAGE, limit - len(sessions))
            query = {"limit": str(size), **({} if cursor is None else {"cursor": cursor})}
            answer = client.request("GET", f"{finding_path}/sessions", query=query)
            sessions.extend(_records(answer.get("sessions"), key="session_id"))
            cursor = (
                answer.get("next_cursor") if isinstance(answer.get("next_cursor"), str) else None
            )
            if cursor is None:
                break
    except CloudApiError as exc:
        # Only a bare 404 (an older Cloud's missing route): Cloud's own codes say more.
        if exc.status_code == 404 and exc.code is None and not sessions:
            raise CloudApiError(
                "finding_sessions_unavailable",
                "No such finding, or this Cayu Cloud can't list a finding's sessions yet.",
                status_code=404,
            ) from None
        raise
    return {
        "finding_id": answer.get("finding_id"),
        "sessions": sessions,
        "total": answer.get("total"),
        "first_seen_at": answer.get("first_seen_at"),
        "last_seen_at": answer.get("last_seen_at"),
        "complete": cursor is None,
    }


def summarize(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The findings counted by rule and tool, largest group first."""

    groups: dict[tuple[str, str | None], dict[str, Any]] = {}
    for item in items:
        rule = str(item.get("rule") or "-")
        tool = finding_tool(item)
        group = groups.setdefault(
            (rule, tool),
            {
                "rule": rule,
                "tool": tool,
                "dimension": item.get("dimension"),
                "findings": 0,
                "proposed": 0,
                "accepted": 0,
                "dismissed": 0,
                "top_score": None,
            },
        )
        group["findings"] += 1
        status = item.get("status")
        if status in {"proposed", "accepted", "dismissed"}:
            group[status] += 1
        score = item.get("score")
        if type(score) is int and (group["top_score"] is None or score > group["top_score"]):
            group["top_score"] = score
    return sorted(
        groups.values(), key=lambda group: (-group["findings"], group["rule"], group["tool"] or "")
    )


def dismiss_by_rule(
    client: CloudApiClient,
    path: str,
    arguments: argparse.Namespace,
    *,
    key: str,
) -> dict[str, Any]:
    """Dismiss the proposed findings that match, after asking (``confirm_on_terminal``) unless
    ``--yes``."""

    matched = list_findings(client, path, arguments, limit=None)["items"]
    proposed = [item for item in matched if item.get("status") == "proposed"]
    skipped = [
        {"finding_id": item.get("finding_id"), "status": item.get("status")}
        for item in matched
        if item.get("status") != "proposed"
    ]
    result: dict[str, Any] = {
        "rule": arguments.rule,
        "matched": len(matched),
        "dismissed": [],
        "skipped": skipped,
        "refused": [],
        "missing": [],
    }
    if not proposed:
        return result
    if not arguments.yes and not confirm_on_terminal(proposed, arguments):
        raise CloudApiError("not_confirmed", "Nothing was dismissed.")
    finding_ids = [str(item["finding_id"]) for item in proposed]
    try:
        _send_bulk(client, path, arguments.note, finding_ids, key=key, result=result)
    except _BulkUnsupported:
        print(
            "cayu cloud: This Cayu Cloud has no bulk dismiss yet; dismissing the "
            f"{len(finding_ids)} findings one at a time.",
            file=sys.stderr,
        )
        _send_each(client, path, arguments.note, finding_ids, key=key, result=result)
    return result


class _BulkUnsupported(Exception):
    """Cloud has no bulk-dismiss route."""


class BulkDismissError(CloudApiError):
    """A bulk dismiss that failed, with how many findings it had dismissed by then."""

    def __init__(self, error: CloudApiError, *, dismissed: int) -> None:
        message = str(error)
        if dismissed:
            message += (
                f" Dismissed {dismissed} findings before that; repeating the command with the "
                "same --idempotency-key dismisses the rest."
            )
        super().__init__(
            error.category,
            message,
            status_code=error.status_code,
            detail=error.detail,
            code=error.code,
            rejected_fields=error.rejected_fields,
            transport_error=error.transport_error,
            request=error.request,
        )
        self.dismissed = dismissed


def _send_bulk(
    client: CloudApiClient,
    path: str,
    note: str,
    finding_ids: Sequence[str],
    *,
    key: str,
    result: dict[str, Any],
) -> None:
    """Every batch under the same key: Cloud derives one key per finding from it."""

    for start in range(0, len(finding_ids), DISMISS_BATCH):
        batch = list(finding_ids[start : start + DISMISS_BATCH])
        try:
            answer = client.request(
                "POST",
                f"{path}/findings/dismiss",
                payload={"finding_ids": batch, "note": note},
                idempotency_key=key,
            )
        except CloudApiError as exc:
            if exc.status_code == 405 and exc.code is None and start == 0:
                raise _BulkUnsupported from None
            raise BulkDismissError(exc, dismissed=len(result["dismissed"])) from None
        except CloudAuthError as exc:
            raise BulkDismissError(_auth_failure(exc), dismissed=len(result["dismissed"])) from None
        result["dismissed"].extend(_ids(answer.get("dismissed")))
        result["missing"].extend(_ids(answer.get("missing")))
        result["skipped"].extend(_outcomes(answer.get("skipped"), ("status",)))
        result["refused"].extend(_outcomes(answer.get("refused"), ("code", "message")))


def _send_each(
    client: CloudApiClient,
    path: str,
    note: str,
    finding_ids: Sequence[str],
    *,
    key: str,
    result: dict[str, Any],
) -> None:
    """One finding at a time, each under a key derived from ``key``. Each is read again just
    before it is dismissed, so one reviewed since the listing is skipped, not overwritten."""

    for finding_id in finding_ids:
        finding = f"{path}/findings/{quote(finding_id, safe='')}"
        derived = "compass-" + hashlib.sha256(f"{key}\0{finding_id}".encode()).hexdigest()
        try:
            status = client.request("GET", finding).get("status")
            if status != "proposed":
                result["skipped"].append({"finding_id": finding_id, "status": status})
                continue
            client.request(
                "POST", f"{finding}/dismiss", payload={"note": note}, idempotency_key=derived
            )
        except CloudApiError as exc:
            if exc.status_code == 404 and exc.code is None:
                result["missing"].append(finding_id)
                continue
            if exc.status_code == 409:
                # Compass refuses this finding's review, as a mission holding it would.
                refusal = {"finding_id": finding_id, "message": str(exc)}
                if exc.code is not None:
                    refusal["code"] = exc.code
                result["refused"].append(refusal)
                continue
            raise BulkDismissError(exc, dismissed=len(result["dismissed"])) from None
        except CloudAuthError as exc:
            raise BulkDismissError(_auth_failure(exc), dismissed=len(result["dismissed"])) from None
        result["dismissed"].append(finding_id)


def _auth_failure(exc: CloudAuthError) -> CloudApiError:
    """A login that failed to refresh partway, kept as the same category and message so the
    dismissed count and the key still reach the person."""

    return CloudApiError(exc.category, str(exc))


def _records(value: object, *, key: str = "finding_id") -> list[dict[str, Any]]:
    """The objects of a list in Cloud's answer, each with a string ``key``."""

    listed = cast("list[object]", value) if isinstance(value, list) else []
    records = [cast("dict[str, Any]", item) for item in listed if isinstance(item, dict)]
    return [item for item in records if isinstance(item.get(key), str)]


def _ids(value: object) -> list[str]:
    listed = cast("list[object]", value) if isinstance(value, list) else []
    return [item for item in listed if isinstance(item, str)] + [
        item["finding_id"] for item in _records(value)
    ]


def _outcomes(value: object, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    outcomes = []
    for item in _records(value):
        outcome: dict[str, Any] = {"finding_id": item["finding_id"]}
        for name in fields:
            text = safe_text(item.get(name), _MESSAGE_LIMIT)
            if text is not None:
                outcome[name] = text
        outcomes.append(outcome)
    return outcomes


def confirm_on_terminal(
    proposed: Sequence[Mapping[str, Any]], arguments: argparse.Namespace
) -> bool:
    """Show what a bulk dismiss would dismiss and ask, on standard error; refuse to guess when
    nobody is at a terminal to answer."""

    count = len(proposed)
    if not sys.stdin.isatty():
        raise CloudApiError(
            "confirmation_required",
            f"Dismissing {count} findings needs a yes: run it at a terminal, or pass --yes. "
            "Nothing was dismissed.",
        )
    print(preview(proposed, arguments.application), file=sys.stderr)
    print(f"Dismiss these {count} findings? [y/N] ", end="", file=sys.stderr, flush=True)
    answer = sys.stdin.readline().strip().lower()
    return answer in {"y", "yes"}


def preview(proposed: Sequence[Mapping[str, Any]], application: str) -> str:
    lines = [f"{len(proposed)} proposed findings of Agent {_cell(application)} to dismiss:", ""]
    lines.extend(_summary_table(summarize(proposed)))
    lines.append("")
    lines.extend(_findings_table(proposed[:_PREVIEW_ROWS]))
    if len(proposed) > _PREVIEW_ROWS:
        lines.append(f"... and {len(proposed) - _PREVIEW_ROWS} more.")
    return "\n".join(lines)


def render_findings(output: Mapping[str, Any]) -> str:
    """What `findings list` and `findings dismiss` print without `--json`."""

    operation = output["operation"]
    result = output["result"]
    if operation == "compass.findings.summary":
        groups = result["groups"]
        if not groups:
            return "No findings match."
        lines = _summary_table(groups)
        lines.extend(["", f"{result['total']} findings."])
        return "\n".join(lines)
    if operation == "compass.findings.list":
        items = result["items"]
        if not items:
            return "No findings match."
        lines = _findings_table(items)
        total = result.get("total", len(items))
        shown = f"{len(items)} of {total}" if total != len(items) else str(total)
        lines.extend(["", f"{shown} findings."])
        if result.get("next_cursor") is not None:
            cursor = _cell(result["next_cursor"], limit=_CURSOR_LIMIT)
            lines.append(f"More findings follow: pass --cursor {cursor} with the same options.")
        return "\n".join(lines)
    key = f"Idempotency key: {_cell(output.get('idempotency_key'), limit=200)}"
    if "matched" not in result:
        return f"Dismissed finding {_cell(result.get('finding_id'))}.\n{key}"
    return "\n".join([*_dismissed_lines(result), key])


def _dismissed_lines(result: Mapping[str, Any]) -> list[str]:
    rule = _cell(result["rule"])
    if not result["matched"]:
        return [f"No findings of rule {rule} match; nothing was dismissed."]
    dismissed = len(result["dismissed"])
    lines = [f"Dismissed {dismissed} of the {result['matched']} matching findings of rule {rule}."]
    reviewed = Counter(_cell(item.get("status")) for item in result["skipped"])
    if reviewed:
        counts = ", ".join(f"{count} {status}" for status, count in sorted(reviewed.items()))
        lines.append(f"Left as they were, already reviewed: {counts}.")
    for item in result["refused"]:
        reason = item.get("message") or item.get("code") or "refused"
        lines.append(f"Refused {_cell(item['finding_id'])}: {_cell(reason, limit=200)}")
    if result["missing"]:
        lines.append(f"No longer found: {', '.join(_cell(item) for item in result['missing'])}.")
    return lines


def render_failure(error: Mapping[str, Any]) -> str:
    lines = [f"Error: {error['message']}"]
    if "idempotency_key" in error:
        lines.append(f"Idempotency key: {_cell(error['idempotency_key'], limit=200)}")
    return "\n".join(lines)


def _findings_table(items: Sequence[Mapping[str, Any]]) -> list[str]:
    rows = [
        (
            _cell(item.get("finding_id")),
            _cell(item.get("status")),
            _cell(item.get("dimension")),
            _cell(item.get("score")),
            _cell(item.get("rule")),
            _cell(finding_tool(item)),
            _cell(item.get("agent")),
        )
        for item in items
    ]
    return _table(("FINDING", "STATUS", "DIMENSION", "SCORE", "RULE", "TOOL", "AGENT"), rows)


def _summary_table(groups: Sequence[Mapping[str, Any]]) -> list[str]:
    rows = [
        (
            _cell(group["rule"]),
            _cell(group["tool"]),
            _cell(group["dimension"]),
            str(group["findings"]),
            str(group["proposed"]),
            str(group["accepted"]),
            str(group["dismissed"]),
            _cell(group["top_score"]),
        )
        for group in groups
    ]
    return _table(
        (
            "RULE",
            "TOOL",
            "DIMENSION",
            "FINDINGS",
            "PROPOSED",
            "ACCEPTED",
            "DISMISSED",
            "TOP SCORE",
        ),
        rows,
    )
