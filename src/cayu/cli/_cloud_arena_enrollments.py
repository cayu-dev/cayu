"""`cayu cloud admin arena-enrollments [--below-schema N]`: Arena case enrollments across
Organizations, with the eval corpus schema each enrolled Agent declares.

Arena's case service keeps its enrollments in its own ledger; Cloud reads them with the
install's Arena inventory token and answers a signed-in Cayu platform administrator at
``GET /v1/admin/arena/enrollments``. Each row is one enrolled Arena agent: its Organization, the
Cloud Agent it is bound to (if any), its latest enrollment, and ``corpus_schema``: the lowest
corpus schema it declares (``version``) and where that came from (``source``):

- ``recorded``: Arena recorded it when the Agent enrolled;
- ``current_release``: the enrollment predates that, so Cloud read it from the bound Agent's
  current Release;
- ``unknown``: neither, with ``reason``.

``--below-schema N`` keeps the rows whose shown schema is below N, with their source, and counts
the rows that show none (unknown, or no corpus declared). Cayu 0.16.3 and later read only corpus
schema 6, so ``--below-schema 6`` lists the Agents a newer Arena would refuse at their next
enrollment until they re-export their corpora.

It prints a table by default; ``--json`` prints the JSON envelope the other ``cayu cloud``
commands print.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any, cast

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_releases import add_output_argument
from cayu.cli._cloud_settings import _cell, _table

PATH = "/v1/admin/arena/enrollments"
_SHORT = 12


def add_arena_enrollments_parser(
    actions: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = actions.add_parser(
        "arena-enrollments",
        help="List Arena case enrollments across Organizations, with each Agent's corpus schema.",
        description=(
            "List every Arena case enrollment, across Organizations: its Organization, the "
            "Cloud Agent it is bound to, the enrolled revision, and the lowest eval corpus "
            "schema the Agent declares, with where that came from: `recorded` (by Arena at "
            "enrollment), `current_release` (read from the bound Agent's current Cloud "
            "Release) or `unknown`."
        ),
    )
    parser.add_argument(
        "--below-schema",
        type=_schema_version,
        metavar="N",
        help=(
            "Only rows whose shown corpus schema is below N (for example 6, the only schema "
            "Cayu 0.16.3 and later read). Rows with no known schema are counted, not listed."
        ),
    )
    add_output_argument(parser, render_arena_enrollments)


def _schema_version(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1 or not value.isdigit():
        raise argparse.ArgumentTypeError("N must be a whole number, 1 or more.")
    return number


def arena_enrollments_command(
    arguments: argparse.Namespace, *, client: CloudApiClient
) -> dict[str, Any]:
    items = _enrollments(client)
    below: int | None = arguments.below_schema
    result: dict[str, Any] = {"items": items}
    if below is not None:
        result = {
            "items": [item for item in items if _below(item, below)],
            "below_schema": below,
            "without_schema": sum(1 for item in items if _version(item) is None),
        }
    return {"operation": "admin.arena-enrollments", "result": result}


def _enrollments(client: CloudApiClient) -> list[Mapping[str, Any]]:
    try:
        answer = client.request("GET", PATH)
    except CloudApiError as exc:
        # The route takes no parameters, so a 404 without one of Cloud's codes, or a 405,
        # means the route itself is missing.
        if exc.status_code == 405 or (exc.status_code == 404 and exc.code is None):
            raise CloudApiError(
                "arena_enrollments_unsupported",
                "This Cayu Cloud doesn't list Arena enrollments yet.",
                status_code=exc.status_code,
            ) from None
        if exc.status_code == 403:
            raise CloudApiError(
                "platform_admin_required",
                "Listing Arena enrollments needs a signed-in Cayu platform administrator "
                "(`cayu cloud login`).",
                status_code=403,
            ) from None
        if exc.code is not None and exc.code.startswith("arena_inventory_"):
            # Cloud's own refusal, whose message names the setting to fix.
            raise CloudApiError(
                exc.code, str(exc), status_code=exc.status_code, code=exc.code
            ) from None
        raise
    items = answer.get("items")
    if not isinstance(items, list) or not all(isinstance(item, Mapping) for item in items):
        raise CloudApiError("api_response_invalid", "The Arena enrollment list is invalid.")
    return cast("list[Mapping[str, Any]]", items)


def _below(item: Mapping[str, Any], below: int) -> bool:
    version = _version(item)
    return version is not None and version < below


def _schema(item: Mapping[str, Any]) -> Mapping[str, Any]:
    value = item.get("corpus_schema")
    return cast("Mapping[str, Any]", value) if isinstance(value, Mapping) else {}


def _version(item: Mapping[str, Any]) -> int | None:
    version = _schema(item).get("version")
    return version if type(version) is int else None


def render_arena_enrollments(result: Mapping[str, Any]) -> str:
    listing: Mapping[str, Any] = result["result"]
    items: Sequence[Mapping[str, Any]] = listing["items"]
    below = listing.get("below_schema")
    lines: list[str] = []
    if items:
        rows = [_row(item) for item in items]
        lines = _table(
            ("ORGANIZATION", "AGENT", "CLOUD AGENT", "REVISION", "ENROLLMENT", "SCHEMA", "SOURCE"),
            rows,
        )
        lines += [
            "",
            "SOURCE: recorded = Arena recorded it at enrollment; current_release = read from "
            "the bound Agent's current Cloud Release; unknown = neither (reason in brackets).",
        ]
    elif below is None:
        lines = ["Arena has no case enrollments."]
    else:
        lines = [f"No enrollment shows a corpus schema below {below}."]
    hidden = listing.get("without_schema")
    if below is not None and type(hidden) is int and hidden:
        noun, verb = (
            ("enrollment shows", "isn't") if hidden == 1 else ("enrollments show", "aren't")
        )
        lines += [
            "",
            f"{hidden} {noun} no corpus schema (unknown, or no corpus declared) and {verb} "
            "listed; run without --below-schema to see them.",
        ]
    errors = [_cell(item.get("agent_id")) for item in items if item.get("arena_error")]
    if errors:
        lines += ["", f"Arena couldn't read the enrollment of: {', '.join(errors)}."]
    return "\n".join(lines)


def _row(item: Mapping[str, Any]) -> tuple[str, ...]:
    enrollment = item.get("enrollment")
    enrollment = enrollment if isinstance(enrollment, Mapping) else {}
    schema = _schema(item)
    revision = enrollment.get("source_revision")
    fingerprint = enrollment.get("fingerprint")
    sequence = enrollment.get("sequence")
    enrolled = (
        "-"
        if type(sequence) is not int
        else f"#{sequence}"
        + (f" {_cell(fingerprint)[:_SHORT]}" if isinstance(fingerprint, str) else "")
    )
    version = schema.get("version")
    source = _cell(schema.get("source"))
    reason = schema.get("reason")
    if reason:
        source += f" ({_cell(reason)})"
    if type(version) is int:
        shown = str(version)
        if any(
            isinstance(corpus, Mapping) and corpus.get("problem")
            for corpus in schema.get("corpora") or ()
        ):
            shown += " (some unread)"
    else:
        shown = "-"
    application = item.get("application_id")
    return (
        _cell(item.get("organization_id")),
        _cell(item.get("agent_id")),
        _cell(application) if application else "unbound",
        _cell(revision)[:_SHORT] if isinstance(revision, str) else "-",
        enrolled,
        shown,
        source,
    )
