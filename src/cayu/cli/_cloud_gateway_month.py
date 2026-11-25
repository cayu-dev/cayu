"""`cayu cloud gateway usage month`: one month's Gateway usage and cost for the Organization.

Cloud sums the requests the Gateway admitted in one UTC calendar month, by the time it received
each, into one row per model and price (and per evaluation, for an evaluation's own key), each
with its requests, tokens and cost, then the Organization's totals (`/v1/billing/usage`). Every
number comes from the request's own record, at the price it was admitted at, so a past month's
report doesn't change. It is measured usage, not an invoice: `reconciliation` says
`not_verified`. Only an Organization admin reads it, and only for the Organization the command
acts in.
"""

from __future__ import annotations

import argparse
import re
from datetime import UTC, datetime
from typing import Any

from cayu.cli._cloud_api import CloudApiClient

_USAGE = "/v1/billing/usage"
_MONTH = re.compile(r"[0-9]{4}-(0[1-9]|1[0-2])")


def add_month_parser(actions: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `month` to `cayu cloud gateway usage`."""

    month = actions.add_parser(
        "month",
        help="One month's requests, tokens and cost per model, for the Organization.",
        description=(
            "One UTC month's Gateway requests, tokens and cost per model and price for the "
            "Organization the command acts in, with its totals, at the price each request was "
            "admitted at. Measured usage, not an invoice. Needs an Organization admin."
        ),
    )
    month.add_argument(
        "month",
        nargs="?",
        type=month_argument,
        help="The UTC month as YYYY-MM, such as 2026-11 (default: the current UTC month).",
    )


def month_argument(value: str) -> str:
    if _MONTH.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a UTC month as YYYY-MM, such as 2026-11")
    return value


def month_usage(arguments: argparse.Namespace, client: CloudApiClient) -> dict[str, Any]:
    """The Organization's report for ``arguments.month``, or for the current UTC month."""

    month = arguments.month or datetime.now(UTC).strftime("%Y-%m")
    return client.request("GET", _USAGE, query={"month": month})
