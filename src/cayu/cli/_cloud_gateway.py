"""`cayu cloud gateway`: the Organization's Gateway keys, model policy, limits and usage.

The Gateway is Cayu Cloud's model gateway; Agents and people call models through it with
Gateway keys. These commands use the install's Gateway management API, which a Cayu Cloud
serves when its operator turns it on (with WorkOS or OIDC sign-in):

- `keys` creates, lists, shows, audits, rotates, narrows and revokes keys
  (`/v1/gateway/keys`). A key's token is shown once, when it is created or rotated: printed in
  the result, or written to `--key-file`, a new file only its owner can read.
- `models` reads the catalog (`/v1/gateway/catalog`), the models an Organization, Agent or key
  may use, and the Runtime's model status for an Agent; `models set` changes the
  Organization's or an Agent's model policy (`/v1/gateway/policies/{read,edit}`).
- `limits` reads and sets spending caps at Organization, Agent and key scope, and alert
  budgets (`/v1/gateway/alerts/configurations`).
- `usage` reads an Agent's settled requests and spend (`/v1/gateway/agents/{id}/usage`),
  one request (`/v1/gateway/requests/{id}`), and the Organization's month
  (`_cloud_gateway_month`).

Organization admins manage everything. Personal keys (a developer's own, within an admin's
allowance) exist only where the install offers them (`/v1/me`'s
`gateway_personal_keys_available`): none do until the Gateway can check a person's membership
when serving, and OIDC installs never do. Every change takes a fresh read first, so it starts
from the current revision.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_api_keys import new_private_file, write_private_file
from cayu.cli._cloud_gateway_month import add_month_parser, month_usage
from cayu.cli._cloud_reference import APPLICATION_HELP
from cayu.cli._cloud_settings import _table
from cayu.cli._output import terminal_safe

_KEYS = "/v1/gateway/keys"
_CATALOG = "/v1/gateway/catalog"
_ALLOWANCE = "/v1/gateway/delegations/mine"
_POLICY_READ = "/v1/gateway/policies/read"
_POLICY_EDIT = "/v1/gateway/policies/edit"
_ALERTS_READ = "/v1/gateway/alerts/configurations/read"
_ALERTS_EDIT = "/v1/gateway/alerts/configurations"
_REQUESTS = "/v1/gateway/requests"
_AGENTS = "/v1/gateway/agents"

KEY_SCOPES = ("inference", "receipts")
#: The periods an admin caps or budgets; a `lifetime` cap is the Gateway's own (an evaluation's).
PERIODS = ("daily", "weekly", "monthly")
_LIFETIME = "lifetime"
_DEFAULT_THRESHOLDS_BASIS_POINTS = (5000, 8000, 10000)
_MAX_MODELS = 256
_MAX_PAGE = 100
#: `usage agent --all` reads at most this many pages before it stops and gives the cursor.
MAX_USAGE_PAGES = 100
_NANO_PER_USD = 1_000_000_000
_USD = re.compile(r"(0|[1-9][0-9]{0,7})(\.[0-9]{1,9})?")
_DURATION = re.compile(r"([1-9][0-9]{0,4})([dh])")
_PERIOD_AMOUNT = re.compile(r"([a-z]+)=(.+)")

# The `/v1/me` answer's flag for each part of Gateway management.
_KEYS_FLAG = "gateway_keys_available"
_POLICY_FLAG = "gateway_policy_available"
_USAGE_FLAG = "gateway_usage_available"
_CATALOG_FLAG = "gateway_catalog_available"
_PERSONAL_FLAG = "gateway_personal_keys_available"
_PARTS = {
    _KEYS_FLAG: "keys",
    _POLICY_FLAG: "model policy and limits",
    _USAGE_FLAG: "usage",
    _CATALOG_FLAG: "model catalog",
}

MANAGEMENT_OFF = (
    "This Cayu Cloud doesn't serve Gateway management. Its operator turns it on where the "
    "Gateway runs and people sign in through WorkOS or OIDC (the chart's "
    "gateway.management.enabled)."
)
MANAGEMENT_UNKNOWN = (
    "This Cayu Cloud has no Gateway management API; it may be older than Gateway management."
)
MANAGEMENT_UNCHECKED = (
    "Cayu Cloud doesn't serve this Gateway route, and whether it serves Gateway management "
    "couldn't be checked (reading `/v1/me` failed). Run the command again."
)
ADMIN_REQUIRED = (
    "Only an Organization admin manages the Gateway's keys, models, limits and usage. An "
    "Organization API key can't manage the Gateway; sign in with `cayu cloud login`."
)
PERSONAL_KEYS_UNAVAILABLE = (
    "Personal Gateway keys aren't available on this install yet. An Organization admin can "
    "create a key for an Agent or a service key instead (`cayu cloud gateway keys create`)."
)
# A key addressed by id is refused alike when it doesn't exist and when it isn't yours.
KEY_NOT_ALLOWED = (
    "Key {key_id} wasn't found, or you aren't allowed to manage it: an Organization admin "
    "manages the Organization's keys, and anyone else only their own personal keys. Check the "
    "id with `cayu cloud gateway keys list`."
)
# The Gateway names the check that refused a `forbidden` in its `request_id`.
_FORBIDDEN = {
    "key-delegation": (
        "Your allowance doesn't cover this personal key: its models, Agent, scopes and expiry "
        "must fit what an admin allowed you (`cayu cloud gateway keys allowance`)."
    ),
    "platform-key": (
        "Cayu Cloud keeps this key for an Agent, an evaluation or Console: it can be revoked, "
        "but not made, rotated or restricted here."
    ),
}
NO_MODELS_NEXT = "Choose them with `cayu cloud gateway models set --model MODEL`."
# What to do next after a Gateway refusal whose own message stays as it is.
_NEXT_STEPS = {
    "model_not_allowed": "See the models allowed with `cayu cloud gateway models show`.",
}


def _error(category: str, message: str) -> Exception:
    """A `gateway` command's own refusal, reported like every `cayu cloud` command's."""

    from cayu.cli.cloud import CloudCommandError

    return CloudCommandError(category, message)


def add_gateway_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    gateway = commands.add_parser(
        "gateway",
        help="Manage the Gateway's keys, model policy, limits and usage.",
        description=(
            "Manage the Organization's Gateway: the keys Agents and people call models with, "
            "the models they may use, spending limits and alert budgets, and usage. Needs a "
            "signed-in Organization admin (`cayu cloud login`); a developer manages only their "
            "own personal keys."
        ),
    )
    areas = gateway.add_subparsers(dest="gateway_command", required=True)
    _add_keys_parser(areas)
    _add_models_parser(areas)
    _add_limits_parser(areas)
    _add_usage_parser(areas)


def _add_keys_parser(areas: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    keys = areas.add_parser(
        "keys",
        help="Create, list, rotate, restrict or revoke Gateway keys.",
        description="Manage Gateway keys. A key's token is shown once, at create or rotate.",
    )
    actions = keys.add_subparsers(dest="gateway_keys_command", required=True)
    create = actions.add_parser(
        "create",
        help="Create a key; its token is shown once.",
        description=(
            "Create a Gateway key: a service key for the Organization or one Agent "
            "(--application), or with --personal your own key where the install offers personal "
            "keys. Its token is printed once in the result, or written to --key-file."
        ),
    )
    create.add_argument("--name", required=True, help="What holds the key, such as ci-evals.")
    _add_models_argument(create, required=True, help="A model the key may call; repeat for more.")
    create.add_argument(
        "--expires",
        type=expiry,
        required=True,
        help="When the key stops working: a duration such as 30d or 12h, or a time with its "
        "offset, such as 2027-01-31T00:00:00Z.",
    )
    create.add_argument("--application", help=f"Bind the key to this Agent. {APPLICATION_HELP}")
    create.add_argument(
        "--personal",
        action="store_true",
        help="Make it your own personal key, within your allowance, where the install offers "
        "personal keys (none do yet; never on OIDC). Without it, a service key (admins).",
    )
    _add_scope_argument(create)
    create.add_argument(
        "--cap",
        action="append",
        type=period_amount,
        default=[],
        metavar="PERIOD=USD",
        help="A spending cap on the key, such as monthly=25; repeat per period. Needs "
        "--key-file, so a cap Cloud refuses never loses the token.",
    )
    _add_key_file_argument(create)

    listed = actions.add_parser(
        "list",
        help="List keys, never their tokens.",
        description="List the Organization's keys (admins), one Agent's, or your own (--mine).",
    )
    whose = listed.add_mutually_exclusive_group()
    whose.add_argument("--application", help=f"Only this Agent's keys. {APPLICATION_HELP}")
    whose.add_argument(
        "--mine",
        action="store_true",
        help="Only your own personal keys, where the install offers personal keys.",
    )
    _add_page_arguments(listed)
    _add_json_argument(listed)
    listed.set_defaults(_cloud_render=render_keys)

    show = actions.add_parser("show", help="Show one key.", description="Show one key's metadata.")
    show.add_argument("key_id", help="The key's id, from `gateway keys list`.")
    audit = actions.add_parser(
        "audit",
        help="List a key's changes.",
        description="List who created, restricted, rotated or revoked the key, and when.",
    )
    audit.add_argument("key_id", help="The key's id.")
    _add_page_arguments(audit)

    rotate = actions.add_parser(
        "rotate",
        help="Replace a key's token; the old one stops working.",
        description=(
            "Rotate a key: Cloud issues a new key with the same models, scopes, expiry and "
            "limits, and the old token stops working. The new token is shown once."
        ),
    )
    rotate.add_argument("key_id", help="The key's id.")
    _add_key_file_argument(rotate)

    restrict = actions.add_parser(
        "restrict",
        help="Narrow a key's models, scopes or expiry.",
        description=(
            "Narrow a key: fewer models, fewer scopes or an earlier expiry. Cloud refuses "
            "widening a key; create a new one instead. What isn't given is kept."
        ),
    )
    restrict.add_argument("key_id", help="The key's id.")
    _add_models_argument(restrict, required=False, help="A model the key keeps; repeat for more.")
    restrict.add_argument("--expires", type=expiry, help="An earlier expiry, as for create.")
    restrict.add_argument(
        "--scope", action="append", choices=KEY_SCOPES, help="A scope the key keeps."
    )

    revoke = actions.add_parser(
        "revoke",
        help="Stop a key working, at once.",
        description="Revoke a key; requests it already started run to settlement.",
    )
    revoke.add_argument("key_id", help="The key's id.")

    actions.add_parser(
        "allowance",
        help="Show what personal keys you may create.",
        description=(
            "Show your allowance for personal keys: the models, Agents, scopes and latest "
            "expiry an admin allowed you, or null when you have none. Refused where the "
            "install offers no personal keys."
        ),
    )


def _add_models_parser(areas: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    models = areas.add_parser(
        "models",
        help="Read the catalog and set the models the Organization or an Agent may use.",
        description=(
            "Read the Gateway's model catalog, the models the Organization, an Agent or a key "
            "may use, and the Runtime's model status; set the Organization's or an Agent's "
            "models and default model."
        ),
    )
    actions = models.add_subparsers(dest="gateway_models_command", required=True)
    actions.add_parser(
        "catalog",
        help="List the models the Gateway serves, with prices.",
        description="List the install's Gateway model catalog, with each model's prices.",
    )
    show = actions.add_parser(
        "show",
        help="Show the models the Organization, an Agent or a key may use.",
        description=(
            "Show the model policy at Organization scope, or for an Agent or key beside the "
            "Organization's: the models each allows, the default and whether it is locked."
        ),
    )
    _add_target_arguments(show, key=True)
    _add_json_argument(show)
    show.set_defaults(_cloud_render=render_models)

    change = actions.add_parser(
        "set",
        help="Set the Organization's or an Agent's models and default model.",
        description=(
            "Change the Organization's model policy, or an Agent's with --application. "
            "--model replaces the models it allows; --default and --lock choose its default. "
            "What isn't given is kept. An Agent's models must be among the Organization's."
        ),
    )
    _add_target_arguments(change, key=False)
    _add_models_argument(
        change, required=False, help="A model it allows; repeat. Replaces the list."
    )
    default = change.add_mutually_exclusive_group()
    default.add_argument("--default", dest="default_model", help="The default model.")
    default.add_argument("--no-default", action="store_true", help="Remove the default model.")
    lock = change.add_mutually_exclusive_group()
    lock.add_argument(
        "--lock",
        action="store_true",
        help="Lock every Agent to the Organization's default model (Organization only).",
    )
    lock.add_argument("--unlock", action="store_true", help="Remove the lock.")
    _add_policy_argument(change)

    status = actions.add_parser(
        "status",
        help="Show which model an Agent's Runtime has installed.",
        description=(
            "Show each of the Agent's Runtime replicas: the model it installed, whether its "
            "report is fresh, and why it refused a policy, if it did."
        ),
    )
    status.add_argument("--application", required=True, help=APPLICATION_HELP)
    _add_page_arguments(status)


def _add_limits_parser(areas: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    limits = areas.add_parser(
        "limits",
        help="Show and set spending caps and alert budgets.",
        description=(
            "Read and set the Gateway's spending caps (requests past a cap are refused) and "
            "alert budgets (admins are told as spend crosses thresholds), at Organization, "
            "Agent (--application) or key (--key) scope."
        ),
    )
    actions = limits.add_subparsers(dest="gateway_limits_command", required=True)
    show = actions.add_parser(
        "show",
        help="Show the caps and alert budgets that apply.",
        description=(
            "Show the caps of the scope and of the Organization beside it, and the scope's "
            "alert budgets."
        ),
    )
    _add_target_arguments(show, key=True)
    _add_json_argument(show)
    show.set_defaults(_cloud_render=render_limits)

    change = actions.add_parser(
        "set",
        help="Set or remove spending caps.",
        description=(
            "Set caps (--cap monthly=100) or remove them (--remove-cap weekly) at the scope. "
            "Other periods' caps are kept; zero forbids spend."
        ),
    )
    _add_target_arguments(change, key=True)
    _add_period_arguments(change, "cap", "A spending cap, such as monthly=100 (USD).")
    _add_policy_argument(change)

    alerts = actions.add_parser(
        "alerts",
        help="Set or remove alert budgets.",
        description=(
            "Set alert budgets (--budget monthly=500) or remove them (--remove-budget daily). "
            "Admins are alerted as spend crosses each threshold of a budget."
        ),
    )
    _add_target_arguments(alerts, key=True)
    _add_period_arguments(alerts, "budget", "An alert budget, such as monthly=500 (USD).")
    alerts.add_argument(
        "--thresholds",
        type=thresholds,
        help="Percentages of the budgets set here to alert at, such as 50,80,100 (the "
        "default for a new budget). Budgets not set here keep theirs.",
    )


def _add_usage_parser(areas: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    usage = areas.add_parser(
        "usage",
        help="Read an Agent's settled requests and spend, one request, or a month's usage.",
        description=(
            "Read Gateway usage: an Agent's settled requests and spend, one request, or the "
            "Organization's usage and cost for one month."
        ),
    )
    actions = usage.add_subparsers(dest="gateway_usage_command", required=True)
    agent = actions.add_parser(
        "agent",
        help="An Agent's settled requests, with spend by model.",
        description=(
            "List the Agent's settled requests with their cost, and totals by model, model "
            "level and model slot. One page by default; --all reads every page (up to "
            f"{MAX_USAGE_PAGES}), and --since/--until keep the requests made in that time. "
            "An evaluation's requests count apart, as evaluation spend."
        ),
    )
    agent.add_argument("--application", required=True, help=APPLICATION_HELP)
    _add_page_arguments(agent)
    agent.add_argument("--all", action="store_true", help="Read every page and total them.")
    agent.add_argument(
        "--since", type=instant, help="Only requests made at or after this time (implies --all)."
    )
    agent.add_argument(
        "--until", type=instant, help="Only requests made before this time (implies --all)."
    )
    _add_json_argument(agent)
    agent.set_defaults(_cloud_render=render_usage)

    request = actions.add_parser(
        "request",
        help="Show one Gateway request and its settlement.",
        description="Show one Gateway request: its key, model, status and settled cost.",
    )
    request.add_argument("request_id", help="The request's id (X-Cayu-Request-Id).")
    request.add_argument(
        "--evidence", action="store_true", help="Show its retained evidence instead."
    )
    add_month_parser(actions)


def _add_models_argument(parser: argparse.ArgumentParser, *, required: bool, help: str) -> None:
    parser.add_argument(
        "--model", dest="models", action="append", required=required, metavar="MODEL", help=help
    )


def _add_scope_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scope",
        action="append",
        choices=KEY_SCOPES,
        help="What the key may do: inference (the default) and receipts; repeat for both.",
    )


def _add_key_file_argument(parser: argparse.ArgumentParser) -> None:
    # Named as `api-keys create --key-file`: the file holds the key's token, the key itself.
    parser.add_argument(
        "--key-file",
        type=Path,
        help="Write the key's token to this new file, only you can read, instead of printing it.",
    )


def _add_target_arguments(parser: argparse.ArgumentParser, *, key: bool) -> None:
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--application", help=f"An Agent's scope instead of the Organization's. {APPLICATION_HELP}"
    )
    if key:
        target.add_argument("--key", dest="key_id", help="A key's scope, by its id.")


def _add_policy_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--policy",
        dest="policy_id",
        help="The policy to change, when the scope has more than one (`show` lists them).",
    )


def _add_period_arguments(parser: argparse.ArgumentParser, name: str, help: str) -> None:
    parser.add_argument(
        f"--{name}", dest="amounts", action="append", type=period_amount, default=[], help=help
    )
    parser.add_argument(
        f"--remove-{name}",
        dest="removed",
        action="append",
        choices=PERIODS,
        default=[],
        help=f"Remove the {name} of this period; repeat for more.",
    )


def _add_page_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--limit",
        type=_page_size,
        help=f"At most this many items (1 to {_MAX_PAGE}).",
    )
    parser.add_argument("--cursor", help="Go on after an earlier page, from its next_cursor.")


def _add_json_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true", help="Print the JSON envelope instead of text."
    )


def _page_size(value: str) -> int:
    if not value.isdecimal() or not 1 <= int(value) <= _MAX_PAGE:
        raise argparse.ArgumentTypeError(f"must be a whole number from 1 to {_MAX_PAGE}")
    return int(value)


def expiry(value: str) -> str:
    """A key's expiry: a duration from now (``30d``, ``12h``) or an aware ISO 8601 time."""

    duration = _DURATION.fullmatch(value)
    if duration is not None:
        amount = int(duration.group(1))
        delta = timedelta(days=amount) if duration.group(2) == "d" else timedelta(hours=amount)
        return (datetime.now(UTC) + delta).isoformat()
    return instant(value).isoformat()


def instant(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        parsed = None
    if parsed is None or parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(
            "must be a time with its offset, such as 2027-01-31T00:00:00Z"
        )
    return parsed.astimezone(UTC)


def period_amount(value: str) -> tuple[str, int]:
    """``PERIOD=USD`` as the period and the amount in nano-USD."""

    match = _PERIOD_AMOUNT.fullmatch(value)
    if match is None or match.group(1) not in PERIODS:
        raise argparse.ArgumentTypeError(
            "must be PERIOD=USD with a period of daily, weekly or monthly, such as monthly=100"
        )
    return match.group(1), dollars_to_nano(match.group(2))


def dollars_to_nano(value: str) -> int:
    if _USD.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "must be a USD amount with at most nine decimal places, such as 25 or 0.50"
        )
    whole, _, fraction = value.partition(".")
    return int(whole) * _NANO_PER_USD + int(fraction.ljust(9, "0") or "0")


def nano_to_dollars(value: object) -> str:
    if type(value) is not int or value < 0:
        return "-"
    whole, fraction = divmod(value, _NANO_PER_USD)
    return f"${whole:,}.{str(fraction).rjust(9, '0').rstrip('0').ljust(2, '0')}"


def thresholds(value: str) -> tuple[int, ...]:
    """``50,80,100`` as basis points, each a percentage from 1 to 100, distinct and ordered."""

    parts = value.split(",")
    if not all(part.isdecimal() and 1 <= int(part) <= 100 for part in parts):
        raise argparse.ArgumentTypeError("must be percentages from 1 to 100, such as 50,80,100")
    return tuple(sorted({int(part) * 100 for part in parts}))


def gateway_command(
    arguments: argparse.Namespace, *, client: CloudApiClient, application_id: str | None
) -> dict[str, Any]:
    """Run one `gateway` action; ``application_id`` is the Agent its --application names."""

    area = arguments.gateway_command
    action = getattr(arguments, f"gateway_{area}_command")
    if (area, action) == ("usage", "month"):
        # Billing's report, not Gateway management: Cloud's own refusals say why.
        return {"operation": "gateway.usage.month", "result": month_usage(arguments, client)}
    run, flag = _ACTIONS[(area, action)]
    try:
        result = run(arguments, client, application_id)
    except CloudApiError as exc:
        raise _explained(
            exc,
            client,
            flag,
            key_id=getattr(arguments, "key_id", None),
            rotating=action == "rotate",
        ) from None
    return {"operation": f"gateway.{area}.{action}", "result": result}


def _explained(
    exc: CloudApiError,
    client: CloudApiClient,
    flag: str,
    *,
    key_id: str | None = None,
    rotating: bool = False,
) -> Exception:
    """``exc`` in words: management turned off, a scope not enrolled yet, or who may act."""

    if _route_missing(exc):
        try:
            identity = client.request("GET", "/v1/me")
        except CloudApiError:
            return _error("gateway_management_unknown", MANAGEMENT_UNCHECKED)
        served = identity.get(_KEYS_FLAG)
        if served is False:
            return _error("gateway_management_off", MANAGEMENT_OFF)
        if identity.get(flag) is False:
            return _error(
                "gateway_management_off",
                f"This Cayu Cloud serves Gateway keys, but not its {_PARTS[flag]}.",
            )
        if served is None:
            return _error("gateway_management_unavailable", MANAGEMENT_UNKNOWN)
        return exc
    detail = exc.body or {}
    if exc.code == "personal_keys_unavailable":
        return _with_message(exc, PERSONAL_KEYS_UNAVAILABLE)
    if exc.code == "key_publication_unknown":
        # Cloud may have published the key, so nothing about it is settled.
        message = _publication_unknown(detail, key_id)
        if rotating and key_id is not None:
            message += f" Key {key_id}'s old token may already have stopped working."
        return _with_message(exc, message)
    if exc.status_code == 403 and exc.code in {None, "forbidden"}:
        return _with_message(exc, _forbidden(exc, detail, key_id))
    if exc.code in _NEXT_STEPS:
        # Another refusal keeps the Gateway's own code and message, with what to do next.
        return _with_message(exc, f"{exc} {_NEXT_STEPS[exc.code]}")
    if exc.code == "no_models":
        return _with_message(exc, f"{exc} {NO_MODELS_NEXT}")
    if exc.code == "not_enrolled":
        return _with_message(exc, f"{exc} Nothing was changed.")
    if exc.code in {"idempotency_conflict", "key_conflict", "alert_conflict"}:
        # The command read the current revision just before; anything it could tell would
        # repeat was refused before asking, so this is a change made in between.
        return _with_message(
            exc,
            f"Cayu Cloud API returned HTTP {exc.status_code}: it changed since this command "
            "read it, and nothing was changed. Run the command again.",
        )
    return exc


def _forbidden(exc: CloudApiError, detail: Mapping[str, Any], key_id: str | None) -> str:
    """Why a 403: the Gateway's own reason where it gives one, else neutral about a key."""

    reason = detail.get("request_id") if exc.code == "forbidden" else None
    if isinstance(reason, str) and reason in _FORBIDDEN:
        return _FORBIDDEN[reason]
    if exc.code == "forbidden" and key_id is not None:
        # The Gateway refuses a key that doesn't exist and one you may not manage alike.
        return KEY_NOT_ALLOWED.format(key_id=key_id)
    return ADMIN_REQUIRED


def _publication_unknown(detail: Mapping[str, Any], key_id: str | None) -> str:
    found = detail.get("key_id")
    named = found if isinstance(found, str) and found else key_id or "the key"
    return (
        f"Cloud couldn't confirm whether key {named} was published, and no token was delivered. "
        f"Check it with `cayu cloud gateway keys audit {named}` and revoke it with "
        f"`cayu cloud gateway keys revoke {named}` before making a replacement."
    )


def _route_missing(exc: CloudApiError) -> bool:
    return exc.status_code == 405 or (
        exc.status_code == 404 and exc.code is None and exc.detail in {None, "Not Found"}
    )


def _with_message(exc: CloudApiError, message: str) -> CloudApiError:
    return CloudApiError(
        exc.category,
        message,
        status_code=exc.status_code,
        code=exc.code,
        body=exc.body,
        transport_error=exc.transport_error,
        request=exc.request,
        retry_after=exc.retry_after,
    )


# Keys


def _create_key(
    arguments: argparse.Namespace, client: CloudApiClient, agent: str | None
) -> dict[str, Any]:
    caps = _period_amounts(arguments.cap)
    if caps and arguments.key_file is None:
        raise _error(
            "invalid_input",
            "--cap needs --key-file: the key is created before its caps are set, so its "
            "token is written to a file first.",
        )
    if arguments.personal:
        _require_personal_keys(client)
    payload: dict[str, object] = {
        "name": arguments.name,
        "owner_kind": "personal" if arguments.personal else "service",
        "agent_id": agent,
        "expires_at": arguments.expires,
        "allowed_models": _models(arguments.models),
        "scopes": _distinct(arguments.scope or ["inference"]),
    }
    lost = (
        f"Cloud may have created the key {arguments.name!r}, but its token is lost: find it "
        "with `cayu cloud gateway keys list` and revoke it."
    )
    result = _issue(client, "POST", _KEYS, payload, arguments.key_file, lost=lost)
    if caps:
        from cayu.cli.cloud import CloudCommandError

        key_id = result["key"]["key_id"]
        try:
            result["limits"] = _set_caps(
                client, _organization_id(client), "key", key_id, caps, (), policy_id=None
            )
        except (CloudApiError, CloudCommandError) as failure:
            exc = (
                _explained(failure, client, _POLICY_FLAG, key_id=key_id)
                if isinstance(failure, CloudApiError)
                else failure
            )
            raise _error(
                "gateway_key_caps_not_set",
                f"Created key {key_id}, with its token in {result['key_file']}, but its caps "
                f"weren't set: {exc} Set them with `cayu cloud gateway limits set --key "
                f"{key_id} --cap PERIOD=USD`, or revoke the key.",
            ) from None
    return result


def _list_keys(
    arguments: argparse.Namespace, client: CloudApiClient, agent: str | None
) -> dict[str, Any]:
    if arguments.mine:
        _require_personal_keys(client)
    query = _page_query(arguments, cursor="after")
    if agent is not None:
        query["agent"] = agent
    return client.request("GET", f"{_KEYS}/mine" if arguments.mine else _KEYS, query=query)


def _show_key(arguments: argparse.Namespace, client: CloudApiClient, _: str | None) -> Any:
    return _key(client, arguments.key_id)


def _audit_key(arguments: argparse.Namespace, client: CloudApiClient, _: str | None) -> Any:
    if arguments.cursor is not None and not arguments.cursor.isdecimal():
        raise _error("invalid_input", "A key audit's cursor is a whole number.")
    return client.request(
        "GET", f"{_key_path(arguments.key_id)}/audit", query=_page_query(arguments, "after")
    )


def _rotate_key(arguments: argparse.Namespace, client: CloudApiClient, _: str | None) -> Any:
    key = _key(client, arguments.key_id)
    _require_active(key)
    lost = (
        f"Cloud may have rotated key {arguments.key_id}, so its old token may already have "
        "stopped working, but the new token is lost: check with `cayu cloud gateway keys "
        f"audit {arguments.key_id}` and rotate the new key again."
    )
    result = _issue(
        client,
        "POST",
        f"{_key_path(arguments.key_id)}/rotate",
        {"expected_revision": key["revision"]},
        arguments.key_file,
        lost=lost,
    )
    return {**result, "replaced_key_id": arguments.key_id}


def _restrict_key(arguments: argparse.Namespace, client: CloudApiClient, _: str | None) -> Any:
    if arguments.models is None and arguments.expires is None and arguments.scope is None:
        raise _error("invalid_input", "Give --model, --scope or --expires to narrow the key.")
    key = _key(client, arguments.key_id)
    _require_active(key)
    models = key["allowed_models"] if arguments.models is None else _models(arguments.models)
    scopes = key["scopes"] if arguments.scope is None else _distinct(arguments.scope)
    expires = key["expires_at"] if arguments.expires is None else arguments.expires
    # Cloud refuses widening a key for good; say which part would widen it.
    widened = [model for model in models if model not in key["allowed_models"]]
    if widened:
        raise _widening(
            key, f"it may call only {', '.join(key['allowed_models'])}, not {', '.join(widened)}"
        )
    added = [scope for scope in scopes if scope not in key["scopes"]]
    if added:
        raise _widening(
            key, f"it has only the scopes {', '.join(key['scopes'])}, not {', '.join(added)}"
        )
    if arguments.expires is not None and _later(expires, key["expires_at"]):
        raise _widening(key, f"it expires at {key['expires_at']}; give an earlier --expires")
    return client.request(
        "PATCH",
        _key_path(arguments.key_id),
        payload={
            "expected_revision": key["revision"],
            "allowed_models": models,
            "scopes": scopes,
            "expires_at": expires,
        },
    )


def _revoke_key(arguments: argparse.Namespace, client: CloudApiClient, _: str | None) -> Any:
    key = _key(client, arguments.key_id)
    if key.get("revoked") is True:
        # Already what the command asks for: say so rather than send a refused change.
        return {**key, "already_revoked": True}
    return client.request(
        "POST",
        f"{_key_path(arguments.key_id)}/revoke",
        payload={"expected_revision": key["revision"]},
    )


def _allowance(_arguments: argparse.Namespace, client: CloudApiClient, _: str | None) -> Any:
    # Refused rather than hidden while the install offers no personal keys, so a script learns
    # why it has no allowance.
    _require_personal_keys(client)
    try:
        return client.request("GET", _ALLOWANCE)
    except CloudApiError as exc:
        if _route_missing(exc):
            # An install without delegations (OIDC) has no personal keys to allow.
            raise _error("personal_keys_unavailable", PERSONAL_KEYS_UNAVAILABLE) from None
        raise


def _require_personal_keys(client: CloudApiClient) -> None:
    """Refuse before asking when Cloud says this install offers no personal keys."""

    try:
        offered = client.request("GET", "/v1/me").get(_PERSONAL_FLAG)
    except CloudApiError:
        return
    if offered is False:
        raise _error("personal_keys_unavailable", PERSONAL_KEYS_UNAVAILABLE)


def _require_active(key: Mapping[str, Any]) -> None:
    if key.get("revoked") is True:
        raise _error(
            "gateway_key_revoked",
            f"Key {key['key_id']} is revoked, and a revoked key can't be changed. Create a new "
            "key instead.",
        )


def _widening(key: Mapping[str, Any], why: str) -> Exception:
    return _error(
        "gateway_key_widening",
        f"Key {key['key_id']} can only be narrowed: {why}. Create a new key to widen it.",
    )


def _later(candidate: str, current: object) -> bool:
    try:
        return datetime.fromisoformat(candidate) > datetime.fromisoformat(str(current))
    except ValueError:
        return False


def _key(client: CloudApiClient, key_id: str) -> dict[str, Any]:
    key = client.request("GET", _key_path(key_id))
    if key.get("key_id") != key_id or type(key.get("revision")) is not int:
        raise CloudApiError("api_response_invalid", "Cayu Cloud answered another key.")
    return key


def _key_path(key_id: str) -> str:
    return f"{_KEYS}/{quote(key_id, safe='')}"


def _issue(
    client: CloudApiClient,
    method: str,
    path: str,
    payload: dict[str, object],
    key_file: Path | None,
    *,
    lost: str,
) -> dict[str, Any]:
    """Ask Cloud for a key and deliver its token once: in the result, or to ``key_file``.

    The file is created before Cloud is asked, so a path that can't be written refuses
    before any key exists; it is removed again when no token comes back.
    """

    path_on_disk = None if key_file is None else key_file.expanduser()
    descriptor = None if path_on_disk is None else new_private_file(path_on_disk)
    try:
        issued = client.request(method, path, payload=payload)
    except CloudApiError as exc:
        _discard(descriptor, path_on_disk)
        if _refused(exc):
            raise
        raise _with_message(exc, f"{exc} {lost}") from None
    except KeyboardInterrupt:
        _discard(descriptor, path_on_disk)
        print(lost, file=sys.stderr)
        raise
    key, token = issued.get("key"), issued.get("secret")
    if not isinstance(key, dict) or not isinstance(token, str) or not token:
        _discard(descriptor, path_on_disk)
        raise CloudApiError("api_response_invalid", f"Cayu Cloud's answer was invalid. {lost}")
    if descriptor is None or path_on_disk is None:
        return {"key": key, "token": token}
    try:
        write_private_file(descriptor, path_on_disk, token)
    except OSError as exc:
        raise _error(
            "local_state_unavailable",
            f"Cloud issued key {key.get('key_id')}, but its token couldn't be written to "
            f"{path_on_disk} ({exc.strerror}); rotate the key to get a new token.",
        ) from None
    return {"key": key, "key_file": str(path_on_disk)}


def _discard(descriptor: int | None, path: Path | None) -> None:
    if descriptor is not None and path is not None:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _refused(exc: CloudApiError) -> bool:
    """Whether Cloud answered that it refused, so it made no key."""

    status = exc.status_code
    return (
        exc.category == "api_request_rejected"
        and status is not None
        and (status < 500 or exc.code == "not_enrolled")
    )


# Model policy and limits


def _catalog(_arguments: argparse.Namespace, client: CloudApiClient, _: str | None) -> Any:
    return client.request("GET", _CATALOG)


def _show_models(
    arguments: argparse.Namespace, client: CloudApiClient, agent: str | None
) -> dict[str, Any]:
    organization = _organization_id(client)
    scope, scope_id = _target(arguments, organization, agent)
    configuration = _read_policy(client, organization, scope, scope_id)
    rows = [row for row in configuration["rows"] if isinstance(row, dict)]
    policies = [policy for row in rows for policy in row.get("policies", [])]
    # A key calls only its own models, whatever its policies allow.
    ceilings = [policy["rules"]["allowed_models"] for policy in policies]
    if scope == "key":
        ceilings.append(_key(client, scope_id)["allowed_models"])
    allowed: list[str] | None = None
    for models in ceilings:
        allowed = list(models) if allowed is None else [m for m in allowed if m in models]
    locked = next(
        (
            policy["rules"].get("default_model")
            for policy in policies
            if policy["rules"].get("model_lock") is True
        ),
        None,
    )
    return {
        "organization_id": organization,
        "scope": scope,
        "scope_id": scope_id,
        "allowed_models": allowed or [],
        "default_model": locked
        or _default_model(rows, scope, scope_id)
        or _default_model(rows, "organization", organization),
        "locked_to": locked,
        "enrolled": len(rows) == len(configuration["rows"]),
        "configuration": configuration,
    }


def _set_models(
    arguments: argparse.Namespace, client: CloudApiClient, agent: str | None
) -> dict[str, Any]:
    if (
        arguments.models is None
        and arguments.default_model is None
        and not arguments.no_default
        and not arguments.lock
        and not arguments.unlock
    ):
        raise _error("invalid_input", "Give --model, --default, --no-default, --lock or --unlock.")
    organization = _organization_id(client)
    scope, scope_id = _target(arguments, organization, agent)
    if scope != "organization" and (arguments.lock or arguments.unlock):
        raise _error(
            "invalid_input", "Only the Organization's policy locks Agents to its default model."
        )
    models = None if arguments.models is None else _models(arguments.models)
    default_changes = arguments.default_model is not None or arguments.no_default

    def change(rules: dict[str, Any]) -> dict[str, Any]:
        if models is not None:
            rules["allowed_models"] = models
        if default_changes:
            rules["default_model"] = arguments.default_model
        if arguments.lock:
            rules["model_lock"] = True
        if arguments.unlock:
            rules.pop("model_lock", None)
        if rules.get("model_lock") is True and rules.get("default_model") is None:
            raise _error("invalid_input", "Choose a default model (--default) to lock Agents to.")
        default = rules.get("default_model")
        if default is not None and default not in rules["allowed_models"]:
            raise _error("invalid_input", f"The default model {default} isn't among its models.")
        return rules

    return _edit_policy(
        client,
        organization,
        scope,
        scope_id,
        change,
        policy_id=arguments.policy_id,
        siblings_default=arguments.default_model if default_changes else _UNCHANGED,
        new_models=models,
    )


def _model_status(
    arguments: argparse.Namespace, client: CloudApiClient, agent: str | None
) -> dict[str, Any]:
    assert agent is not None
    return client.request(
        "GET", f"{_AGENTS}/{quote(agent, safe='')}/replicas", query=_page_query(arguments)
    )


def _show_limits(
    arguments: argparse.Namespace, client: CloudApiClient, agent: str | None
) -> dict[str, Any]:
    organization = _organization_id(client)
    scope, scope_id = _target(arguments, organization, agent)
    configuration = _read_policy(client, organization, scope, scope_id)
    caps = [
        {
            "scope": row["principal"]["scope"],
            "scope_id": row["principal"]["scope_id"],
            "policy_id": policy["binding"]["policy_id"],
            "period": cap["period"],
            "ceiling_nano_usd": cap["ceiling_nano_usd"],
        }
        for row in configuration["rows"]
        if isinstance(row, dict)
        for policy in row.get("policies", [])
        for cap in policy["rules"].get("hard_limits", [])
    ]
    alerts = _read_alerts(client, scope, scope_id)
    return {
        "organization_id": organization,
        "scope": scope,
        "scope_id": scope_id,
        "caps": caps,
        "alert_budgets": alerts["configuration"],
        "configuration": configuration,
    }


def _set_limits(
    arguments: argparse.Namespace, client: CloudApiClient, agent: str | None
) -> dict[str, Any]:
    caps = _period_amounts(arguments.amounts)
    removed = _removed(arguments.removed, caps)
    if not caps and not removed:
        raise _error("invalid_input", "Give --cap PERIOD=USD or --remove-cap PERIOD.")
    organization = _organization_id(client)
    scope, scope_id = _target(arguments, organization, agent)
    return _set_caps(
        client, organization, scope, scope_id, caps, removed, policy_id=arguments.policy_id
    )


def _set_caps(
    client: CloudApiClient,
    organization: str,
    scope: str,
    scope_id: str,
    caps: Mapping[str, int],
    removed: Iterable[str],
    *,
    policy_id: str | None,
) -> dict[str, Any]:
    removed = set(removed)

    def change(rules: dict[str, Any]) -> dict[str, Any]:
        kept = [cap for cap in rules.get("hard_limits", []) if cap["period"] not in caps]
        rules["hard_limits"] = sorted(
            [cap for cap in kept if cap["period"] not in removed]
            + [
                {"schema_version": 1, "period": period, "ceiling_nano_usd": amount}
                for period, amount in caps.items()
            ],
            key=lambda cap: _PERIOD_ORDER.get(cap["period"], len(_PERIOD_ORDER)),
        )
        return rules

    return _edit_policy(client, organization, scope, scope_id, change, policy_id=policy_id)


def _set_alerts(
    arguments: argparse.Namespace, client: CloudApiClient, agent: str | None
) -> dict[str, Any]:
    budgets = _period_amounts(arguments.amounts)
    removed = _removed(arguments.removed, budgets)
    if not budgets and not removed:
        raise _error("invalid_input", "Give --budget PERIOD=USD or --remove-budget PERIOD.")
    if any(amount == 0 for amount in budgets.values()):
        raise _error("invalid_input", "An alert budget must be more than zero.")
    organization = _organization_id(client)
    scope, scope_id = _target(arguments, organization, agent)
    current = _read_alerts(client, scope, scope_id)
    before = current["configuration"]
    owner = current["scope_id"]
    existing = {item["period"]: item for item in (before or {}).get("rules", {}).get("budgets", [])}
    chosen = arguments.thresholds
    after_budgets = [
        item for period, item in existing.items() if period not in budgets and period not in removed
    ] + [
        {
            "schema_version": 1,
            "period": period,
            "amount_nano_usd": amount,
            "thresholds_basis_points": list(
                chosen
                or existing.get(period, {}).get("thresholds_basis_points")
                or _DEFAULT_THRESHOLDS_BASIS_POINTS
            ),
        }
        for period, amount in budgets.items()
    ]
    payload: dict[str, object] = {
        "operation_id": str(uuid.uuid4()),
        "before": before,
        "after": {
            "schema_version": 1,
            "organization_id": organization,
            "scope": scope,
            "scope_id": owner,
            "revision": (before or {}).get("revision", 0) + 1,
            "rules": {
                "schema_version": 1,
                "budgets": sorted(after_budgets, key=lambda item: _PERIOD_ORDER[item["period"]]),
            },
        },
    }
    if scope == "key":
        payload["key_id"] = scope_id
    return _with_operation(client, _ALERTS_EDIT, payload)


_PERIOD_ORDER = {period: index for index, period in enumerate((*PERIODS, _LIFETIME))}
_UNCHANGED = object()


def _edit_policy(
    client: CloudApiClient,
    organization: str,
    scope: str,
    scope_id: str,
    change: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    policy_id: str | None,
    siblings_default: object = _UNCHANGED,
    new_models: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Read the scope's policy, change one of its policies' rules and send the edit.

    The edit names what it read as its `before`, so Cloud refuses it if anything changed in
    between. A scope without a policy gets a new one; its models are ``new_models`` or, below
    the Organization, the models every policy above it allows.
    """

    configuration = _read_policy(client, organization, scope, scope_id)
    rows = configuration["rows"]
    if configuration.get("schedule") is None or any(row is None for row in rows):
        raise _not_enrolled(configuration, scope)
    row = _row(rows, scope, scope_id)
    policies = row.get("policies", [])
    policy = _chosen_policy(policies, policy_id)
    operation = str(uuid.uuid4())
    if policy is None:
        models = list(new_models) if new_models is not None else _inherited_models(rows, scope)
        if not models:
            raise _error(
                "invalid_input",
                "The Organization allows no models yet. " + NO_MODELS_NEXT
                if scope == "organization"
                else "No models are allowed above this scope. " + NO_MODELS_NEXT,
            )
        rules = {
            "schema_version": 1,
            "allowed_models": models,
            "default_model": None,
            "hard_limits": [],
        }
        new_id = f"cli-{operation}"
    else:
        if any(cap["period"] == _LIFETIME for cap in policy["rules"].get("hard_limits", [])):
            raise _error(
                "invalid_input",
                f"Policy {policy['binding']['policy_id']} has a lifetime cap, which only the "
                "Gateway sets; it can't be changed here.",
            )
        rules = dict(policy["rules"])
        new_id = policy["binding"]["policy_id"]
    changes = [
        {
            "schema_version": 1,
            "scope": scope,
            "scope_id": scope_id,
            "policy_id": new_id,
            "before": policy,
            "rules": change(rules),
        }
    ]
    if siblings_default is not _UNCHANGED:
        # A scope has one default model, so its other policies follow it (as the portal does).
        changes += [
            {
                "schema_version": 1,
                "scope": scope,
                "scope_id": scope_id,
                "policy_id": sibling["binding"]["policy_id"],
                "before": sibling,
                "rules": {**sibling["rules"], "default_model": siblings_default},
            }
            for sibling in policies
            if sibling is not policy
        ]
    changes.sort(key=lambda item: item["policy_id"])
    payload: dict[str, object] = {
        "schema_version": 1,
        "operation_id": operation,
        "before": {"schema_version": 1, "schedule": configuration["schedule"], "scopes": rows},
        "changes": changes,
    }
    return _with_operation(client, _POLICY_EDIT, payload)


def _with_operation(client: CloudApiClient, path: str, payload: dict[str, object]) -> Any:
    """Send one edit; when its outcome is unknown, say how to find out."""

    try:
        return client.request("POST", path, payload=payload)
    except CloudApiError as exc:
        if _refused(exc) and exc.code != "outcome_unknown":
            raise
        raise _with_message(
            exc,
            f"{exc} The change (operation {payload['operation_id']}) may have been applied: "
            "read it back with `cayu cloud gateway limits show` or `models show` before "
            "changing it again.",
        ) from None


def _read_policy(
    client: CloudApiClient, organization: str, scope: str, scope_id: str
) -> dict[str, Any]:
    keys = [["organization", organization]]
    if scope != "organization":
        keys.insert(0, [scope, scope_id])
    configuration = client.request("POST", _POLICY_READ, payload={"keys": keys})
    rows = configuration.get("rows")
    if (
        configuration.get("organization_id") != organization
        or not isinstance(rows, list)
        or len(rows) != len(keys)
        or any(
            row is not None and _principal(row) != key for row, key in zip(rows, keys, strict=True)
        )
    ):
        raise CloudApiError("api_response_invalid", "Cayu Cloud answered another policy.")
    return configuration


def _principal(row: Any) -> list[object] | None:
    principal = row.get("principal") if isinstance(row, dict) else None
    if not isinstance(principal, dict):
        return None
    return [principal.get("scope"), principal.get("scope_id")]


def _read_alerts(client: CloudApiClient, scope: str, scope_id: str) -> dict[str, Any]:
    """The scope's alert budgets, and the scope id they are kept under (a key's own owner)."""

    answer = client.request("POST", _ALERTS_READ, payload={"scope": scope, "scope_id": scope_id})
    owner = answer.get("scope_id", scope_id) if scope == "key" else scope_id
    configuration = answer.get("configuration")
    if not isinstance(owner, str) or (
        configuration is not None
        and (
            not isinstance(configuration, dict)
            or configuration.get("scope") != scope
            or configuration.get("scope_id") != owner
        )
    ):
        raise CloudApiError("api_response_invalid", "Cayu Cloud answered another alert budget.")
    return {"configuration": configuration, "scope_id": owner}


def _not_enrolled(configuration: Mapping[str, Any], scope: str) -> Exception:
    rows = configuration["rows"]
    organization = rows[-1]
    if configuration.get("schedule") is None or organization is None:
        what = "This Organization isn't"
    elif scope == "agent" and not any(
        policy["rules"].get("allowed_models") for policy in organization.get("policies", [])
    ):
        return _error("gateway_no_models", f"This Organization allows no models. {NO_MODELS_NEXT}")
    else:
        what = "This Agent isn't" if scope == "agent" else "This key isn't"
    return _error(
        "gateway_not_enrolled",
        f"{what} enrolled in the Gateway yet; nothing was changed. Retry in a few seconds.",
    )


def _row(rows: Sequence[Mapping[str, Any]], scope: str, scope_id: str) -> Mapping[str, Any]:
    return next(
        row
        for row in rows
        if row["principal"]["scope"] == scope and row["principal"]["scope_id"] == scope_id
    )


def _chosen_policy(
    policies: Sequence[dict[str, Any]], policy_id: str | None
) -> dict[str, Any] | None:
    if policy_id is not None:
        for policy in policies:
            if policy["binding"]["policy_id"] == policy_id:
                return policy
        raise _error(
            "invalid_input", f"The scope has no policy {policy_id}; `show` lists its policies."
        )
    if len(policies) > 1:
        names = ", ".join(policy["binding"]["policy_id"] for policy in policies)
        raise _error(
            "invalid_input", f"The scope has several policies ({names}); name one with --policy."
        )
    return policies[0] if policies else None


def _inherited_models(rows: Sequence[Mapping[str, Any]], scope: str) -> list[str]:
    allowed: list[str] | None = None
    for row in rows:
        if row["principal"]["scope"] == scope:
            continue
        for policy in row.get("policies", []):
            models = policy["rules"]["allowed_models"]
            allowed = list(models) if allowed is None else [m for m in allowed if m in models]
    return allowed or []


def _default_model(rows: Sequence[Mapping[str, Any]], scope: str, scope_id: str) -> str | None:
    for row in rows:
        if (row["principal"]["scope"], row["principal"]["scope_id"]) != (scope, scope_id):
            continue
        for policy in row.get("policies", []):
            if policy["rules"].get("default_model"):
                return policy["rules"]["default_model"]
    return None


def _target(arguments: argparse.Namespace, organization: str, agent: str | None) -> tuple[str, str]:
    if agent is not None:
        return "agent", agent
    key_id = getattr(arguments, "key_id", None)
    if key_id is not None:
        return "key", key_id
    return "organization", organization


def _organization_id(client: CloudApiClient) -> str:
    organization = client.request("GET", "/v1/me").get("organization_id")
    if not isinstance(organization, str) or not organization:
        raise CloudApiError(
            "api_response_invalid", "Cayu Cloud didn't say which Organization this acts in."
        )
    return organization


# Usage


def _agent_usage(
    arguments: argparse.Namespace, client: CloudApiClient, agent: str | None
) -> dict[str, Any]:
    assert agent is not None
    path = f"{_AGENTS}/{quote(agent, safe='')}/usage"
    query = _page_query(arguments)
    if not (arguments.all or arguments.since or arguments.until):
        return client.request("GET", path, query=query)
    query.setdefault("limit", str(_MAX_PAGE))
    items: list[dict[str, Any]] = []
    totals: dict[str, dict[str, list[int]]] = {
        name: {} for name in ("by_model", "by_level", "by_slot", "by_evaluation")
    }
    page: dict[str, Any] = {}
    pages = 0
    while pages < MAX_USAGE_PAGES:
        page = client.request("GET", path, query=query)
        pages += 1
        items += page.get("items", [])
        for name, summed in totals.items():
            for total in page.get(name, []):
                counted = summed.setdefault(total["key"], [0, 0])
                counted[0] += total["requests"]
                counted[1] += total["customer_cost_nano_usd"]
        cursor = page.get("next_cursor")
        if cursor is None:
            break
        query = {**query, "cursor": cursor}
    windowed = arguments.since is not None or arguments.until is not None
    if windowed:
        items = [item for item in items if _within(item, arguments.since, arguments.until)]
        # Evaluations' requests carry no times on these pages, so their totals stay over
        # every page read (`evaluations_windowed` says so).
        totals = {
            "by_model": _summed(items, lambda item: item.get("model_id")),
            "by_level": _summed(items, lambda item: item.get("model_level")),
            "by_slot": _summed(items, lambda item: item.get("model_slot")),
            "by_evaluation": totals["by_evaluation"],
        }
    result: dict[str, Any] = {
        "organization_id": page.get("organization_id"),
        "agent_id": page.get("agent_id"),
        "items": items,
        "complete": page.get("next_cursor") is None,
        "next_cursor": page.get("next_cursor"),
        "pages": pages,
        "requests": len(items),
        "customer_cost_nano_usd": sum(item["customer_cost_nano_usd"] for item in items),
        **{
            name: [
                {"key": key, "requests": requests, "customer_cost_nano_usd": cost}
                for key, (requests, cost) in sorted(summed.items())
            ]
            for name, summed in totals.items()
        },
    }
    if windowed:
        result["since"] = None if arguments.since is None else arguments.since.isoformat()
        result["until"] = None if arguments.until is None else arguments.until.isoformat()
        result["evaluations_windowed"] = False
    return result


def _within(item: Mapping[str, Any], since: datetime | None, until: datetime | None) -> bool:
    try:
        at = datetime.fromisoformat(item["requested_at"])
    except (KeyError, TypeError, ValueError):
        return False
    return (since is None or at >= since) and (until is None or at < until)


def _summed(
    items: Sequence[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], object]
) -> dict[str, list[int]]:
    summed: dict[str, list[int]] = {}
    for item in items:
        name = key(item)
        if isinstance(name, str):
            counted = summed.setdefault(name, [0, 0])
            counted[0] += 1
            counted[1] += item["customer_cost_nano_usd"]
    return summed


def _request(arguments: argparse.Namespace, client: CloudApiClient, _: str | None) -> Any:
    path = f"{_REQUESTS}/{quote(arguments.request_id, safe='')}"
    return client.request("GET", f"{path}/evidence" if arguments.evidence else path)


# Shared


def _page_query(arguments: argparse.Namespace, cursor: str = "cursor") -> dict[str, str]:
    query: dict[str, str] = {}
    if arguments.limit is not None:
        query["limit"] = str(arguments.limit)
    if arguments.cursor is not None:
        query[cursor] = arguments.cursor
    return query


def _models(models: Sequence[str]) -> list[str]:
    chosen = _distinct(models)
    if len(chosen) > _MAX_MODELS:
        raise _error("invalid_input", f"Give at most {_MAX_MODELS} models.")
    return chosen


def _distinct(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _period_amounts(values: Iterable[tuple[str, int]]) -> dict[str, int]:
    amounts: dict[str, int] = {}
    for period, amount in values:
        if period in amounts:
            raise _error("invalid_input", f"The {period} amount is given twice.")
        amounts[period] = amount
    return amounts


def _removed(removed: Sequence[str], set_here: Mapping[str, int]) -> list[str]:
    both = sorted(set(removed) & set(set_here))
    if both:
        raise _error("invalid_input", f"{', '.join(both)} is both set and removed; choose one.")
    return _distinct(removed)


_Action = Callable[[argparse.Namespace, CloudApiClient, str | None], Any]
_ACTIONS: dict[tuple[str, str], tuple[_Action, str]] = {
    ("keys", "create"): (_create_key, _KEYS_FLAG),
    ("keys", "list"): (_list_keys, _KEYS_FLAG),
    ("keys", "show"): (_show_key, _KEYS_FLAG),
    ("keys", "audit"): (_audit_key, _KEYS_FLAG),
    ("keys", "rotate"): (_rotate_key, _KEYS_FLAG),
    ("keys", "restrict"): (_restrict_key, _KEYS_FLAG),
    ("keys", "revoke"): (_revoke_key, _KEYS_FLAG),
    ("keys", "allowance"): (_allowance, _KEYS_FLAG),
    ("models", "catalog"): (_catalog, _CATALOG_FLAG),
    ("models", "show"): (_show_models, _POLICY_FLAG),
    ("models", "set"): (_set_models, _POLICY_FLAG),
    ("models", "status"): (_model_status, _USAGE_FLAG),
    ("limits", "show"): (_show_limits, _POLICY_FLAG),
    ("limits", "set"): (_set_limits, _POLICY_FLAG),
    ("limits", "alerts"): (_set_alerts, _POLICY_FLAG),
    ("usage", "agent"): (_agent_usage, _USAGE_FLAG),
    ("usage", "request"): (_request, _KEYS_FLAG),
}


# Text output


def render_keys(outcome: Mapping[str, Any]) -> str:
    result = outcome["result"]
    lines = _table(
        ("KEY", "NAME", "OWNER", "AGENT", "MODELS", "EXPIRES", "REVOKED"),
        [
            (
                _cell(item.get("key_id")),
                _cell(item.get("name")),
                _cell(item.get("owner_kind")),
                _cell(item.get("agent_id")),
                _cell(", ".join(item.get("allowed_models") or [])),
                _cell(item.get("expires_at")),
                "yes" if item.get("revoked") else "no",
            )
            for item in result.get("items") or []
        ],
    )
    return "\n".join([*lines, *_next_page(result)])


def render_models(outcome: Mapping[str, Any]) -> str:
    result = outcome["result"]
    lines = [
        f"Scope: {_cell(result['scope'])} {_cell(result['scope_id'])}",
        "Models it may use: " + _cell(", ".join(result["allowed_models"])),
        "Default model: "
        + _cell(result["default_model"])
        + (" (every Agent is locked to it)" if result["locked_to"] else ""),
    ]
    if not result["enrolled"]:
        lines.append("Not enrolled in the Gateway yet.")
    lines.append("")
    lines += _table(
        ("SCOPE", "POLICY", "MODELS", "DEFAULT"),
        [
            (
                _cell(row["principal"]["scope"]),
                _cell(policy["binding"]["policy_id"]),
                _cell(", ".join(policy["rules"]["allowed_models"])),
                _cell(policy["rules"].get("default_model")),
            )
            for row in result["configuration"]["rows"]
            if isinstance(row, dict)
            for policy in row.get("policies", [])
        ],
    )
    return "\n".join(lines)


def render_limits(outcome: Mapping[str, Any]) -> str:
    result = outcome["result"]
    lines = [f"Scope: {_cell(result['scope'])} {_cell(result['scope_id'])}", ""]
    lines += _table(
        ("SCOPE", "POLICY", "PERIOD", "CAP"),
        [
            (
                _cell(cap["scope"]),
                _cell(cap["policy_id"]),
                _cell(cap["period"]),
                nano_to_dollars(cap["ceiling_nano_usd"]),
            )
            for cap in result["caps"]
        ],
    )
    budgets = (result["alert_budgets"] or {}).get("rules", {}).get("budgets", [])
    lines.append("")
    if not budgets:
        lines.append("Alert budgets: none")
        return "\n".join(lines)
    lines += _table(
        ("PERIOD", "ALERT BUDGET", "ALERT AT"),
        [
            (
                _cell(item["period"]),
                nano_to_dollars(item["amount_nano_usd"]),
                ", ".join(f"{points / 100:g}%" for points in item["thresholds_basis_points"]),
            )
            for item in budgets
        ],
    )
    return "\n".join(lines)


def render_usage(outcome: Mapping[str, Any]) -> str:
    result = outcome["result"]
    items = result.get("items") or []
    lines = _table(
        ("REQUESTED", "REQUEST", "MODEL", "STATUS", "COST"),
        [
            (
                _cell(item.get("requested_at")),
                _cell(item.get("request_id")),
                _cell(item.get("model_id")),
                _cell(item.get("execution_status")),
                nano_to_dollars(item.get("customer_cost_nano_usd")),
            )
            for item in items
        ],
    )
    lines.append("")
    lines += _table(
        ("MODEL", "REQUESTS", "COST"),
        [
            (
                _cell(total["key"]),
                str(total["requests"]),
                nano_to_dollars(total["customer_cost_nano_usd"]),
            )
            for total in result.get("by_model") or []
        ],
    )
    cost = sum(item.get("customer_cost_nano_usd", 0) for item in items)
    cursor = result.get("next_cursor")
    # A page can be empty and still have more after it, so totals are final only at the end.
    label = "Total" if cursor is None else "Total so far"
    lines.append("")
    lines.append(f"{label}: {len(items)} settled requests, {nano_to_dollars(cost)}")
    evaluations = result.get("by_evaluation") or []
    if evaluations:
        spent = sum(total["customer_cost_nano_usd"] for total in evaluations)
        apart = (
            "Evaluations, apart, over every page read (not only the window)"
            if result.get("evaluations_windowed") is False
            else "Evaluations, apart"
        )
        lines.append(f"{apart}: {nano_to_dollars(spent)}")
    if cursor is not None and "pages" in result:
        window = " the window may hold more;" if "since" in result else ""
        lines.append(
            f"Stopped after {result['pages']} pages with more remaining:{window} go on with "
            "--cursor below."
        )
    return "\n".join([*lines, *_next_page(result)])


def _next_page(result: Mapping[str, Any]) -> list[str]:
    cursor = result.get("next_cursor")
    return [] if cursor is None else ["", f"More: --cursor {_cell(cursor)}"]


def _cell(value: object) -> str:
    return "-" if value is None or value == "" else terminal_safe(str(value))
