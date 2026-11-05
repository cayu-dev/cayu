"""`cayu cloud admin system-agents`, and what `service status` says about a system Agent.

System Agents are the install's own services, which Cloud calls directly (Compound, Compass,
Arena's case service). The install names them as ``<organization id>/<agent id>`` entries in
``CAYU_SYSTEM_AGENTS``; the deprecated ``CAYU_AGENT_ALWAYS_ON`` still names some by a bare Agent
id. ECS installs set both in the API's task definition and EKS installs in the chart's
``agents.systemAgents`` and ``agents.alwaysOn`` values; either way Cloud's API reports what it
read (``GET /v1/admin/system-agents``), so this command needs no AWS or Kubernetes access.

Each entry comes with the Agent it matches today and whether that Agent's service was recorded
as a system Agent when it was last published or restarted. An Agent named after its last
publish or restart is recorded as not one until its next one. The entries name other
Organizations' Agents, so Cloud answers only a signed-in platform administrator.

It prints a table by default; ``--json`` prints the JSON envelope the other ``cayu cloud``
commands print.

``service status`` reports ``system`` from the install's current setting and
``recorded_system`` from the service's last publish or restart; `system_change_warning` says
when they differ. A Cloud that doesn't report ``recorded_system`` reports the recorded value as
``system``, and gets no warning.
"""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_releases import add_output_argument
from cayu.cli._cloud_runtime_record import add_runtime_record_parser, runtime_record_command
from cayu.cli._cloud_settings import _cell, _table

_PATH = "/v1/admin/system-agents"


def add_admin_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    application: Callable[[str], str],
) -> None:
    """``application`` checks an Agent slug argument."""

    admin = commands.add_parser(
        "admin",
        help="Read the install's own configuration and any Agent's runtime record (Cayu "
        "platform administrators).",
        description=(
            "Read how this Cayu Cloud install is configured, and read or correct any "
            "Organization's Agent's runtime record. Needs a signed-in Cayu platform "
            "administrator (`cayu cloud login`); an Organization admin or an API key can't."
        ),
    )
    actions = admin.add_subparsers(dest="admin_command", required=True)
    system_agents = actions.add_parser(
        "system-agents",
        help="List the install's system Agents.",
        description=(
            "List the install's system Agents as Cloud's API read them: each "
            "<organization id>/<agent id> entry of CAYU_SYSTEM_AGENTS (chart "
            "agents.systemAgents), then any bare Agent id the deprecated CAYU_AGENT_ALWAYS_ON "
            "(chart agents.alwaysOn) still names, with the Agent each matches and whether its "
            "service was recorded as a system Agent at its last publish or restart."
        ),
    )
    add_output_argument(system_agents, render_system_agents)
    add_runtime_record_parser(actions, application=application)


def admin_command(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    recovery_arguments: Sequence[str] = (),
) -> dict[str, Any]:
    """``recovery_arguments`` are the command's own ``--context`` and ``--api-key-file``."""

    if arguments.admin_command == "system-agents":
        return {"operation": "admin.system-agents", "result": _system_agents(client)}
    if arguments.admin_command == "runtime-record":
        return runtime_record_command(
            arguments, client=client, recovery_arguments=recovery_arguments
        )
    raise ValueError(f"Unknown admin command {arguments.admin_command!r}.")


def _system_agents(client: CloudApiClient) -> dict[str, Any]:
    try:
        answer = client.request("GET", _PATH)
    except CloudApiError as exc:
        # The route takes no parameters, so a 404 without one of Cloud's codes, or a 405,
        # means the route itself is missing.
        if exc.status_code == 405 or (exc.status_code == 404 and exc.code is None):
            raise CloudApiError(
                "system_agents_unsupported",
                "This Cayu Cloud doesn't list its system Agents yet.",
                status_code=exc.status_code,
            ) from None
        if exc.status_code == 403:
            raise CloudApiError(
                "platform_admin_required",
                "Listing the install's system Agents needs a signed-in Cayu platform "
                "administrator (`cayu cloud login`).",
                status_code=403,
            ) from None
        raise
    items = answer.get("items")
    if not isinstance(items, list) or not all(isinstance(item, Mapping) for item in items):
        raise CloudApiError("api_response_invalid", "The system Agent list is invalid.")
    return {"items": items}


def restart_pending(item: Mapping[str, Any]) -> bool:
    """Whether the entry's Agent runs a service recorded before the entry named it."""

    return _mapping(item.get("agent")).get("recorded_system") is False


def render_system_agents(result: Mapping[str, Any]) -> str:
    items: Sequence[Mapping[str, Any]] = result["result"]["items"]
    if not items:
        return "This install names no system Agents."
    rows = [
        (
            _cell(item.get("entry")),
            _cell(item.get("setting"))
            + (" (deprecated)" if item.get("deprecated") is True else ""),
            *_agent_cells(item.get("agent")),
        )
        for item in items
    ]
    lines = _table(("ENTRY", "SETTING", "ORGANIZATION", "LIFECYCLE", "SERVICE"), rows)
    pending = [_cell(item.get("entry")) for item in items if restart_pending(item)]
    if pending:
        lines += [
            "",
            f"Recorded as not a system Agent: {', '.join(pending)}. Each such service was "
            "last published or restarted before its entry named it, and becomes one at its "
            "next publish or restart.",
        ]
    if any(item.get("agent") is None for item in items):
        lines += ["", "No Agent matches an entry marked 'no such Agent': check its ids."]
    return "\n".join(lines)


def _agent_cells(value: object) -> tuple[str, str, str]:
    if not isinstance(value, Mapping):
        return ("-", "no such Agent", "-")
    agent = cast("Mapping[str, Any]", value)
    recorded = agent.get("recorded_system")
    if recorded is True:
        service = "system"
    elif recorded is False:
        service = "not system until restarted"
    else:
        service = "not recorded"
    return (_cell(agent.get("organization_id")), _cell(agent.get("lifecycle")), service)


def _mapping(value: object) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", value) if isinstance(value, Mapping) else {}


def system_change_warning(
    service: Mapping[str, Any],
    *,
    application: str,
    recovery_arguments: Sequence[str] = (),
) -> str | None:
    """A warning when the Agent's service was recorded under an older system Agent setting.

    The restart command it names keeps ``recovery_arguments`` (the status command's own
    ``--context`` and ``--api-key-file``), so it restarts the same Cloud's Agent.
    """

    current, recorded = service.get("system"), service.get("recorded_system")
    if type(current) is not bool or type(recorded) is not bool or current == recorded:
        return None
    command = shlex.join(
        ["cayu", "cloud", *recovery_arguments, "service", "restart", "--application", application]
    )
    restart = f"`{command}`"
    if current:
        return (
            "The install now names this Agent one of its system Agents, but its service was "
            "last published or restarted before that and is recorded as not one "
            "(`recorded_system: false`). The install's default idle timeout may still apply "
            f"to its web process until its next publish or restart ({restart})."
        )
    return (
        "The install no longer names this Agent one of its system Agents, but its service was "
        "last published or restarted as one (`recorded_system: true`) and keeps that until its "
        f"next publish or restart ({restart})."
    )
