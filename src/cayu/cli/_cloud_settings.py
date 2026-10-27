"""`cayu cloud settings`: read and change an Agent's managed settings in Cayu Cloud.

The commands call Cloud's settings API (`GET`, `PUT` and `DELETE
/v1/applications/{id}/settings`, contract `docs/managed-settings.md` section 5), the
same one the portal uses, with the `cayu cloud login` or a selected context's API key.
Each setting is in one of two namespaces: `platform` (how Cloud runs the Agent, such as
`web.sleep`) or `agent` (what the Agent's own parts declare). `set` and `unset` find a
key's namespace in the current document; a key that both namespaces have needs a
`platform:` or `agent:` prefix.

`set` sends all its assignments in one all-or-nothing write, under the revision it read
and an operation id. Cloud answers a repeated operation id with its first answer, so a
write whose answer was lost can be repeated with `--operation-id`.

Unlike the other `cayu cloud` commands, these print a table by default; `--json` prints
the same JSON envelope the others print.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import secrets
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_diagnostics import safe_text

NAMESPACES = ("platform", "agent")
# Platform keys can be a single word (`sign_in`); Agent keys are dotted and namespaced by
# the part that declares them. Cloud validates the exact rules.
_KEY = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z0-9_-]+)*")
_KEY_MAX_LENGTH = 128
_OPERATION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
# A refused setting's reason, and the allowed values Cloud offers with it.
_REASON_LIMIT = 500
_ALLOWED_LIMIT = 2048
_ERRORS_LIMIT = 100
_CELL_LIMIT = 60


class CloudSettingsError(CloudApiError):
    """A refused settings write, with what Cloud said about it and the operation id."""

    def __init__(self, error: CloudApiError, message: str, details: dict[str, object]) -> None:
        super().__init__(
            error.category,
            message,
            status_code=error.status_code,
            detail=error.detail,
            code=error.code,
            rejected_fields=error.rejected_fields,
        )
        self.details = details


def add_settings_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    settings = commands.add_parser(
        "settings",
        help="Read and change an Agent's managed settings.",
        description=(
            "Read and change an Agent's managed settings without a new deploy. The Release "
            "gives each setting its value; a value set here wins until it is unset."
        ),
    )
    actions = settings.add_subparsers(dest="settings_command", required=True)
    listing = actions.add_parser(
        "list",
        description=(
            "Show each setting's effective value, where it comes from (the Release or "
            "Cloud), and whether it has applied."
        ),
    )
    setter = actions.add_parser(
        "set",
        description=(
            "Set one or more settings in one all-or-nothing change. Prefix a key with "
            "platform: or agent: when both namespaces have it."
        ),
    )
    setter.add_argument(
        "assignments",
        nargs="+",
        metavar="KEY=VALUE",
        help=(
            "VALUE is read as JSON when it parses (60, true, null, {...}) and as a string "
            "otherwise; write '\"60\"' to send the string 60."
        ),
    )
    setter.add_argument(
        "--operation-id",
        type=_operation_id,
        help=(
            "Repeat a change whose answer was lost: Cloud answers a repeated id with its "
            "first answer. Defaults to a new id, which the output names."
        ),
    )
    unsetter = actions.add_parser(
        "unset",
        description="Clear a setting's Cloud value, so the Release's value applies again.",
    )
    unsetter.add_argument("key", metavar="KEY")
    for parser in (listing, setter, unsetter):
        parser.add_argument("--application", required=True)
        parser.add_argument(
            "--json",
            action="store_true",
            help="Print the JSON envelope the other `cayu cloud` commands print.",
        )
        parser.set_defaults(
            _cloud_render=render_settings, _cloud_render_failure=render_settings_failure
        )


def settings_command(
    arguments: argparse.Namespace, *, client: CloudApiClient, application_id: str
) -> dict[str, Any]:
    path = f"/v1/applications/{quote(application_id, safe='')}/settings"
    action = arguments.settings_command
    if action == "list":
        return {"operation": "settings.list", "result": _read(client, path)}
    if action == "unset":
        reference = _reference(arguments.key)
        namespace, key = _resolve(_read(client, path), reference)
        try:
            result = client.request("DELETE", f"{path}/{namespace}/{quote(key, safe='')}")
        except CloudApiError as exc:
            raise _settings_error(exc) from None
        return {
            "operation": "settings.unset",
            "result": _document(result),
            "unset": [{"key": key, "namespace": namespace}],
        }
    # Check every assignment before reading the settings, so a typo costs no settings read.
    assignments = [_assignment(value) for value in arguments.assignments]
    operation_id = arguments.operation_id or f"op_{secrets.token_hex(16)}"
    document = _read(client, path)
    changes: dict[str, dict[str, object]] = {namespace: {} for namespace in NAMESPACES}
    for reference, value in assignments:
        namespace, key = _resolve(document, reference)
        if key in changes[namespace]:
            raise CloudApiError(
                "setting_repeated", f"{namespace}:{key} is assigned more than once."
            )
        changes[namespace][key] = value
    payload: dict[str, object] = {
        "clear": {namespace: [] for namespace in NAMESPACES},
        "operation_id": operation_id,
        "revision": document["revision"],
        "set": changes,
    }
    try:
        # An answer that isn't the document still names the operation id: the change may
        # have applied, and repeating it under the same id is the safe way to find out.
        result = _document(client.request("PUT", path, payload=payload))
    except CloudApiError as exc:
        raise _settings_error(exc, operation_id=operation_id) from None
    return {
        "operation": "settings.set",
        "operation_id": operation_id,
        "result": result,
        "set": [
            {"key": key, "namespace": namespace}
            for namespace in NAMESPACES
            for key in changes[namespace]
        ],
    }


def _operation_id(value: str) -> str:
    if _OPERATION_ID.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be 1 to 128 letters, digits, hyphens or underscores")
    return value


def _reference(value: str) -> tuple[str | None, str]:
    """A key as written, split into its namespace prefix (if any) and the key."""

    namespace: str | None = None
    key = value
    prefix, separator, rest = value.partition(":")
    if separator:
        if prefix not in NAMESPACES:
            raise CloudApiError(
                "setting_key_invalid",
                f"{value!r} has an unknown prefix; use platform: or agent:.",
            )
        namespace, key = prefix, rest
    if not _is_key(key):
        raise CloudApiError(
            "setting_key_invalid",
            f"{value!r} isn't a setting key; keys look like web.sleep or agent.scout.model.",
        )
    return namespace, key


def _is_key(value: object) -> bool:
    return isinstance(value, str) and len(value) <= _KEY_MAX_LENGTH and bool(_KEY.fullmatch(value))


def _assignment(value: str) -> tuple[tuple[str | None, str], object]:
    reference, separator, raw = value.partition("=")
    if not separator:
        raise CloudApiError(
            "setting_assignment_invalid",
            f"{value!r} isn't KEY=VALUE; `settings unset KEY` clears a setting.",
        )
    return _reference(reference), _value(raw)


def _value(raw: str) -> object:
    """JSON when the text parses as JSON, the text itself otherwise."""

    def refuse(constant: str) -> object:
        raise ValueError(constant)

    def finite(text: str) -> float:
        number = float(text)
        if not math.isfinite(number):
            raise ValueError(text)
        return number

    try:
        # NaN and Infinity aren't JSON, so they stay strings, as does a number too large
        # for a float (1e999).
        return json.loads(raw, parse_constant=refuse, parse_float=finite)
    except ValueError:
        return raw


def _read(client: CloudApiClient, path: str) -> dict[str, Any]:
    """The Agent's settings document, naming a Cloud that has no settings API yet."""

    try:
        return _document(client.request("GET", path))
    except CloudApiError as exc:
        # The Agent was just resolved, so a 404 without one of Cloud's codes, or a 405,
        # means the route itself is missing.
        if exc.status_code == 405 or (exc.status_code == 404 and exc.code is None):
            raise CloudApiError(
                "settings_unsupported", "This Cayu Cloud can't manage Agent settings yet."
            ) from None
        raise


def _document(value: dict[str, Any]) -> dict[str, Any]:
    """The settings document, checked for what the commands read from it."""

    revision = value.get("revision")
    settings = value.get("settings")
    if type(revision) is not int or not isinstance(settings, list):
        raise CloudApiError("api_response_invalid", "The settings response is invalid.")
    return value


def _known(document: Mapping[str, Any]) -> dict[str, set[str]]:
    """Each namespace's keys: its settings, and stored values that conflict with the Release.

    A conflicting value has no descriptor any more, but `unset` must still reach it.
    """

    known: dict[str, set[str]] = {namespace: set() for namespace in NAMESPACES}
    for entry in document.get("settings", []):
        if not isinstance(entry, dict):
            continue
        descriptor = entry.get("descriptor")
        key = descriptor.get("key") if isinstance(descriptor, dict) else None
        if entry.get("namespace") in known and isinstance(key, str):
            known[entry["namespace"]].add(key)
    conflicts = document.get("conflicts")
    for conflict in conflicts if isinstance(conflicts, list) else []:
        if (
            isinstance(conflict, dict)
            and conflict.get("namespace") in known
            and isinstance(conflict.get("key"), str)
        ):
            known[conflict["namespace"]].add(conflict["key"])
    return known


def _resolve(document: Mapping[str, Any], reference: tuple[str | None, str]) -> tuple[str, str]:
    namespace, key = reference
    if namespace is not None:
        # An explicit namespace is sent as written, and Cloud judges the key. That also
        # reaches an Agent setting while the Release hasn't published its settings yet.
        return namespace, key
    known = _known(document)
    matches = [candidate for candidate in NAMESPACES if key in known[candidate]]
    if len(matches) == 1:
        return matches[0], key
    if matches:
        raise CloudApiError(
            "setting_ambiguous",
            f"{key} is both a platform and an agent setting; write platform:{key} or agent:{key}.",
        )
    message = f"The Agent has no setting named {key}; `cayu cloud settings list` shows them."
    if document.get("agent_settings") == "waiting_for_release":
        message += (
            " Its own settings appear once its Release starts; write agent:"
            f"{key} to change one before that."
        )
    raise CloudApiError("setting_unknown", message)


def _settings_error(error: CloudApiError, *, operation_id: str | None = None) -> CloudApiError:
    """Cloud's refusal of a settings write, saying what Cloud said about each setting."""

    body = error.body or {}
    details: dict[str, object] = {}
    if operation_id is not None:
        details["operation_id"] = operation_id
    message = str(error)
    if error.status_code == 409 and error.code == "settings_revision_conflict":
        message = (
            "The Agent's settings changed while this command ran, so nothing was changed. "
            "Check them with `cayu cloud settings list` and run the command again."
        )
        if type(body.get("revision")) is int:
            details["revision"] = body["revision"]
    elif error.status_code == 422 and error.code == "invalid_setting":
        errors = _setting_errors(body.get("errors"))
        details["errors"] = errors
        count = "a setting" if len(errors) == 1 else f"{len(errors) or 'some'} settings"
        message = f"Cayu Cloud refused {count}, so nothing was changed."
    elif error.status_code == 403 and error.code == "organization_admin_required":
        key = body.get("key")
        if _is_key(key):
            details["key"] = key
            message = (
                f"Only an organization admin can change {key}, so nothing was changed. "
                "Sign in again with `cayu cloud login` if you are one."
            )
    if not details:
        return error
    return CloudSettingsError(error, message, details)


def _setting_errors(value: object) -> list[dict[str, object]]:
    """Cloud's reasons for refusing each setting, keeping only well-formed, bounded fields."""

    errors: list[dict[str, object]] = []
    for item in value[:_ERRORS_LIMIT] if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not _is_key(key):
            continue
        error: dict[str, object] = {"key": key}
        if item.get("namespace") in NAMESPACES:
            error["namespace"] = item["namespace"]
        reason = safe_text(item.get("reason"), _REASON_LIMIT)
        error["reason"] = reason if reason is not None else "refused"
        if "allowed" in item:
            try:
                encoded = json.dumps(item["allowed"], allow_nan=False, separators=(",", ":"))
            except (TypeError, ValueError):
                encoded = None
            if encoded is not None and len(encoded.encode()) <= _ALLOWED_LIMIT:
                error["allowed"] = item["allowed"]
        errors.append(error)
    return errors


def render_settings(output: Mapping[str, Any]) -> str:
    """The table `list`, `set` and `unset` print without `--json`."""

    document = output["result"]
    lines: list[str] = []
    changed = output.get("set") or output.get("unset") or []
    if changed:
        names = ", ".join(f"{item['namespace']}:{item['key']}" for item in changed)
        verb = "Set" if "set" in output else "Unset"
        lines.append(f"{verb} {names}.")
        if "operation_id" in output:
            lines.append(f"Operation {output['operation_id']}.")
        lines.append("")
    heading = [f"Agent {_cell(document.get('application_id'))}"]
    if document.get("release_id") is not None:
        heading.append(f"Release {_cell(document.get('release_id'))}")
    heading.append(f"revision {document['revision']}")
    lines.append(", ".join(heading))
    if document.get("agent_settings") == "waiting_for_release":
        lines.append("Agent settings: waiting for the Release to start.")
    rows: list[tuple[str, ...]] = []
    notes: list[str] = []
    for entry in document["settings"]:
        if not isinstance(entry, dict):
            continue
        descriptor = entry.get("descriptor")
        key = _cell(descriptor.get("key") if isinstance(descriptor, dict) else None)
        status = _cell(entry.get("status"))
        reports = entry.get("reports")
        if isinstance(reports, dict) and type(reports.get("processes")) is int:
            parts = [f"{_count(reports.get('adopted'))} of {reports['processes']} adopted"]
            refused = _count(reports.get("refused"))
            if refused:
                parts.append(f"{refused} refused")
            status += f" ({', '.join(parts)})"
            reasons = reports.get("refused_reasons")
            for reason in reasons[:3] if isinstance(reasons, list) else []:
                notes.append(f"{key} refused: {_cell(reason, limit=200)}")
        rows.append(
            (
                key,
                _cell(entry.get("namespace")),
                _value_cell(entry.get("effective")),
                _cell(entry.get("source")),
                status,
            )
        )
    lines.append("")
    if rows:
        lines.extend(_table(("SETTING", "NAMESPACE", "VALUE", "SOURCE", "STATUS"), rows))
    else:
        lines.append("No settings.")
    if notes:
        lines.extend(["", *notes])
    conflicts = document.get("conflicts")
    conflicts = [
        item
        for item in (conflicts if isinstance(conflicts, list) else [])
        if isinstance(item, dict)
    ]
    if conflicts:
        lines.extend(["", "Not applied, conflicting with the Release:"])
        lines.extend(
            f"  {_cell(item.get('namespace'))}:{_cell(item.get('key'))}  "
            f"{_cell(item.get('reason'))}"
            for item in conflicts
        )
    return "\n".join(lines)


def render_settings_failure(error: Mapping[str, Any]) -> str:
    """What a failed `list`, `set` or `unset` prints to standard error without `--json`."""

    lines = [f"Error: {error['message']}"]
    for item in error.get("errors", []):
        prefix = f"{item['namespace']}:" if "namespace" in item else ""
        line = f"  {prefix}{item['key']}: {_cell(item['reason'], limit=_REASON_LIMIT)}"
        if "allowed" in item:
            line += f" (allowed: {_value_cell(item['allowed'], limit=200)})"
        lines.append(line)
    if "operation_id" in error:
        lines.append(f"Operation {error['operation_id']}.")
    return "\n".join(lines)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(row[column]) for row in (header, *rows)) for column in range(len(header))]
    return [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in (header, *rows)
    ]


def _count(value: object) -> int:
    return value if type(value) is int else 0


def _value_cell(value: object, *, limit: int = _CELL_LIMIT) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return _cell(value, limit=limit)
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return "?"
    return _cell(text, limit=limit)


def _cell(value: object, *, limit: int = _CELL_LIMIT) -> str:
    """Text for the terminal: control characters replaced, long values shortened."""

    if value is None:
        return "-"
    text = "".join(
        "?" if unicodedata.category(character) in {"Cc", "Cf"} else character
        for character in str(value)
    )
    return text if len(text) <= limit else text[: limit - 3] + "..."
