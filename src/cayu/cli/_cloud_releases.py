"""`cayu cloud applications show` and `cayu cloud deployment list`: what an Agent runs now, and
its Releases.

`applications show NAME` reads the Agent, its selected Release and its newest Releases, and
prints the selected Release (when it was selected, its source revision and the Cayu runtime
its image is built with) and any candidate or in-flight Release. `deployment list
--application NAME` lists the Agent's Releases, newest first.

Both read routes every Cloud has. Older Clouds omit `selected_at` and `cayu_runtime`; the table
says "not recorded" for those missing fields.

`release_application_id` finds the Agent of a Release from its id alone (`GET
/v1/deployments/{id}`), so `deployment status|logs|timeline|wait` take a `dep_...` id without
`--application`. A Cloud without that route still finds the Release an Agent currently selects;
for any other Release it asks for `--application`.

These commands print a table by default; `--json` prints the JSON envelope the other `cayu
cloud` commands print, with `schema_version` in its result.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, cast

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_settings import _cell, _table

SCHEMA_VERSION = 1
DEFAULT_LIMIT = 20
MAX_LIMIT = 1000
# Cloud's page size bound for an Agent's Releases.
_PAGE_LIMIT = 100
# Releases `applications show` reads for a candidate or in-flight one.
_RECENT = 20
_IN_FLIGHT = frozenset(
    {
        "accepted",
        "image_built",
        "image_scanned",
        "policy_compiled",
        "sandbox_template_ready",
        "source_resolved",
    }
)


def release_limit(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if not 1 <= number <= MAX_LIMIT:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_LIMIT}")
    return number


def add_output_argument(parser: argparse.ArgumentParser, render: Callable[..., str]) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the JSON envelope the other `cayu cloud` commands print.",
    )
    parser.set_defaults(_cloud_render=render, _cloud_render_failure=render_failure)


def show_application(client: CloudApiClient, application_id: str) -> dict[str, Any]:
    """The Agent, its selected Release and its candidate or in-flight Releases."""

    path = f"/v1/applications/{application_id}"
    application = client.request("GET", path)
    selected_id = application.get("current_deployment_id")
    selected = (
        client.request("GET", f"{path}/deployments/{selected_id}")
        if isinstance(selected_id, str) and selected_id
        else None
    )
    recent = _page(client, application_id, limit=_RECENT, cursor=None)[0]
    pending = [
        release
        for release in recent
        if release.get("id") != selected_id and _pending_kind(release) is not None
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "application": {
            key: application.get(key)
            for key in ("id", "name", "lifecycle", "revision", "archived_at", "home")
        },
        "selected_release": None if selected is None else release_summary(selected),
        "pending_releases": [
            {**release_summary(release), "pending": _pending_kind(release)} for release in pending
        ],
    }


def list_releases(client: CloudApiClient, application_id: str, *, limit: int) -> dict[str, Any]:
    """The Agent's newest ``limit`` Releases, newest first."""

    items: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        page, cursor = _page(
            client, application_id, limit=min(_PAGE_LIMIT, limit - len(items)), cursor=cursor
        )
        items.extend(page)
        if cursor is None or len(items) >= limit:
            break
    return {
        "schema_version": SCHEMA_VERSION,
        "application_id": application_id,
        "items": [release_summary(release) for release in items[:limit]],
        "next_cursor": cursor,
    }


def release_application_id(client: CloudApiClient, deployment_id: str) -> str:
    """The Agent of the Release ``deployment_id``, found from its id alone."""

    try:
        release = client.request("GET", f"/v1/deployments/{deployment_id}")
    except CloudApiError as exc:
        if exc.status_code != 404 or exc.code is not None:
            # `release_not_found` is Cloud saying there is no such Release.
            raise
        # A Cloud without this route: the Releases Agents select are listed with them.
        return _selecting_application_id(client, deployment_id)
    application_id = release.get("application_id")
    if not isinstance(application_id, str) or not application_id:
        raise CloudApiError("api_response_invalid", "The Release names no Agent.")
    return application_id


def _selecting_application_id(client: CloudApiClient, deployment_id: str) -> str:
    applications = client.request("GET", "/v1/applications", query={"lifecycle": "all"}).get(
        "items"
    )
    if not isinstance(applications, list):
        raise CloudApiError("api_response_invalid", "Application list is invalid.")
    for application in applications:
        if isinstance(application, dict) and application.get("current_deployment_id") == (
            deployment_id
        ):
            return str(application["id"])
    raise CloudApiError(
        "release_lookup_unsupported",
        f"This Cayu Cloud can't find Release {deployment_id} from its id alone yet; pass "
        "--application with the Agent it belongs to.",
    )


def release_summary(release: Mapping[str, Any]) -> dict[str, Any]:
    """What these commands show of a Release, the same in `show` and `list`."""

    manifest = release.get("manifest")
    source = manifest.get("source") if isinstance(manifest, dict) else None
    provenance = release.get("provenance")
    revision = (source or {}).get("revision") if isinstance(source, dict) else None
    if not isinstance(revision, str) and isinstance(provenance, dict):
        revision = provenance.get("source_revision")
    runtime = release.get("cayu_runtime")
    return {
        "id": release.get("id"),
        "version": release.get("version"),
        "status": release.get("status"),
        "candidate": isinstance(release.get("candidate"), dict),
        "stage_only": release.get("stage_only") is True,
        "created_at": release.get("created_at"),
        "updated_at": release.get("updated_at"),
        # Absent on a Cloud from before it was answered: None, like a Release never selected.
        "selected_at": release.get("selected_at"),
        "source_repository": (source or {}).get("repository") if isinstance(source, dict) else None,
        "source_revision": revision if isinstance(revision, str) else None,
        "manifest_digest": release.get("manifest_digest"),
        "runtime_artifact_id": release.get("runtime_artifact_id"),
        "cayu_runtime": runtime if isinstance(runtime, dict) else None,
        "publication_error": release.get("publication_error"),
        "session_preflight": release.get("session_preflight"),
    }


def _pending_kind(release: Mapping[str, Any]) -> str | None:
    status = release.get("status")
    if status in _IN_FLIGHT:
        return "in_flight"
    if status == "smoke_tested":
        preflight = release.get("session_preflight")
        if isinstance(preflight, dict) and preflight.get("state") == "superseded":
            # Another release was selected while this one waited: Cloud never publishes it.
            return None
        # Stage-only Releases wait for someone to promote them; others are about to be selected.
        staged = isinstance(release.get("candidate"), dict) or release.get("stage_only") is True
        return "candidate" if staged else "in_flight"
    return None


def _page(
    client: CloudApiClient, application_id: str, *, limit: int, cursor: str | None
) -> tuple[list[dict[str, Any]], str | None]:
    query = {"limit": str(limit)}
    if cursor is not None:
        query["cursor"] = cursor
    answer = client.request("GET", f"/v1/applications/{application_id}/deployments", query=query)
    items = answer.get("items")
    if not isinstance(items, list):
        raise CloudApiError("api_response_invalid", "The Release list is invalid.")
    next_cursor = answer.get("next_cursor")
    return (
        [item for item in items if isinstance(item, dict)],
        next_cursor if isinstance(next_cursor, str) and next_cursor else None,
    )


def render_application(result: Mapping[str, Any]) -> str:
    shown = result["result"]
    application = shown["application"]
    lines = [
        f"Agent: {_cell(application.get('name'))} ({_cell(application.get('id'))})",
        f"Lifecycle: {_cell(application.get('lifecycle'))}",
    ]
    selected = shown["selected_release"]
    if selected is None:
        lines.append("Selected release: none")
    else:
        lines += [
            f"Selected release: {_cell(selected['id'])} (version {_cell(selected['version'])}, "
            f"{_cell(selected['status'])})",
            f"  Selected at: {_when(selected['selected_at'], missing='not recorded')}",
            f"  Created at: {_when(selected['created_at'])}",
            f"  Source: {_source(selected)}",
            f"  Cayu runtime: {_runtime(selected['cayu_runtime'])}",
        ]
        provenance = _provenance(selected["cayu_runtime"])
        if provenance is not None:
            lines.append(f"  Cayu installed from: {provenance}")
        if selected.get("publication_error"):
            error = selected["publication_error"]
            message = error.get("message") if isinstance(error, dict) else None
            lines.append(f"  Not published: {_cell(message)}")
    pending = shown["pending_releases"]
    if not pending:
        lines.append("Candidate or in-flight releases: none")
    else:
        lines += ["", "Candidate or in-flight releases:"]
        rows = [
            (
                _cell(item["id"]),
                _cell(item["pending"].replace("_", "-")),
                _cell(item["version"]),
                _cell(item["status"]),
                _when(item["created_at"]),
                _short(item["source_revision"]),
                _runtime(item["cayu_runtime"]),
            )
            for item in pending
        ]
        lines += _table(("RELEASE", "KIND", "VERSION", "STATUS", "CREATED", "SOURCE", "CAYU"), rows)
    return "\n".join(lines)


def render_releases(result: Mapping[str, Any]) -> str:
    listed = result["result"]
    items = listed["items"]
    if not items:
        return f"No releases for {_cell(listed['application_id'])}."
    rows = [
        (
            _cell(item["id"]),
            _cell(item["version"]),
            _cell(item["status"]),
            _when(item["created_at"]),
            _when(item["selected_at"]),
            _short(item["source_revision"]),
            _runtime(item["cayu_runtime"]),
        )
        for item in items
    ]
    lines = _table(("RELEASE", "VERSION", "STATUS", "CREATED", "SELECTED", "SOURCE", "CAYU"), rows)
    if listed.get("next_cursor") is not None:
        lines.append(f"Showing the newest {len(items)}; pass a larger --limit for more.")
    return "\n".join(lines)


def render_failure(error: Mapping[str, Any]) -> str:
    return f"Error: {error['message']}"


def _source(release: Mapping[str, Any]) -> str:
    repository, revision = release.get("source_repository"), release.get("source_revision")
    if repository is None:
        return _cell(revision)
    return f"{_cell(repository)} @ {_cell(revision)}"


def _mapping(value: object) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", value) if isinstance(value, dict) else {}


def _runtime(runtime: object) -> str:
    version = _mapping(runtime).get("version")
    if not isinstance(version, str):
        return "not recorded"
    return f"cayu {_cell(version, limit=64)}"


def _provenance(runtime: object) -> str | None:
    """Where the image's Cayu came from, as the lock it was built from says."""

    source = _mapping(_mapping(runtime).get("source"))
    kind = source.get("kind")
    if not isinstance(kind, str):
        return None
    text = _cell(kind)
    where = source.get("url") or source.get("path")
    if isinstance(where, str):
        text += f" {_cell(where, limit=200)}"
    commit = source.get("commit")
    if isinstance(commit, str):
        text += f" @ {_short(commit)}"
    reference = next(
        (source[name] for name in ("tag", "branch", "rev") if isinstance(source.get(name), str)),
        None,
    )
    if reference is not None:
        text += f" ({_cell(reference, limit=64)})"
    return text


def _short(revision: object) -> str:
    """A commit as its first 12 characters, the way git abbreviates it."""

    if isinstance(revision, str) and len(revision) >= 40 and revision.isalnum():
        return revision[:12]
    return _cell(revision, limit=40)


def _when(value: object, *, missing: str = "-") -> str:
    if not isinstance(value, str):
        return missing
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return _cell(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).strftime("%Y-%m-%d %H:%MZ")
