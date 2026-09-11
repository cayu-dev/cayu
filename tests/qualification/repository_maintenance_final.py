"""Operator-only recorded delivery projection; no provider calls or effect authority."""

import os
import re
from urllib.parse import urlencode, urlsplit

from domain.maintenance_acceptance import (  # ty: ignore[unresolved-import]
    MaintenanceAcceptance,
    MaintenanceAcceptanceRejected,
)
from integrations.maintenance_github_host import (  # ty: ignore[unresolved-import]
    configured_github_repository,
)

from tests.qualification.repository_maintenance_cost import inspect_cost_evidence
from tests.qualification.repository_maintenance_github_intake import load_verified_github_result
from tests.qualification.repository_maintenance_identity import copy_identity
from tests.qualification.repository_maintenance_intake import MaintenanceTaskConflict
from tests.qualification.repository_maintenance_results import load_verified_coding_result


def _digest(value):
    if type(value) is not str:
        raise MaintenanceAcceptanceRejected()
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise MaintenanceAcceptanceRejected()
    return "sha256:" + digest


async def _acceptance(application, task, product):
    accepted = await application.verify(task, product)
    if (
        type(accepted) is not MaintenanceAcceptance
        or accepted.result_reference_id != product.result_reference.reference_id
        or accepted.result_digest != product.result_reference.digest
        or accepted.request_fingerprint != product.candidate.request_fingerprint
        or accepted.final_revision != product.candidate.final_revision
    ):
        raise MaintenanceAcceptanceRejected()
    return {
        field: _digest(getattr(accepted, field))
        for field in (
            "result_digest",
            "request_fingerprint",
            "final_revision",
            "corpus_fingerprint",
            "probe_output_digest",
        )
    }


def _pr_url(result):
    configured = configured_github_repository()
    origin = os.environ.get("CAYU_MAINTENANCE_GITHUB_WEB_ORIGIN")
    if type(origin) is not str or len(origin) > 2048 or not origin.isascii():
        raise ValueError("Invalid maintenance GitHub web origin.")
    parsed = urlsplit(origin)
    host, separator, port = parsed.netloc.partition(":")
    if (
        parsed.scheme != "https"
        or origin != "https://" + parsed.netloc
        or len(host) > 253
        or any(
            re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) is None
            for label in host.split(".")
        )
        or (
            separator
            and (not port.isdecimal() or not 1 <= int(port) <= 65535 or str(int(port)) != port)
        )
    ):
        raise ValueError("Invalid maintenance GitHub web origin.")
    pr = result.pull_request
    if (
        pr is None
        or type(pr.number) is not int
        or pr.number < 1
        or (result.repository_id, result.installation_id, result.account_id)
        != (configured.repository_id, configured.installation_id, configured.account_id)
    ):
        raise MaintenanceTaskConflict()
    expected = f"{origin}/{configured.owner}/{configured.name}/pull/{pr.number}"
    if type(pr.url) is not str or pr.url != expected:
        raise MaintenanceTaskConflict()
    return expected


def _link(identity, suffix):
    return f"/operator/runs/{identity.public_id}/{suffix}?" + urlencode(
        {"tenant": identity.intent.tenant}
    )


async def inspect_acceptance_evidence(application, reservations, expected):
    """Host must authorize the reservation; raw artifact contents are not exported."""
    identity = copy_identity(expected)
    task, product = await load_verified_coding_result(application, reservations, identity)
    return {"id": identity.public_id, **await _acceptance(application, task, product)}


async def inspect_final_result(application, reservations, expected):
    """Return recorded verification, not current remote state or permission to merge."""
    identity = copy_identity(expected)
    task, product, remote, github = await load_verified_github_result(
        application, reservations, identity
    )
    acceptance = await _acceptance(application, task, product)
    url = _pr_url(github.result)
    cost = await inspect_cost_evidence(application.app, identity)
    return {
        "id": identity.public_id,
        "outcome": "recorded_verified_delivery",
        "observation": "durable_history",
        "commit": github.result.head_commit,
        "tree": remote.result.tree,
        "workspace_revision": acceptance["final_revision"],
        "pull_request": {"url": url, "number": github.result.pull_request.number},
        "acceptance": {"href": _link(identity, "acceptance"), **acceptance},
        "cost": {"href": _link(identity, "cost"), **cost},
        "git_result_digest": _digest(remote.artifact.sha256),
        "github_result_digest": _digest(github.artifact.sha256),
    }
