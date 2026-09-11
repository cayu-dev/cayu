"""Required operator-owned configuration; no default credentials or live prices."""

import json
import os

from cayu import (
    BudgetPolicy,
    GitHubCheckPolicy,
    GitHubDeliveryLimits,
    GitHubPullRequestMetadata,
    GitHubReviewPolicy,
    GitHubSecurityAuthority,
    RemoteGitCommitAuthority,
    RemoteGitDeliveryLimits,
    RemoteGitRepositoryAuthority,
    RemoteGitSecurityAuthority,
)
from cayu.server import ProductPrincipal
from tests.qualification.repository_maintenance_auth import MaintenanceAccess
from tests.qualification.repository_maintenance_budget import require_maintenance_budget


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError
        value[key] = item
    return value


def _invalid_constant(_value):
    raise ValueError


def _configured_json(name):
    raw = os.environ.get(name)
    if raw is None or not raw or len(raw.encode("utf-8")) > 65536:
        raise ValueError
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=_invalid_constant)


def configured_maintenance_budget() -> BudgetPolicy:
    """Read a fresh policy at construction, before acquiring app resources."""
    try:
        value = _configured_json("CAYU_MAINTENANCE_BUDGET_JSON")
        return require_maintenance_budget(BudgetPolicy.model_validate(value))
    except (ValueError, TypeError, RecursionError):
        raise ValueError("Invalid or missing maintenance budget configuration.") from None


def configured_maintenance_access() -> MaintenanceAccess:
    """Load host-only credentials; the existing access adapter owns authentication."""
    try:
        value = _configured_json("CAYU_MAINTENANCE_ACCESS_JSON")
        if type(value) is not dict or set(value) != {"operator_token", "product_tokens"}:
            raise ValueError
        tokens = value["product_tokens"]
        if type(tokens) is not dict or not 1 <= len(tokens) <= 64:
            raise ValueError
        principals = {}
        for token, principal in tokens.items():
            if type(principal) is not dict or set(principal) != {"tenant_id", "subject_id"}:
                raise ValueError
            principals[token] = ProductPrincipal(**principal)
        return MaintenanceAccess(product_tokens=principals, operator_token=value["operator_token"])
    except (ValueError, TypeError, RecursionError):
        raise ValueError("Invalid or missing maintenance access configuration.") from None


def configured_maintenance_git_authority() -> tuple[
    RemoteGitRepositoryAuthority,
    RemoteGitCommitAuthority,
    RemoteGitSecurityAuthority,
    RemoteGitDeliveryLimits,
]:
    """Load host data, not approval; retain it in the first accepted phase request.

    In particular, authored_at is explicit configuration, never a fresh retry
    timestamp. Native delivery still validates source, configuration and refs.
    """
    try:
        value = _configured_json("CAYU_MAINTENANCE_GIT_JSON")
        if type(value) is not dict or set(value) != {"repository", "commit", "security", "limits"}:
            raise ValueError
        if any(type(section) is not dict for section in value.values()):
            raise ValueError
        return (
            RemoteGitRepositoryAuthority.model_validate(value["repository"], strict=True),
            RemoteGitCommitAuthority.model_validate(value["commit"], strict=True),
            RemoteGitSecurityAuthority.model_validate(value["security"], strict=True),
            RemoteGitDeliveryLimits.model_validate(value["limits"], strict=True),
        )
    except (ValueError, TypeError, RecursionError):
        raise ValueError("Invalid or missing maintenance Git configuration.") from None


def configured_maintenance_github_authority():
    """Explicit request configuration; the native factory validates the final tuple."""
    try:
        value = _configured_json("CAYU_MAINTENANCE_GITHUB_JSON")
        sections = {
            "metadata": GitHubPullRequestMetadata,
            "checks": GitHubCheckPolicy,
            "reviews": GitHubReviewPolicy,
            "security": GitHubSecurityAuthority,
            "limits": GitHubDeliveryLimits,
        }
        scalars = {
            "requested_at",
            "repository_alias",
            "installation_id",
            "account_id",
            "mode",
            "existing_pull_request_number",
        }
        if type(value) is not dict or set(value) != scalars | set(sections):
            raise ValueError
        for name, model in sections.items():
            if type(value[name]) is not dict:
                raise ValueError
            value[name] = model.model_validate_json(
                json.dumps(value[name], allow_nan=False), strict=True
            )
        return value
    except (ValueError, TypeError, RecursionError):
        raise ValueError("Invalid or missing maintenance GitHub configuration.") from None
