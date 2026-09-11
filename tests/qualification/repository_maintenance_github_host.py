"""Host-only connector factories; no network calls or token resolution at setup."""

from functools import partial

from configuration.maintenance import (  # ty: ignore[unresolved-import]
    _configured_json,
    configured_maintenance_git_authority,
    configured_maintenance_github_authority,
)
from integrations.github import build_github_connector  # ty: ignore[unresolved-import]

from cayu import GitHubCredentials, GitHubRepositoryConfig, github_connector_behavior_fingerprint
from cayu.vaults import LocalEnvVault, SecretRef


def configured_github_repository():
    """Read host-owned native configuration without resolving credentials or doing I/O."""
    try:
        host = _configured_json("CAYU_MAINTENANCE_GITHUB_HOST_JSON")
        if type(host) is not dict or set(host) != {"owner", "repository_name", "api_base_url"}:
            raise ValueError
        if any(type(value) is not str or not value for value in host.values()):
            raise ValueError
        repository, _commit, _security, _limits = configured_maintenance_git_authority()
        authority = configured_maintenance_github_authority()
        security = authority["security"]
        if (
            authority["repository_alias"] != "github"
            or security.connector_id != "maintenance-app-github"
            or security.connector_behavior_fingerprint != github_connector_behavior_fingerprint()
            or security.credential_profile_id != "github-installation-token"
            or security.egress_profile_id != "github-api-only"
        ):
            raise ValueError
        credentials = GitHubCredentials(
            credential_profile_id="github-installation-token",
            token=SecretRef(name="github-token"),
            resolver=LocalEnvVault({"github-token": "CAYU_MAINTENANCE_GITHUB_TOKEN"}),
        )
        return GitHubRepositoryConfig(
            alias="github",
            repository_id=repository.repository_id,
            installation_id=authority["installation_id"],
            account_id=authority["account_id"],
            api_base_url=host["api_base_url"],
            owner=host["owner"],
            name=host["repository_name"],
            credential_profile_id=credentials.credential_profile_id,
            egress_profile_id=security.egress_profile_id,
            credentials=credentials,
        )
    except (ValueError, TypeError, RecursionError):
        raise ValueError("Invalid or missing maintenance GitHub host configuration.") from None


def configured_github_connector_factory(artifact_store):
    """Validate before claiming tasks; return a fresh native connector on each call."""
    configured = configured_github_repository()
    return partial(
        build_github_connector,
        artifact_store=artifact_store,
        repository_id=configured.repository_id,
        installation_id=configured.installation_id,
        account_id=configured.account_id,
        owner=configured.owner,
        repository_name=configured.name,
        token_ref=configured.credentials.token,
        secret_resolver=configured.credentials.resolver,
        api_base_url=configured.api_base_url,
    )
