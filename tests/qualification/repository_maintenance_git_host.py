"""Host-only broker construction; never import into coding tools or environments."""

from pathlib import Path

from configuration.maintenance import (  # ty: ignore[unresolved-import]
    _configured_json,
    configured_maintenance_git_authority,
)
from integrations.remote_git import (  # ty: ignore[unresolved-import]
    build_remote_git_delivery_broker,
)

from cayu import RemoteGitHttpCredentials
from cayu.vaults import LocalEnvVault, SecretRef


def configured_git_broker(artifact_store):
    """Construct without secret resolution or remote operations; native checks apply."""
    try:
        value = _configured_json("CAYU_MAINTENANCE_GIT_HOST_JSON")
        if type(value) is not dict or set(value) != {"broker_root", "git_executable", "remote_url"}:
            raise ValueError
        if any(type(item) is not str or not item for item in value.values()):
            raise ValueError
        if not all(Path(value[key]).is_absolute() for key in ("broker_root", "git_executable")):
            raise ValueError
        repository, _commit, security, _limits = configured_maintenance_git_authority()
        if repository.remote_alias != "origin":
            raise ValueError
        credentials = None
        if security.credential_profile_id == "maintenance-git-https":
            credentials = RemoteGitHttpCredentials(
                credential_profile_id=security.credential_profile_id,
                username=SecretRef(name="git-user"),
                password=SecretRef(name="git-token"),
                resolver=LocalEnvVault(
                    {
                        "git-user": "CAYU_MAINTENANCE_GIT_USER",
                        "git-token": "CAYU_MAINTENANCE_GIT_TOKEN",
                    }
                ),
            )
        elif security.credential_profile_id != "none":
            raise ValueError
        return build_remote_git_delivery_broker(
            artifact_store=artifact_store,
            broker_root=value["broker_root"],
            git_executable=value["git_executable"],
            remote_url=value["remote_url"],
            repository_id=repository.repository_id,
            broker_repository_id=repository.broker_repository_id,
            remote_identity=repository.remote_identity,
            default_branch_ref=repository.base_ref,
            credentials=credentials,
            egress_profile_id=security.egress_profile_id,
        )
    except (ValueError, TypeError, RecursionError):
        raise ValueError("Invalid or missing maintenance Git host configuration.") from None
