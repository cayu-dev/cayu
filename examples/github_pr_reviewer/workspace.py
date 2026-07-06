from __future__ import annotations

import json
from pathlib import Path

from cayu import (
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    GitRepositoryBinding,
    LocalEnvVault,
    LocalRunner,
    LocalWorkspace,
    PassthroughProxy,
)

_shared_vault = LocalEnvVault({"github_token": "GITHUB_TOKEN"})
_shared_proxy = PassthroughProxy(_shared_vault)


class PRReviewWorkspaceFactory(EnvironmentFactory):
    """One fresh clone per review session, checked out at the queued PR head SHA."""

    def __init__(self, base_root: Path, *, with_credentials: bool = True) -> None:
        self._base_root = base_root
        self._with_credentials = with_credentials

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        repo_url = request.metadata.get("repo_url")
        pr_number = request.metadata.get("pr_number")
        head_sha = request.metadata.get("head_sha")
        if (
            type(repo_url) is not str
            or not repo_url
            or type(pr_number) is not int
            or pr_number <= 0
            or type(head_sha) is not str
            or not head_sha
        ):
            raise ValueError(
                "PRReviewWorkspaceFactory requires 'repo_url', integer 'pr_number', and "
                "'head_sha' in RunRequest.metadata; "
                "got: " + json.dumps(request.metadata)
            )
        checkout_ref = f"refs/cayu/pr-{pr_number}"
        root = self._base_root / request.session_id
        root.mkdir(parents=True, exist_ok=True)
        environment = Environment(
            EnvironmentSpec(name=request.environment_name),
            workspace=LocalWorkspace(root, workspace_id=request.session_id),
            runner=LocalRunner(root),
            binding=GitRepositoryBinding(
                repo_url=repo_url,
                ref=head_sha,
                fetch_refspecs=[f"+refs/pull/{pr_number}/head:{checkout_ref}"],
            ),
            vault=_shared_vault if self._with_credentials else None,
            proxy=_shared_proxy if self._with_credentials else None,
        )
        return EnvironmentFactoryResult(environment=environment)
