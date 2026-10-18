from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from cayu.environments import (
    EnvironmentFactoryRequest,
    GitRepositoryBinding,
    LocalWorkspaceEnvironmentFactory,
)
from cayu.proxies import PassthroughProxy
from cayu.vaults import LocalEnvVault

_shared_vault = LocalEnvVault({"github_token": "GITHUB_TOKEN"})
_shared_proxy = PassthroughProxy(_shared_vault)


def _review_binding(root: Path, request: EnvironmentFactoryRequest) -> GitRepositoryBinding:
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
            "'head_sha' in RunRequest.metadata."
        )
    return GitRepositoryBinding(
        repo_url=repo_url,
        ref=head_sha,
        fetch_refspecs=[f"+refs/pull/{pr_number}/head:refs/cayu/pr-{pr_number}"],
    )


class PRReviewWorkspaceFactory(LocalWorkspaceEnvironmentFactory):
    """Bind the queued PR head inside a prepared, allowed request directory."""

    def __init__(self, base_root: Path, *, with_credentials: bool = True) -> None:
        base_root = base_root.resolve()
        super().__init__(
            allowed_roots=[base_root],
            root_from_request=lambda request: (
                base_root / sha256(request.session_id.encode()).hexdigest()
            ),
            binding_factory=_review_binding,
            vault=_shared_vault if with_credentials else None,
            proxy=_shared_proxy if with_credentials else None,
        )
