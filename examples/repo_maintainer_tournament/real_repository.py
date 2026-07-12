from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from examples._advanced_support.fake_github import GitHubClient
from examples.repo_maintainer_tournament.candidate_gates import (
    apply_candidate,
    run_candidate_gates,
)

if TYPE_CHECKING:
    from examples._advanced_support.results import ScenarioResult

REPOSITORY_FILES = ("calculator.py", "test_calculator.py")


class GitHubPullClient(Protocol):
    async def get_pull(self, owner: str, repo: str, number: int) -> dict[str, Any]: ...

    async def list_pull_files(
        self,
        owner: str,
        repo: str,
        number: int,
    ) -> list[dict[str, Any]]: ...

    async def ensure_pull(
        self,
        owner: str,
        repo: str,
        *,
        title: str,
        head: str,
        base: str,
        body: str,
    ) -> dict[str, Any]: ...

    async def list_open_pulls(
        self,
        owner: str,
        repo: str,
        *,
        head: str,
        base: str,
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class LiveRepositoryConfig:
    owner: str
    repo: str
    source_pull_number: int
    token: str = field(repr=False)
    clone_url: str | None = None
    api_url: str = "https://api.github.com"

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @classmethod
    def from_environment(cls) -> LiveRepositoryConfig | None:
        full_name = os.environ.get("CAYU_REPO_MAINTAINER_REPOSITORY")
        if not full_name:
            return None
        parts = full_name.strip().split("/")
        if len(parts) != 2 or not all(parts):
            raise RuntimeError(
                "CAYU_REPO_MAINTAINER_REPOSITORY must use the owner/repository form."
            )
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise RuntimeError(
                "Set GITHUB_TOKEN to run the repository maintainer against real GitHub."
            )
        try:
            source_pull_number = int(os.environ.get("CAYU_REPO_MAINTAINER_SOURCE_PULL", "1"))
        except ValueError as exc:
            raise RuntimeError("CAYU_REPO_MAINTAINER_SOURCE_PULL must be an integer.") from exc
        if source_pull_number < 1:
            raise RuntimeError("CAYU_REPO_MAINTAINER_SOURCE_PULL must be positive.")
        owner, repo = parts
        return cls(
            owner=owner,
            repo=repo,
            source_pull_number=source_pull_number,
            token=token,
            clone_url=os.environ.get("CAYU_REPO_MAINTAINER_CLONE_URL"),
        )


@dataclass(frozen=True)
class RepositoryPreparation:
    repository_dir: Path
    worktree_root: Path
    source_sha: str
    base_ref: str
    pull: dict[str, Any]
    files: list[dict[str, Any]]
    baseline: dict[str, str]


@dataclass(frozen=True)
class PromotionEvidence:
    branch_name: str
    commit_sha: str
    pull_request_number: int
    pull_request_url: str
    assertions: dict[str, bool]
    metrics: dict[str, Any]
    outputs: dict[str, Any]


class RealRepositoryBoundary:
    def __init__(
        self,
        config: LiveRepositoryConfig,
        *,
        github_client: GitHubPullClient | None = None,
    ) -> None:
        self.config = config
        self._injected_client = github_client

    def _client(self) -> GitHubPullClient:
        if self._injected_client is not None:
            return self._injected_client
        return GitHubClient(self.config.api_url, token=self.config.token)

    async def prepare(self, root: Path) -> RepositoryPreparation:
        client = self._client()
        pull = await client.get_pull(
            self.config.owner,
            self.config.repo,
            self.config.source_pull_number,
        )
        files = await client.list_pull_files(
            self.config.owner,
            self.config.repo,
            self.config.source_pull_number,
        )
        source_sha = _nested_nonblank(pull, "head", "sha")
        base_ref = _nested_nonblank(pull, "base", "ref")
        run_root = root / ".cayu-example-repositories" / f"repo-{uuid4().hex}"
        repository_dir = run_root / "repository"
        worktree_root = run_root / "worktrees"
        run_root.mkdir(parents=True)
        clone_url = self.config.clone_url or (
            f"git@github.com:{self.config.owner}/{self.config.repo}.git"
        )
        _git("clone", "--no-checkout", clone_url, str(repository_dir))
        _git("checkout", "--detach", source_sha, cwd=repository_dir)
        baseline = {
            relative: (repository_dir / relative).read_text(encoding="utf-8")
            for relative in REPOSITORY_FILES
        }
        return RepositoryPreparation(
            repository_dir=repository_dir,
            worktree_root=worktree_root,
            source_sha=source_sha,
            base_ref=base_ref,
            pull=pull,
            files=files,
            baseline=baseline,
        )

    def create_candidate_workspace(
        self,
        preparation: RepositoryPreparation,
        strategy: str,
    ) -> Path:
        workspace = preparation.worktree_root / strategy
        workspace.parent.mkdir(parents=True, exist_ok=True)
        _git(
            "worktree",
            "add",
            "--detach",
            str(workspace),
            preparation.source_sha,
            cwd=preparation.repository_dir,
        )
        return workspace

    async def promote(
        self,
        preparation: RepositoryPreparation,
        workspace: Path,
        *,
        winner: str,
    ) -> PromotionEvidence:
        branch_name = f"cayu/live-maintainer-{uuid4().hex[:12]}"
        _git("switch", "-c", branch_name, cwd=workspace)
        _git("config", "user.name", "Cayu Repository Maintainer", cwd=workspace)
        _git(
            "config",
            "user.email",
            "cayu-repo-maintainer@users.noreply.github.com",
            cwd=workspace,
        )
        _git("add", *REPOSITORY_FILES, cwd=workspace)
        _git("commit", "-m", "Fix divide-by-zero behavior", cwd=workspace)
        commit_sha = _git("rev-parse", "HEAD", cwd=workspace)
        changed_from_source = set(
            filter(
                None,
                _git(
                    "diff",
                    "--name-only",
                    f"{preparation.source_sha}..{commit_sha}",
                    cwd=workspace,
                ).splitlines(),
            )
        )
        _git("push", "origin", f"HEAD:refs/heads/{branch_name}", cwd=workspace)
        remote_line = _git(
            "ls-remote",
            "origin",
            f"refs/heads/{branch_name}",
            cwd=workspace,
        )
        remote_sha = remote_line.split(maxsplit=1)[0] if remote_line else ""

        body = f"Promoted {winner} after deterministic and evaluator gates."
        created = await self._client().ensure_pull(
            self.config.owner,
            self.config.repo,
            title="Fix divide by zero",
            head=branch_name,
            base=preparation.base_ref,
            body=body,
        )
        retried = await self._client().ensure_pull(
            self.config.owner,
            self.config.repo,
            title="Fix divide by zero",
            head=branch_name,
            base=preparation.base_ref,
            body=body,
        )
        pull_number = _positive_int(created.get("number"), "created pull number")
        verified = await self._client().get_pull(
            self.config.owner,
            self.config.repo,
            pull_number,
        )
        verified_files = await self._client().list_pull_files(
            self.config.owner,
            self.config.repo,
            pull_number,
        )
        matching = await self._client().list_open_pulls(
            self.config.owner,
            self.config.repo,
            head=branch_name,
            base=preparation.base_ref,
        )
        source_changed_files = {
            str(item.get("filename")) for item in preparation.files if item.get("filename")
        }
        expected_pull_files = source_changed_files | changed_from_source
        actual_pull_files = {
            str(item.get("filename")) for item in verified_files if item.get("filename")
        }
        verified_head = verified.get("head")
        verified_base = verified.get("base")
        retried_number = _positive_int(retried.get("number"), "retried pull number")
        pull_url = str(verified.get("html_url") or created.get("html_url") or "")
        assertions = {
            "exactly_one_pull_request": (
                pull_number == retried_number
                and len(matching) == 1
                and matching[0].get("number") == pull_number
            ),
            "real_github_pull_verified": (
                isinstance(verified_head, dict)
                and verified_head.get("ref") == branch_name
                and verified_head.get("sha") == commit_sha
                and isinstance(verified_base, dict)
                and verified_base.get("ref") == preparation.base_ref
                and actual_pull_files == expected_pull_files
            ),
            "real_repository_cloned": (
                (preparation.repository_dir / ".git").exists()
                and all((preparation.repository_dir / path).is_file() for path in REPOSITORY_FILES)
            ),
            "winner_commit_pushed": remote_sha == commit_sha,
        }
        return PromotionEvidence(
            branch_name=branch_name,
            commit_sha=commit_sha,
            pull_request_number=pull_number,
            pull_request_url=pull_url,
            assertions=assertions,
            metrics={
                "repository": self.config.full_name,
                "source_pull_number": self.config.source_pull_number,
                "pull_request_number": pull_number,
                "branch_name": branch_name,
                "commit_sha": commit_sha,
            },
            outputs={
                "pull_request_url": pull_url,
                "pull_request_files": sorted(actual_pull_files),
                "source_pull_files": sorted(source_changed_files),
                "winner_changed_files": sorted(changed_from_source),
            },
        )


async def promote_tournament_result(
    result: ScenarioResult,
    *,
    root: Path,
    boundary: RealRepositoryBoundary,
    preparation: RepositoryPreparation,
) -> ScenarioResult:
    config = boundary.config
    candidates = result.outputs.get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        raise RuntimeError("Tournament result did not include candidate changes.")
    original_gates = result.outputs.get("gates")
    if not isinstance(original_gates, dict):
        raise RuntimeError("Tournament result did not include gate evidence.")

    real_gates: dict[str, dict[str, Any]] = {}
    workspaces: dict[str, Path] = {}
    for candidate_value in candidates.values():
        if not isinstance(candidate_value, dict):
            raise RuntimeError("Tournament candidate was not an object.")
        strategy = candidate_value.get("strategy")
        if not isinstance(strategy, str) or not strategy:
            raise RuntimeError("Tournament candidate did not include a strategy.")
        workspace = boundary.create_candidate_workspace(preparation, strategy)
        apply_candidate(workspace, candidate_value, preparation.baseline)
        real_gates[strategy] = run_candidate_gates(workspace, preparation.baseline)
        workspaces[strategy] = workspace

    winner = result.metrics.get("winner")
    if not isinstance(winner, str) or winner not in workspaces:
        raise RuntimeError("Tournament result did not identify a valid winning strategy.")
    eligible = [
        (strategy, gate["diff_lines"])
        for strategy, gate in real_gates.items()
        if gate["tests_passed"] and not gate["test_files_changed"]
    ]
    if not eligible:
        raise RuntimeError("No real-repository candidate passed the deterministic gates.")
    expected_winner = min(eligible, key=lambda item: item[1])[0]
    normalized_original = {
        strategy: {
            "tests_passed": gate.get("tests_passed"),
            "test_files_changed": gate.get("test_files_changed"),
            "diff_lines": gate.get("diff_lines"),
        }
        for strategy, gate in original_gates.items()
        if isinstance(strategy, str) and isinstance(gate, dict)
    }
    normalized_real = {
        strategy: {
            "tests_passed": gate["tests_passed"],
            "test_files_changed": gate["test_files_changed"],
            "diff_lines": gate["diff_lines"],
        }
        for strategy, gate in real_gates.items()
    }
    pre_promotion_assertions = {
        "real_candidate_changes_applied": all(
            all(
                (workspaces[strategy] / change["path"]).read_text(encoding="utf-8")
                == change["content"]
                for change in candidate["changes"]
            )
            for candidate in candidates.values()
            if isinstance(candidate, dict)
            for strategy in [candidate["strategy"]]
        ),
        "real_tournament_gates_replayed": (
            normalized_real == normalized_original and expected_winner == winner
        ),
        "real_worktrees_are_isolated": (
            len({path.resolve() for path in workspaces.values()}) == len(workspaces)
            and all((path / ".git").exists() for path in workspaces.values())
        ),
    }
    failed_preconditions = [name for name, passed in pre_promotion_assertions.items() if not passed]
    if failed_preconditions:
        raise RuntimeError(
            "Refusing to publish because real-repository verification failed: "
            + ", ".join(failed_preconditions)
        )
    promotion = await boundary.promote(preparation, workspaces[winner], winner=winner)
    result.assertions.update(
        {
            **pre_promotion_assertions,
            **promotion.assertions,
        }
    )
    for key in (
        "github_create_requests",
        "github_list_requests",
        "github_created_pulls",
        "pull_request_number",
    ):
        result.metrics.pop(key, None)
    result.metrics.update(promotion.metrics)
    result.outputs.update(
        {
            "real_repository_gates": real_gates,
            "real_repository_source": {
                "repository": config.full_name,
                "source_pull_number": config.source_pull_number,
                "source_sha": preparation.source_sha,
                "base_ref": preparation.base_ref,
            },
            **promotion.outputs,
        }
    )
    result.status = "verified" if all(result.assertions.values()) else "failed"
    result.write(root)
    result.require_verified()
    return result


def _git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _nested_nonblank(payload: dict[str, Any], section: str, key: str) -> str:
    nested = payload.get(section)
    if not isinstance(nested, dict):
        raise RuntimeError(f"GitHub pull response did not include {section}.{key}.")
    value = nested.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"GitHub pull response did not include {section}.{key}.")
    return value


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RuntimeError(f"GitHub response did not include a valid {name}.")
    return value
