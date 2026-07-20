from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest
from examples._advanced_support.results import ScenarioResult
from examples.repo_maintainer_tournament.real_repository import (
    LiveRepositoryConfig,
    PromotionEvidence,
    RealRepositoryBoundary,
    RepositoryPreparation,
    promote_tournament_result,
)


class RecordingGitHubClient:
    def __init__(self, source_sha: str, origin: Path) -> None:
        self.source_sha = source_sha
        self.origin = origin
        self.published_branch = ""
        self.published_sha = ""
        self.ensure_calls = 0
        self.repository_calls: list[tuple[str, str, str]] = []

    async def get_pull(self, owner: str, repo: str, number: int) -> dict[str, Any]:
        self.repository_calls.append(("get_pull", owner, repo))
        if number == 1:
            return {
                "number": 1,
                "title": "Handle divide by zero",
                "body": "The regression test must pass.",
                "head": {"ref": "bug-report", "sha": self.source_sha},
                "base": {"ref": "main"},
            }
        return {
            "number": 2,
            "html_url": "https://github.example/pull/2",
            "head": {"ref": self.published_branch, "sha": self.published_sha},
            "base": {"ref": "main"},
        }

    async def list_pull_files(
        self,
        owner: str,
        repo: str,
        number: int,
    ) -> list[dict[str, Any]]:
        self.repository_calls.append(("list_pull_files", owner, repo))
        if number == 1:
            return [{"filename": "test_calculator.py"}]
        return [{"filename": "calculator.py"}, {"filename": "test_calculator.py"}]

    async def ensure_pull(
        self,
        owner: str,
        repo: str,
        *,
        title: str,
        head: str,
        base: str,
        body: str,
    ) -> dict[str, Any]:
        self.repository_calls.append(("ensure_pull", owner, repo))
        self.ensure_calls += 1
        self.published_branch = head
        self.published_sha = _git(
            f"--git-dir={self.origin}",
            "rev-parse",
            f"refs/heads/{head}",
        )
        return {"number": 2, "html_url": "https://github.example/pull/2"}

    async def list_open_pulls(
        self,
        owner: str,
        repo: str,
        *,
        head: str,
        base: str,
    ) -> list[dict[str, Any]]:
        self.repository_calls.append(("list_open_pulls", owner, repo))
        return [{"number": 2, "head": {"ref": head}, "base": {"ref": base}}]


class NoPublishBoundary(RealRepositoryBoundary):
    def __init__(self, config: LiveRepositoryConfig, github: RecordingGitHubClient) -> None:
        super().__init__(config, github_client=github)
        self.promote_calls = 0

    async def promote(
        self,
        preparation: RepositoryPreparation,
        workspace: Path,
        *,
        winner: str,
    ) -> PromotionEvidence:
        self.promote_calls += 1
        raise AssertionError("promotion must not run after gate divergence")


def _git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _seed_remote(root: Path) -> tuple[Path, str]:
    origin = root / "origin.git"
    seed = root / "seed"
    _git("init", "--bare", str(origin))
    _git("init", "-b", "main", str(seed))
    _git("config", "user.name", "Fixture", cwd=seed)
    _git("config", "user.email", "fixture@example.com", cwd=seed)
    (seed / "calculator.py").write_text(
        "def divide(a: float, b: float) -> float:\n    return a / b\n",
        encoding="utf-8",
    )
    (seed / "test_calculator.py").write_text(
        "from calculator import divide\n\n"
        "def test_divide_returns_quotient() -> None:\n    assert divide(8, 2) == 4\n",
        encoding="utf-8",
    )
    _git("add", ".", cwd=seed)
    _git("commit", "-m", "Seed main", cwd=seed)
    _git("remote", "add", "origin", str(origin), cwd=seed)
    _git("push", "-u", "origin", "main", cwd=seed)
    _git("switch", "-c", "bug-report", cwd=seed)
    (seed / "test_calculator.py").write_text(
        "import pytest\n\nfrom calculator import divide\n\n"
        "def test_divide_returns_quotient() -> None:\n    assert divide(8, 2) == 4\n\n"
        "def test_divide_rejects_zero_divisor() -> None:\n"
        '    with pytest.raises(ValueError, match="divisor must not be zero"):\n'
        "        divide(1, 0)\n",
        encoding="utf-8",
    )
    _git("add", "test_calculator.py", cwd=seed)
    _git("commit", "-m", "Add regression", cwd=seed)
    _git("push", "-u", "origin", "bug-report", cwd=seed)
    return origin, _git("rev-parse", "HEAD", cwd=seed)


def test_real_repository_boundary_pushes_and_verifies_one_pull_request(tmp_path: Path) -> None:
    origin, source_sha = _seed_remote(tmp_path)
    github = RecordingGitHubClient(source_sha, origin)
    boundary = RealRepositoryBoundary(
        LiveRepositoryConfig(
            owner="fixture-owner",
            repo="fixture-repository",
            source_pull_number=1,
            token="test-token",
            clone_url=str(origin),
        ),
        github_client=github,
    )

    preparation = asyncio.run(boundary.prepare(tmp_path / "run"))
    workspace = boundary.create_candidate_workspace(preparation, "minimal-boundary-check")
    (workspace / "calculator.py").write_text(
        "def divide(a: float, b: float) -> float:\n"
        "    if b == 0:\n"
        '        raise ValueError("divisor must not be zero")\n'
        "    return a / b\n",
        encoding="utf-8",
    )
    promotion = asyncio.run(
        boundary.promote(
            preparation,
            workspace,
            winner="minimal-boundary-check",
        )
    )
    assert all(promotion.assertions.values())
    assert github.ensure_calls == 2
    assert {method for method, _, _ in github.repository_calls} == {
        "ensure_pull",
        "get_pull",
        "list_open_pulls",
        "list_pull_files",
    }
    assert all(
        (owner, repo) == ("fixture-owner", "fixture-repository")
        for _, owner, repo in github.repository_calls
    )
    assert promotion.pull_request_number == 2
    assert _git(
        f"--git-dir={origin}",
        "show",
        f"refs/heads/{promotion.branch_name}:calculator.py",
    ).startswith("def divide")


def test_live_repository_config_is_explicit_and_requires_api_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAYU_REPO_MAINTAINER_REPOSITORY", raising=False)
    assert LiveRepositoryConfig.from_environment() is None

    monkeypatch.setenv(
        "CAYU_REPO_MAINTAINER_REPOSITORY",
        "fixture-owner/fixture-repository",
    )
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        LiveRepositoryConfig.from_environment()

    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("CAYU_REPO_MAINTAINER_SOURCE_PULL", "7")
    config = LiveRepositoryConfig.from_environment()

    assert config is not None
    assert config.full_name == "fixture-owner/fixture-repository"
    assert config.source_pull_number == 7


def test_real_gate_divergence_fails_before_push_or_pull_request(tmp_path: Path) -> None:
    origin, source_sha = _seed_remote(tmp_path)
    github = RecordingGitHubClient(source_sha, origin)
    config = LiveRepositoryConfig(
        owner="fixture-owner",
        repo="fixture-repository",
        source_pull_number=1,
        token="test-token",
        clone_url=str(origin),
    )
    boundary = NoPublishBoundary(config, github)
    preparation = asyncio.run(boundary.prepare(tmp_path / "run"))
    fixed_calculator = (
        "def divide(a: float, b: float) -> float:\n"
        "    if b == 0:\n"
        '        raise ValueError("divisor must not be zero")\n'
        "    return a / b\n"
    )
    weakened_test = (
        "from calculator import divide\n\n"
        "def test_divide_returns_quotient() -> None:\n    assert divide(8, 2) == 4\n"
    )
    candidates = {
        "minimal": {
            "strategy": "minimal-boundary-check",
            "changes": [{"path": "calculator.py", "content": fixed_calculator}],
        },
        "test-weakener": {
            "strategy": "weaken-tests",
            "changes": [{"path": "test_calculator.py", "content": weakened_test}],
        },
        "broad": {
            "strategy": "broad-rewrite",
            "changes": [{"path": "calculator.py", "content": fixed_calculator}],
        },
    }
    result = ScenarioResult(
        scenario="repo-maintainer-tournament",
        mode="live",
        status="verified",
        assertions={"preliminary_tournament_verified": True},
        sessions=[],
        metrics={"winner": "minimal-boundary-check"},
        outputs={
            "candidates": candidates,
            "gates": {
                strategy: {
                    "tests_passed": True,
                    "test_files_changed": False,
                    "diff_lines": 999,
                }
                for strategy in (
                    "minimal-boundary-check",
                    "weaken-tests",
                    "broad-rewrite",
                )
            },
        },
    )

    with pytest.raises(RuntimeError, match="Refusing to publish"):
        asyncio.run(
            promote_tournament_result(
                result,
                root=tmp_path / "run",
                boundary=boundary,
                preparation=preparation,
            )
        )

    assert boundary.promote_calls == 0
    assert github.ensure_calls == 0
