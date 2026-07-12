from __future__ import annotations

import asyncio
from pathlib import Path

from examples._advanced_support.fake_github import FakeGitHubServer, GitHubClient
from examples.repo_maintainer_tournament.deterministic import run


def test_fake_github_exercises_the_http_contract_and_idempotent_pr_creation() -> None:
    with FakeGitHubServer() as server:
        client = GitHubClient(server.base_url)
        pull = asyncio.run(client.get_pull("acme", "calculator", 1))
        files = asyncio.run(client.list_pull_files("acme", "calculator", 1))
        first = asyncio.run(
            client.ensure_pull(
                "acme",
                "calculator",
                title="Fix divide by zero",
                head="cayu/fix-divide",
                base="main",
                body="Automated candidate winner.",
            )
        )
        second = asyncio.run(
            client.ensure_pull(
                "acme",
                "calculator",
                title="Fix divide by zero",
                head="cayu/fix-divide",
                base="main",
                body="Automated candidate winner.",
            )
        )

    assert pull["number"] == 1
    assert files[0]["filename"] == "calculator.py"
    assert first["number"] == second["number"] == 101
    assert server.state.list_pull_requests == 2
    assert server.state.create_pull_requests == 1
    assert len(server.state.created_pulls) == 1


def test_repo_tournament_promotes_only_the_smallest_correct_candidate(tmp_path: Path) -> None:
    result = asyncio.run(run(tmp_path))

    assert result.status == "verified"
    assert result.assertions == {
        "candidate_changes_applied": True,
        "evaluator_rejected_test_weakening": True,
        "exactly_one_pull_request": True,
        "fake_github_api_exercised": True,
        "smallest_correct_patch_selected": True,
        "workspaces_are_isolated": True,
    }
    assert result.metrics["winner"] == "minimal-boundary-check"
    assert result.metrics["pull_request_number"] == 101
    assert result.metrics["github_list_requests"] == 2
    assert result.metrics["github_create_requests"] == 1
    assert result.metrics["github_created_pulls"] == 1
    assert all(candidate["changes"] for candidate in result.outputs["candidates"].values())
