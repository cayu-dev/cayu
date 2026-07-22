from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import httpx
from examples.github_pr_reviewer.github_tools import GITHUB_API, DemoPRDiffTool
from examples.github_pr_reviewer.reviewer_app import build_app, build_provider
from examples.github_pr_reviewer.worker import enqueue_pr_review, run_pr_review_worker_once

from cayu import ScriptedModelProvider
from cayu.providers import ModelStreamEvent


async def review_pr(owner: str, repo: str, pr_number: int) -> None:
    """Live review of a real PR. Requires a provider key + GITHUB_TOKEN."""
    provider, model = build_provider()
    async with httpx.AsyncClient(timeout=20.0) as client:
        pr = (
            await client.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}",
                headers={
                    "Accept": "application/vnd.github+json",
                    **(
                        {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}
                        if os.environ.get("GITHUB_TOKEN")
                        else {}
                    ),
                },
            )
        ).json()
    scratch = Path(tempfile.mkdtemp(prefix="cayu_pr_reviewer_"))
    app, task_store = build_app(
        scratch / "data" / "cayu.db",
        scratch / "workspaces",
        provider=provider,
        model=model,
    )
    task = await enqueue_pr_review(
        task_store,
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        repo_url=pr["base"]["repo"]["clone_url"],
        head_ref=pr["head"]["ref"],
        head_sha=pr["head"]["sha"],
        base_ref=pr["base"]["ref"],
    )
    handled = await run_pr_review_worker_once(app, task_store, worker_id="live-worker")
    assert handled == 1
    finished = await task_store.load_task(task.id)
    assert finished is not None
    print("task:", finished.id, finished.status, finished.session_id)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _create_demo_origin(root: Path) -> tuple[Path, str]:
    """Create a local bare origin with a GitHub-style refs/pull/1/head ref."""
    origin = root / "origin.git"
    seed = root / "seed"
    seed.mkdir(parents=True)

    _git(root, "init", "--bare", str(origin))
    _git(seed, "init")
    _git(seed, "checkout", "-b", "main")
    _git(seed, "config", "user.email", "demo@example.com")
    _git(seed, "config", "user.name", "Cayu Demo")
    (seed / "README.md").write_text("demo repo\n", encoding="utf-8")
    _git(seed, "add", "README.md")
    _git(seed, "commit", "-m", "initial")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "origin", "main")
    _git(origin, "symbolic-ref", "HEAD", "refs/heads/main")

    (seed / "review_target.py").write_text(
        "def answer() -> int:\n    return 42\n",
        encoding="utf-8",
    )
    _git(seed, "add", "review_target.py")
    _git(seed, "commit", "-m", "add review target")
    pr_head = _git(seed, "rev-parse", "HEAD")
    _git(seed, "push", "origin", "HEAD:refs/pull/1/head")
    return origin, pr_head


async def demo() -> None:
    """Run a deterministic no-key PR review against a local fixture repository."""
    owner, repo, pr_number = "local", "fixture", 1
    scratch = Path(tempfile.mkdtemp(prefix="cayu_pr_reviewer_demo_"))
    origin, pr_head = _create_demo_origin(scratch)
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(id="c1", name="get_pr_diff", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="c2",
                    name="exec_command",
                    arguments={"kind": "process", "argv": ["python3", "--version"]},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="c3", name="exec_command", arguments={"kind": "shell", "shell": "rm -rf /"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="c4",
                    name="post_pr_comment",
                    arguments={"body": "Ran python3 --version as a smoke QA check; LGTM."},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Review posted."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    os.environ.pop("GITHUB_TOKEN", None)
    app, task_store = build_app(
        scratch / "data" / "cayu.db",
        scratch / "workspaces",
        provider=provider,
        model="scripted-model",
        with_credentials=False,
        pr_diff_tool=DemoPRDiffTool(pr_number=pr_number, head_sha=pr_head),
    )
    task = await enqueue_pr_review(
        task_store,
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        repo_url=str(origin),
        head_ref="fixture-pr",
        head_sha=pr_head,
        base_ref="main",
    )
    handled = await run_pr_review_worker_once(app, task_store, worker_id="demo-worker")
    assert handled == 1
    finished = await task_store.load_task(task.id)
    assert finished is not None
    print("task:", finished.id, finished.status, "| model requests:", len(provider.requests))
