from __future__ import annotations

from cayu import (
    CayuApp,
    Message,
    RunLimits,
    RunRequest,
    SQLiteTaskStore,
    Task,
    TaskCreate,
    TaskQuery,
    run_task_worker,
)


async def enqueue_pr_review(
    task_store: SQLiteTaskStore,
    *,
    owner: str,
    repo: str,
    pr_number: int,
    repo_url: str,
    head_ref: str,
    head_sha: str,
    base_ref: str,
    task_id: str | None = None,
) -> Task:
    return await task_store.create_task(
        TaskCreate(
            task_id=task_id,
            type="review_pr",
            title=f"Review {owner}/{repo}#{pr_number}",
            assigned_agent_name="pr-reviewer",
            input={
                "owner": owner,
                "repo": repo,
                "pr_number": pr_number,
                "repo_url": repo_url,
                "head_ref": head_ref,
                "head_sha": head_sha,
                "base_ref": base_ref,
            },
        )
    )


async def _handle_pr_review_task(app: CayuApp, task: Task, worker_id: str) -> None:
    owner, repo, pr_number = task.input["owner"], task.input["repo"], task.input["pr_number"]
    head_sha = task.input["head_sha"]
    short_head_sha = str(head_sha)[:12]
    request = RunRequest(
        agent_name="pr-reviewer",
        session_id=f"pr-review-{owner}-{repo}-{pr_number}-{short_head_sha}",
        task_id=task.id,
        task_worker_id=worker_id,
        environment_name="pr-workspace",
        metadata={
            "repo_owner": owner,
            "repo_name": repo,
            "pr_number": pr_number,
            "repo_url": task.input["repo_url"],
            "head_ref": task.input["head_ref"],
            "head_sha": head_sha,
            "base_ref": task.input["base_ref"],
        },
        messages=[
            Message.text(
                "user",
                f"Review pull request #{pr_number} in {owner}/{repo} "
                f"({task.input['head_ref']}@{short_head_sha} -> {task.input['base_ref']}).",
            )
        ],
        limits=RunLimits(max_tool_calls=20, max_elapsed_seconds=600),
    )
    async for event in app.run(request):
        print(event.type, event.tool_name or "-", str(event.payload)[:160])


async def run_pr_review_worker_once(
    app: CayuApp, task_store: SQLiteTaskStore, *, worker_id: str = "worker-1"
) -> int:
    """Claim and handle one pending review_pr task through the generic task worker."""
    return await run_task_worker(
        app,
        task_store,
        _handle_pr_review_task,
        worker_id=worker_id,
        query=TaskQuery(type="review_pr", assigned_agent_name="pr-reviewer"),
        lease_seconds=900,
        max_tasks=1,
    )
