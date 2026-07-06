from __future__ import annotations

import json

from examples.github_pr_reviewer.worker import enqueue_pr_review

from cayu import SQLiteTaskStore, verify_webhook_signature, webhook_task_id

try:
    from fastapi import FastAPI, HTTPException, Request

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover - only when fastapi is absent
    _HAS_FASTAPI = False


def build_webhook_app(task_store: SQLiteTaskStore, *, webhook_secret: str | None):
    """A tiny GitHub webhook receiver that enqueues review_pr tasks."""
    if not _HAS_FASTAPI:
        raise RuntimeError("pip install fastapi (or cayu[server]) to use the webhook app.")
    api = FastAPI(title="pr-reviewer-webhook")

    @api.post("/webhooks/github")
    async def github_webhook(request: Request) -> dict:
        raw = await request.body()
        if webhook_secret and not verify_webhook_signature(
            webhook_secret, raw, request.headers.get("X-Hub-Signature-256")
        ):
            raise HTTPException(status_code=401, detail="bad signature")
        if request.headers.get("X-GitHub-Event") != "pull_request":
            return {"ignored": True}
        payload = json.loads(raw)
        if payload.get("action") not in {"opened", "reopened", "synchronize"}:
            return {"ignored": True}
        pr = payload["pull_request"]
        delivery = request.headers.get("X-GitHub-Delivery")
        try:
            task = await enqueue_pr_review(
                task_store,
                owner=payload["repository"]["owner"]["login"],
                repo=payload["repository"]["name"],
                pr_number=pr["number"],
                repo_url=payload["repository"]["clone_url"],
                head_ref=pr["head"]["ref"],
                head_sha=pr["head"]["sha"],
                base_ref=pr["base"]["ref"],
                task_id=webhook_task_id("github", delivery) if delivery else None,
            )
        except ValueError:
            return {"deduplicated": True}
        return {"task_id": task.id}

    return api
