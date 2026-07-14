# GitHub PR reviewer

A cloud PR-review agent composed end-to-end on cayu: it is triggered by a pull
request, checks the queued PR head SHA out into a fresh review workspace, QAs the
change by running the project's tests, and posts one review comment back —
durably, on a worker pool. The bundled runnable version uses the trusted local
runner; swap in E2B or another sandbox runner for isolation.

This is the worked example behind [`docs/recipes/pr-reviewer.md`](../../docs/recipes/pr-reviewer.md).
Read that recipe for the narrative; this directory is the runnable code.

## Structure

The entrypoint stays small so the agent shape is visible:

| File | Purpose |
| --- | --- |
| `pr_reviewer.py` | Runnable CLI map for the demo/live review. |
| `reviewer_app.py` | Assembles the `CayuApp`, agent, provider, tools, and environment factory. |
| `github_tools.py` | GitHub REST tools and credential-proxy egress. |
| `qa_policy.py` | QA command allowlist for `exec_command`. |
| `workspace.py` | Per-review checkout and runner/workspace construction. |
| `worker.py` | Durable task enqueueing and `run_task_worker` handling. |
| `webhook.py` | GitHub webhook verification and task creation. |
| `demo.py` | Deterministic no-key fixture plus live-review runner. |

## Pipeline

```
GitHub webhook (pull_request)  ─▶ HMAC-verify ─▶ durable Task (SQLiteTaskStore)
        │                                                     │
        │                                          worker claims the Task
        ▼                                                     ▼
  build_webhook_app()                          fresh git checkout of the queued PR head SHA
                                                (PRReviewWorkspaceFactory)
                                                          │
                                                          ▼
                          get_pr_diff ▶ read changed code ▶ run QA commands
                          (restricted to a test/build allowlist) ▶ post_pr_comment
```

## Run it

```bash
# 1) Deterministic demo — no model key needed. Uses a local fixture repo and
#    drives the real runtime with a scripted provider. Shows the QA allowlist
#    denying a raw-shell attempt and the post path failing closed with no token.
PYTHONPATH=src python examples/github_pr_reviewer/pr_reviewer.py

# 2) Live review of a real PR — needs a provider key + a GitHub token.
OPENAI_API_KEY=sk-...  GITHUB_TOKEN=ghp-... \
  PYTHONPATH=src python examples/github_pr_reviewer/pr_reviewer.py --live owner/repo#123
```

Set `CAYU_MODEL` to override the model (defaults: `gpt-5.6-luna` for OpenAI,
`claude-sonnet-4-6` for Anthropic).

To run the webhook trigger instead of a one-shot review, serve
`build_webhook_app(task_store, webhook_secret=...)` with any ASGI server and run
a worker loop that calls `run_task_worker(...)`. The example's
`run_pr_review_worker_once(...)` helper uses that same durable
claim/heartbeat/reclaim path with `max_tasks=1`.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Provider key (auto-selected). |
| `GITHUB_TOKEN` | Resolved via the vault as the `github_token` secret; used to read private PRs and to post comments. |
| `CAYU_MODEL` | Optional model override. |

## Two ways to reach GitHub

cayu ships no GitHub-specific primitive, by design (MCP is an interoperability
layer, not the only tool model). You have two composable options:

1. **A custom `Tool`** through the credential proxy — used here
   (`GetPRDiffTool` / `PostPRCommentTool`). Full control, zero extra infra.
2. **The GitHub MCP server** via `McpServerSpec(url=..., secret_headers=...)` and
   `connect_mcp_toolset` — zero custom tool code. See the recipe and the README's
   "Streamable HTTP MCP" section.

## Making it genuinely cloud

Swap `LocalRunner`/`LocalWorkspace` in `PRReviewWorkspaceFactory` for
`E2BRunner`/`E2BWorkspace` (the `cayu[e2b]` extra) to run the checkout and QA
commands in an isolated cloud sandbox instead of the trusted local-dev runner,
and back the app with `PostgresSessionStore`/`PostgresTaskStore` for multi-worker
durability.
