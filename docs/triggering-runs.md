# Triggering runs: which start verb do I use?

Cayu has several ways to start (or continue) an agent run. They are not
interchangeable — each fits a different trigger. Pick by answering *what kicks
this off?*

| Your trigger | Use | Notes |
| --- | --- | --- |
| A synchronous request you can await in-process | `app.run(RunRequest(...))` | The base case. Returns an async event stream. No durability beyond the session stores. |
| Durable follow-up work on an **existing** session (retry, continue, deferred step) | `app.dispatch(DispatchRequest(...))` + a worker running `TaskStoreDispatcher.run_worker(app, ...)` | The dispatcher **resumes** existing sessions from durable dispatch requests. |
| Durable **new** work pulled from a queue (e.g. "review this PR") | `run_task_worker(app, task_store, handler, ...)` | Claims arbitrary `Task`s and starts a **fresh** session per task. This is the PR-reviewer shape. |
| The model itself wants to delegate a sub-task | `SubagentTool` | Model-facing. Creates a child session with `parent_session_id`; foreground or background. |
| React to Cayu's **own** durable events (budget alerts, session completion) | `EventWatcher` | Trusted app code that pulls the durable event log. **Not** an external-webhook receiver. |
| Continue one specific session by id | `ResumeRequest` / `ForkSessionRequest` / `InterruptSessionRequest` | Resume appends messages; fork branches without mutating the source; interrupt stops a pending/running session. |

## The two worker loops

Both claim durable work with leases, but they cover different shapes:

- **`TaskStoreDispatcher.run_worker`** — resumes *existing* sessions from
  `DispatchRequest`-shaped tasks. Use it for durable retries / continuations of
  runs that already started. See `examples/dispatch_worker.py`.
- **`run_task_worker(app, task_store, handler, *, worker_id, query=..., ...)`** —
  claims *arbitrary* `Task`s and starts a *new* session for each. The handler
  turns a claimed task into an `app.run(RunRequest(task_id=task.id,
  task_worker_id=worker_id, ...))`. Use it when an external event (a webhook, a
  cron tick) enqueues a job. This is what the
  [PR-reviewer recipe](recipes/pr-reviewer.md) uses.

Minimal `run_task_worker` usage:

```python
async def handle(app, task, worker_id):
    outcome = None
    async for _event in app.run(RunRequest(
        agent_name=task.assigned_agent_name or "assistant",
        session_id=f"job-{task.id}",
        task_id=task.id,
        task_worker_id=worker_id,
        messages=[Message.text("user", task.input["prompt"])],
    )):
        if _event.type == EventType.SESSION_INTERRUPTED:
            outcome = TaskHandlerOutcome.SESSION_INTERRUPTED
    return outcome

# Run N of these across processes; the task store's lease + FOR UPDATE SKIP LOCKED
# claiming (Postgres) keeps workers from colliding.
await run_task_worker(app, task_store, handle, worker_id="worker-1",
                      query=TaskQuery(type="review_pr"))
```

The loop owns claim → heartbeat → handle → reclaim-expired-leases, and keeps
going if one task's handler raises or returns without terminalizing its task (it
marks that task failed). Pass a `stop: asyncio.Event` for graceful shutdown and
`max_tasks=N` to bound it.

The explicit `SESSION_INTERRUPTED` outcome is the exception to terminal handler
completion. The helper verifies that the task is attached and its durable
session state is actually `interrupted`, then clears only the worker identity
and lease. The task stays `running`, attached, and ineligible for fresh-task
claim/reclaim while an approval, user-input response, operator resume, or
recovery process continues the session. Returning `None` preserves the original
terminal-or-fail behavior; do not return the handoff outcome merely to abandon
unfinished work.

## What does *not* trigger a run

`EventWatcher` watches Cayu's own event log — it does not receive external
webhooks. To be triggered by an outside system (a GitHub `pull_request` event, a
Stripe hook), your app terminates the HTTP request and enqueues a `Task`, then a
`run_task_worker` loop picks it up. `cayu.webhooks.verify_webhook_signature` and
`webhook_task_id` cover the verify-and-enqueue-idempotently step. The
[PR-reviewer recipe](recipes/pr-reviewer.md) shows that end to end.
