# Triggering runs: which start verb do I use?

Cayu has several ways to start (or continue) an agent run. They are not
interchangeable — each fits a different trigger. Pick by answering *what kicks
this off?*

Applications can call these entry points directly or enqueue Cayu tasks; an
external workflow engine is optional.

When trusted in-process application code has already authenticated a root
caller, it can retain that identity across the complete delegated session tree:

```python
RunRequest(
    agent_name="assistant",
    messages=[Message.text("user", prompt)],
    invocation_origin=InvocationOriginClaim(
        subject=authenticated_user_id,
        tenant=authenticated_tenant_id,
    ),
)
```

This is a host assertion, not authentication performed by Cayu. The protected
HTTP `/api/run` boundary instead stamps its verified `AuthContext` itself, and
open or unattributed SDK calls remain explicitly unattributed. Child sessions
inherit both the root origin and Cayu-minted `root_invocation_id`
automatically; applications must not copy either into child requests. That
invocation ID remains unique even if a deleted root's session ID is reused.

Task-only roots use the same bounded origin contract. Trusted application code
may add `invocation_origin=InvocationOriginClaim(...)` to `TaskCreate`, which is
recorded as `host_asserted`, and classify a webhook or scheduled root without
putting that authority in task metadata:

```python
request = task_create_with_execution_source(
    TaskCreate(
        type="nightly-report",
        invocation_origin=InvocationOriginClaim(subject="scheduler:nightly"),
        available_at=next_run,
    ),
    source=TaskExecutionSource.SCHEDULED,
)
task = await app.create_task(request)
```

An unattributed `TaskCreate` remains valid. Parent tasks, durable dispatch, and
task-backed sessions inherit the immutable root origin automatically.

| Your trigger | Use | Notes |
| --- | --- | --- |
| A synchronous request you can await in-process | `app.run(RunRequest(...))` | The base case. Returns an async event stream. No durability beyond the session stores. |
| Durable follow-up work on an **existing** session (retry, continue, deferred step) | `app.dispatch(DispatchRequest(...))` + a worker running `TaskStoreDispatcher.run_worker(app, ...)` | The dispatcher **resumes** existing sessions from durable dispatch requests. |
| Durable **new** work pulled from a queue (e.g. "review this PR") | `run_task_worker(app, task_store, handler, ...)` | Claims arbitrary `Task`s and starts a **fresh** session per task. This is the PR-reviewer shape. |
| The model itself wants to delegate a sub-task | `SubagentTool` | Model-facing. Creates a child session with `parent_session_id`; foreground, in-process background, or task-backed durable. |
| React to Cayu's **own** durable events (budget alerts, session completion) | `EventWatcher` | Trusted app code that pulls the durable event log. **Not** an external-webhook receiver. |
| Continue one specific session by id | `ResumeRequest` / `ForkSessionRequest` / `InterruptSessionRequest` | Resume appends messages; fork branches without mutating the source; interrupt stops a pending/running session. |
| Run and evaluate one bounded in-process candidate population | `await app.run_fork_group(ForkGroupRequest(...))` | Freezes one source, runs 2-16 profiled sibling sessions under one causal budget, applies deterministic app gates, and produces one tool-free bounded judgment. Exact retries reconstruct; any failed sibling fails version 1. |

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

A project can expose either loop as a thin named entrypoint and let
`cayu worker <name>` own project discovery, one-time application construction,
signals, and bounded cooperative shutdown. The CLI does not choose between the
two loops or add recovery policy; see
[project workers](project-workers.md).

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

`verify_webhook_signature` returns `False` for malformed untrusted header values,
including non-ASCII or invalid hexadecimal signatures. It raises
`WebhookSignatureError` only for invalid verifier configuration or caller inputs,
such as an unsupported algorithm, invalid secret, or non-bytes request body.
