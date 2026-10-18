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
| A customer message in a new or existing conversation, with one result to await | `send_turn(app, TurnRequest(...))` | Returns `RunOutcome`. An absent ID creates a session; a supplied ID selects creation or resume once. Live executions conflict; abandoned owners use native recovery. Admission exceptions propagate; approvals and ambiguous effects keep their dedicated APIs. |
| A synchronous request you can await in-process | `app.run(RunRequest(...))` | The base case. Returns an async event stream. No durability beyond the session stores. |
| Durable follow-up work on an **existing** session (retry, continue, deferred step) | `app.dispatch(DispatchRequest(...))` + a worker running `TaskStoreDispatcher.run_worker(app, ...)` | The dispatcher **resumes** existing sessions from durable dispatch requests. |
| Durable **new** work pulled from a queue (e.g. "review this PR") | `run_task_worker(app, task_store, handler, ...)` | Claims arbitrary `Task`s and starts a **fresh** session per task. This is the PR-reviewer shape. |
| The model itself wants to delegate a sub-task | `SubagentTool` | Model-facing. Creates a child session with `parent_session_id`; foreground, in-process background, or task-backed durable. |
| React to Cayu's **own** durable events (budget alerts, session completion) | `EventWatcher` | Trusted app code that pulls the durable event log. **Not** an external-webhook receiver. |
| Continue one specific session by id | `ResumeRequest` / `ForkSessionRequest` / `InterruptSessionRequest` | Resume appends messages; fork branches without mutating the source; interrupt stops a pending/running session. |
| Start independent durable work from one exact session checkpoint | `snapshot_fork_source(...)` + `fork_session(ForkSessionRequest(...))` + `dispatch(DispatchRequest(...))` | Bind the snapshot, exact first invocation, and one matching `initial_dispatch_id` to each child, then let ordinary workers run them. Application code consumes results independently and owns any join or evaluation policy. |

For a product response, select the fields its customer needs rather than
serializing every event or tool result:

```python
outcome = await send_turn(app, TurnRequest(
    agent_name="support", session_id=conversation_id,
    messages=[Message.text("user", customer_message)],
))
response = {
    "conversation_id": outcome.session_id,
    "status": outcome.status.value,
    "answer": outcome.final_text,
}
```

`outcome.summary()` supplies bounded presentation data when a richer view is
useful. Applications still decide which results and errors a customer may see.
See [terminal-turn convenience](runtime-contracts.md#terminal-turn-convenience)
for interruption, concurrency and opt-in payload rules.

For a coordinator model that must notice independently completing children,
register `ChildSessionContextContributor` on that parent agent and expose
`ChildSessionResultTool`. Cayu recomputes a bounded lifecycle projection before
each model request, marks a terminal occurrence consumed only when that request
crosses the final provider-start transition after optional provider-backed token
counting, and retrieves output separately through the parent-authorized
reference. It does not wait for siblings or recreate a group abstraction.
Application code should still use session queries or event watchers for
control-plane orchestration; the projection is model context, not a durable
workflow join or a user-steering channel.

## The two worker loops

Both claim durable work with leases, but they cover different shapes:

- **`TaskStoreDispatcher.run_worker`** — resumes *existing* sessions from
  `DispatchRequest`-shaped tasks. Use it for durable retries / continuations of
  runs that already started. See `examples/dispatch_worker.py`.
- **`run_task_worker(app, task_store, handler, *, worker_id, query=..., ...)`** —
  claims *arbitrary* `Task`s and starts a *new* session for each. The handler
  turns a claimed task into an `app.run(RunRequest(task_id=task.id,
  task_worker_id=worker_id, task_lease_expires_at=task.lease_expires_at, ...))`.
  Use it when an external event (a webhook, a
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
        task_lease_expires_at=task.lease_expires_at,
        messages=[Message.text("user", task.input["prompt"])],
    )):
        if _event.type == EventType.SESSION_INTERRUPTED:
            outcome = TaskHandlerOutcome.SESSION_INTERRUPTED
    return outcome

# Four independently heartbeated tasks in one process. More processes can
# share the store; its atomic claims keep them from taking the same task.
await run_task_worker(app, task_store, handle, worker_id="worker-1",
                      query=TaskQuery(type="review_pr"), concurrency=4)
```

`concurrency` defaults to one. `max_tasks`, when set, is a total for the whole
call, not a per-slot quota; recovered continuations and contracted-task parking
also count. One handler failing does not cancel its peers. Each active handler
retains its own lease, and one slot runs the shared recovery/reclaim scans.
Setting `stop` stops new claims and lets all active handlers finish. Cancelling
the worker applies the same cancellation-fence and quiescent-settlement rules to
every slot as to one worker; ordinary handlers may continue draining. An
infrastructure exception stops and drains all slots before propagating.

The loop owns claim → heartbeat → handle → reclaim-expired-leases, and keeps
going if one task's handler raises or returns without terminalizing its task (it
marks that task failed). Pass a `stop: asyncio.Event` for graceful shutdown and
`max_tasks=N` to bound it. `stop` ends claiming; handlers that are already
running finish.

The handler passes the claim snapshot it received. The worker loop privately
serializes later heartbeat generations with session attachment and terminal
mutation, so the handler must not reload or guess a replacement lease while
`run_task_worker(...)` is supervising it. If the handler itself completes or
fails the task, use `complete_managed_task(task_store, task, worker_id, result)`
or `fail_managed_task(task_store, task, worker_id, error)` with the original
claimed `task`; those boundaries present the latest acknowledged private lease
to the store without allowing a same-ID successor claim to be mistaken for the
original owner.

The explicit `SESSION_INTERRUPTED` outcome is the exception to terminal handler
completion. The helper verifies that the task is attached and its durable
session state is actually `interrupted`, then uses an exact durable receipt to
clear only the worker identity and lease. Transient and ambiguous publication
failures are retried and reconciled by receipt readback. If those bounded
attempts are exhausted, the task remains `running`, attached, and eligible for
expired-owner handoff recovery instead of being rewritten as failed. It stays
ineligible for fresh-task claim/reclaim while an approval, user-input response,
operator resume, or recovery process continues the session. Returning `None`
preserves the original terminal-or-fail behavior; do not return the handoff
outcome merely to abandon unfinished work.

### When a worker dies mid-task

Reclaiming expired leases returns a task to `pending` only when its handler never
started. A started task can have done part of its work, so Cayu never gives it to
another worker: reclamation fences it as a cancellation request instead. Cancelling
the `run_task_worker(...)` coroutine records the same fence and then waits for the
running handler to finish; it does not interrupt the handler. The worker settles
the fence once the handler returns.

If the process exits before that, for example because a service manager stopped
it while a long handler was running, an ordinary (non-retry) task stays `claimed`
with `status_reason == "cancellation_requested"`. No worker claims or reclaims it
again. Retry-series tasks use `retry_cancellation_requested` and require
`TaskStore.reconcile_task_retry_cancellation` with retry-specific evidence; the
helper below does not settle them.
Once the application knows the original worker can no longer act and its handler
left nothing half-done, settle it:

```python
from cayu.tasks import reconcile_abandoned_task_cancellation

await reconcile_abandoned_task_cancellation(
    task_store,
    task.id,
    validator_id="my-service.restart-recovery",
    validator_version="1",
    evidence_id=f"process-exit:{previous_pid}",
)
```

The task becomes `cancelled` with that evidence recorded; create a new task if the
work should run again. The store refuses while the original lease is live, but
lease expiry does not prove that the handler stopped: a cancellation-fenced handler
may still be draining after expiry. The application must independently establish
quiescence before calling this helper. This helper does not support session-attached
tasks. Interrupted-handoff recovery covers attached tasks without pending
cancellation; cancellation-fenced attached tasks require separate reconciliation.

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

### Restartable queue jobs

Set `TaskCreate(restartable=True, ...)` only for an ordinary root queue job that
can safely repeat from its beginning. External writes need application-level
idempotency keys or another way to tolerate repetition. The handler must join
all subprocesses and other work during cancellation; returning while detached
work can still act is not a safe restart boundary.

```python
await app.create_task(TaskCreate(type="index_repository", restartable=True,
                               input={"repository": "example"}))
```

When its worker coroutine is cancelled during shutdown, `run_task_worker`
cancels the handler, keeps renewing its lease while the handler cleans up, and
waits for it to finish before atomically returning the task to `pending`. The
next worker claims the same task ID and input with an incremented
`task.restart_count`. Its previous session remains available for inspection and is durably fenced against resuming;
the next handler must create a fresh session, using the default `RunRequest`
session ID or an ID that includes `restart_count`. Restartable tasks attach only
through an exact worker claim, so an old session cannot start them without one.
Connect SIGTERM to cancellation of the worker coroutine in your service.
Setting the worker's `stop` event continues to drain active handlers normally.

An operator cancellation always wins over shutdown requeueing. Lost ownership
also prevents a restart. Non-restartable tasks keep their existing cancellation
behavior. Retry series, schedules, graph/group members, work contracts, and
pre-attached or child tasks do not accept this flag. A crash or SIGKILL cannot
prove handler quiescence and does not automatically restart a started job.

Custom workers can use `requeue_restartable_task(task_store, task, worker_id)`
after stopping and fully joining the handler themselves. Keep the lease renewed
until that point and pass its current snapshot. This standalone helper neither
cancels work nor treats an expired lease as proof of quiescence.
