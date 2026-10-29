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
                      query=TaskQuery(type="review_pr"), concurrency=4,
                      execution_timeout_s=600)
```

`execution_timeout_s` is the longest one handler call may take, grace period
included, counted from handler start, whether or not its runs are attached to
the task. It defaults to `None` (no limit). The last `execution_grace_s`
seconds of it are the grace period: 30 seconds by default, or half the limit
when that is shorter. An explicit grace that is not smaller than the limit is
refused with `ValueError`. With `execution_timeout_s=600`, from 570 seconds a
new top-level run in the call is refused with `ExecutionDeadlineExceeded` and
`cayu.deadlines.current_handler_time_limit().expired` is true, so a retry loop
can stop. Runs the handler builds without their own `execution_deadline` get
that 570-second mark as their default, so a hung tool or model stream ends the
run with `session.interrupted` (`limit_reached`) and an attached task fails
with `ExecutionDeadlineExceeded`. At 600 seconds the handler is cancelled and a
run still in flight ends `session.interrupted`. When the handler has stopped,
a task it did not settle fails with `ExecutionDeadlineExceeded` (`scope`
`task_handler`); the failure lists the IDs of the sessions the call started
and whether it had to be cancelled. A task with a `TaskRetryPolicy` records a
retryable failed attempt instead, a pending cancellation still settles as
cancelled, and a restartable task is not requeued. The `cayu.tasks.worker`
logger warns when the soft limit passes and again when it fails the task, with
the task ID, both settings and the session IDs. A handler that keeps
running after it is cancelled keeps its lease, and its task gets a durable
cancellation request with code `task_handler_timeout` that settles when the
handler returns. The limit also bounds the drain when the worker itself is
cancelled. Without `run_task_worker`, wrap a handler in `async with
HandlerTimeLimit(600, grace_s=30):` from `cayu.deadlines`; it raises
`ExecutionDeadlineExceeded` when it cancels the block.

A session keeps the deadline it started with, and records where it came from
(`Session.execution_deadline.origin`): `explicit` when the request set
`execution_deadline`, `run_default` when it took the handler's default, and
`enclosing` when it started inside an `execution_deadline_scope`. A run started
in the handler that pauses for an approval or a question returns to the handler,
and the handler can finish while the person decides. For a top-level session
whose deadline is `run_default`, the wait doesn't count: settling the approval
or answering the question gives the session back the time it had left when it
paused. Child sessions, and deadlines set explicitly or by an enclosing scope,
keep running while a person decides. The inbox and `GET /api/pending-actions`
show each pending item's deadline, its origin and whether the wait counts.

Give a run its own deadline when the handler's limit is the wrong bound for it,
and give a settle or a resume a new one when the stored deadline has passed:

```python
from cayu.approvals import ToolApprovalRequest
from cayu.deadlines import ExecutionDeadline, current_handler_time_limit

# A filing run started in a scan handler: it may wait a week for a person.
RunRequest(..., execution_deadline=ExecutionDeadline.after(7 * 24 * 3600))

# Make a person's wait count against the handler's limit after all.
RunRequest(..., execution_deadline=current_handler_time_limit().run_deadline())

# Continue a session whose own deadline passed. The new deadline is admitted
# like a new run's and replaces the stored one.
app.resolve_tool_approval(ToolApprovalRequest(..., execution_deadline=ExecutionDeadline.after(3600)))
app.resume(ResumeRequest(..., execution_deadline=ExecutionDeadline.after(3600)))
```

`UserInputResponse` takes `execution_deadline` the same way. A settle or resume
after a deadline passed, without a new one, fails with `ExecutionDeadlineExceeded`
(409 over HTTP), and its message names the deadline's expiry, origin and source.

The limit ends one handler call. An application that retries an event by
creating a new task for it (a sweeper, a reconciler, a redelivered webhook)
should pass the same `TaskCreate(retry_key=...)`, such as the event's ID, for
every task of that event. Tasks with one key share a `TaskRetryBudget`: by
default 3 tasks, with no time limit (`DEFAULT_TASK_RETRY_BUDGET`). The store counts each
new task with the key in the transaction that inserts it; restarts of one
task and duplicate task IDs do not count. The next task past the budget is not
created: `create_task` raises `TaskRetryBudgetExhausted`, whose `state` is
the key's durable `TaskRetryKeyState`, and the key stays `given_up` whatever
budget a later request carries. `TaskStore.load_task_retry_key(key)` and
`cayu inspect --retry-key KEY` show it. Pass
`retry_budget=TaskRetryBudget(max_attempts=..., max_elapsed_seconds=...)` to
choose the budget, or `TaskRetryBudget.unlimited()` to count without a cap; the
first admission fixes it. A budget with `max_elapsed_seconds` also ends running
work: the worker cancels a handler when the event's time runs out, counted from
its first admission, queue time included, even if `execution_timeout_s` is longer.
`run_task_worker` logs a warning the first time it claims a task of a type whose
budget is shorter than its `execution_timeout_s`. Keys are store-wide, so scoped
(tenant-bound) callers cannot set or read them, and keyed tasks cannot join
graphs or groups. A `TaskRetryPolicy` still works per task alongside a key.

`concurrency` defaults to one. `max_tasks`, when set, is a total for the whole
call, not a per-slot quota; recovered continuations and contracted-task parking
also count. One handler failing does not cancel its peers. Each active handler
retains its own lease, and one slot runs the shared recovery/reclaim scans.
Setting `stop` stops new claims and lets all active handlers finish. Cancelling
the worker applies the same cancellation-fence and quiescent-settlement rules to
every slot as to one worker; ordinary handlers may continue draining. An
infrastructure exception stops and drains all slots before propagating.

To serialize jobs for the same resource across workers, set a durable key when
creating the task:

```python
await task_store.create_task(TaskCreate(
    type="review_pr",
    concurrency_key="acme/app#42",
    input={"repository": "acme/app", "pull_request": 42},
))
```

Concurrency keys share a namespace across the task store. Choose them in trusted
host code; scoped task creation (including `/resources/tasks`) rejects keys
because a caller-chosen tenant prefix cannot authorize access to a shared key.
A keyed task always needs an exact worker or interrupted-continuation claim;
releasing its worker does not authorize direct `app.resume(...)`.


Only one task with that key can hold a live worker lease in the same task store.
Claims skip busy keys, leaving those jobs pending while unrelated keys and
unkeyed jobs run. Completion, failure, settled cancellation, explicit release,
and lease expiry free the key. A cancellation request keeps the key while its
owner drains. A claimed keyed task cannot be paused, blocked, or held for attention
until its handler stops and releases ownership. Interrupted continuations and
verified-work admissions use the same exclusion. A verified continuation that
finds its key busy stays waiting while the worker services unrelated jobs; it
retries after the key is released. Retries and restartable jobs retain their key.
Keyed tasks must start as queue jobs and attach through an exact worker claim.

Keys are case-sensitive public identifiers: 1–256 ASCII characters, starting
with a letter or digit and using only letters, digits, `._:/@#=+-` afterward.
Use namespaced keys when applications share a store. The key appears in task
inspection and dashboard details. This is lease-based mutual exclusion, with no
per-key ordering or rate guarantee. A worker that loses its lease must stop its
work; the key does not fence writes to an external business system.


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

If the process exits before that, the task stays fenced until something
reconciles it. A deploy that replaces the worker while a non-restartable task
runs past the shutdown grace does exactly this. An ordinary (non-retry) task
stays `claimed` with `status_reason == "cancellation_requested"`, and with
`status_payload["error"]["code"] == "task_worker_lease_expired_after_dispatch"`
when reclamation fenced it. No worker claims or reclaims it again. Retry-series
tasks use `retry_cancellation_requested` and require
`TaskStore.reconcile_task_retry_cancellation` with retry-specific evidence; the
helpers below do not settle them.

A fenced task looks like a live claim unless you check its lease, so Cayu gives
you ways to tell them apart:

- `is_fenced_after_worker_loss(task, now=None)` is true for a task held only by
  a cancellation fence whose worker's lease has expired.
- `TaskQuery(fenced_after_worker_loss=True)` lists only those tasks, and `False`
  leaves them out. An "is a run already queued or running?" check should use
  `TaskQuery(status=TaskStatus.CLAIMED, fenced_after_worker_loss=False)`, or the
  fenced task blocks new work until it is settled.
- Every `run_task_worker` that reclaims expired leases logs a warning once for
  each fenced task in its `query`'s scope, naming the task and how to settle it.
- `cayu inspect --fenced-tasks` lists them, and the Cayu Cloud activity contract
  counts them in `task_health`.

Once the application knows the original worker can no longer act and its handler
left nothing half-done, settle the task:

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

In a deployment with one worker, a task held by another worker id whose lease
expired minutes ago has no live owner, and that is enough evidence. Declare it
and the worker settles such tasks itself:

```python
await run_task_worker(
    app,
    app.task_store,
    handle_task,
    worker_id=f"jobs-{socket.gethostname()}-{os.getpid()}",
    query=TaskQuery(type="domain_job"),
    reconcile_abandoned_after_s=120,
    stop=stop,
)
```

At its reclaim cadence it settles every fenced task in its `query`'s scope that
another worker id holds, once it has seen the task fenced for at least 120
seconds, through `reconcile_abandoned_task_cancellation` with validator
`cayu.task-worker.single-worker`, and logs a warning for each. It leaves tasks
under its own worker id, session-attached tasks and retry series alone. `cayu
worker --reconcile-abandoned-after-seconds 120` sets the same option for a
project worker. Don't set it when several workers or processes claim the same
tasks: a worker whose heartbeat is slow can still be running a task whose lease
expired. The store decides that a lease expired by its own clock, and the
worker counts the 120 seconds on its own monotonic clock, so skew between the
two can't shorten the wait. `reconcile_abandoned_tasks(task_store,
live_worker_ids=..., validator_id=..., validator_version=...)` settles every
fenced task held by a worker not in `live_worker_ids` at once, for an
application that knows which workers are live some other way, for example from
its platform's record of stopped containers.

#### Restarting a restartable task after worker loss

Cancelling throws away whatever the handler finished. A task created with
`TaskCreate(restartable=True)` can run again instead, once the caller vouches
that the old worker is gone, with the same evidence the cancellation takes:

```python
await run_task_worker(
    app,
    app.task_store,
    handle_task,
    worker_id=f"jobs-{socket.gethostname()}-{os.getpid()}",
    query=TaskQuery(type="domain_job"),
    reconcile_abandoned_after_s=120,
    restart_abandoned=True,
    stop=stop,
)
```

`cayu worker --reconcile-abandoned-after-seconds 120 --restart-abandoned` does
the same for a project worker, and `reconcile_abandoned_tasks(...,
settlement="restart")` for an application that knows which workers are live.
`restart_abandoned_task(task_store, task_id, validator_id=...,
validator_version=..., evidence_id=...)` restarts one task, the way
`reconcile_abandoned_task_cancellation` cancels one.

The task goes back to `pending` and keeps its id and input. `restart_count` goes
up by one, `restart_history` gains a `TaskRestartRecord` (when, the previous
worker and lease, the validator and the evidence), and `status_payload["restart"]`
carries the same details, so the next handler can tell it is a restart and decide
what to reuse. Use new session ids on the new attempt, for example ones that
include `restart_count`.

Restarts after worker loss are capped: two by default, set with
`max_abandoned_restarts=` on the worker, `max_restarts=` on
`reconcile_abandoned_tasks` and `restart_abandoned_task`, or for one task with
`TaskCreate(max_restarts=...)`, which wins. After the last allowed restart, the
next loss cancels the task as before, with evidence id
`restart-limit-reached:<cap>:lease-expired-at:...`, so a handler that crashes
its worker every time doesn't loop forever. Requeues after a quiescent shutdown
(below) don't count toward the cap. A task whose cancellation someone asked
for (`cancel_task`, a handler time limit) before its worker was lost is
cancelled, not restarted. Tasks that aren't restartable, and session-attached
and retry-series tasks, are settled as they are without the option;
`abandoned_task_settlement(task)` says why.

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
behavior, and so does a restartable task that the worker is running as the
continuation of an interrupted session: shutdown cancels it rather than
requeueing it. Retry series, schedules, graph/group members, work contracts, and
pre-attached or child tasks do not accept this flag. A crash or SIGKILL cannot
prove handler quiescence and does not automatically restart a started job; once
you can vouch that the worker is gone, [restart it after worker
loss](#restarting-a-restartable-task-after-worker-loss).

Custom workers can use `requeue_restartable_task(task_store, task, worker_id)`
after stopping and fully joining the handler themselves. Keep the lease renewed
until that point and pass its current snapshot. This standalone helper neither
cancels work nor treats an expired lease as proof of quiescence.
