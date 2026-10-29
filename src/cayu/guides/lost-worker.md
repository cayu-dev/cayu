# Writing a task handler that survives a lost worker

A task handler that runs for an hour can lose its worker halfway through: a
deploy replaces the process, the machine runs out of memory, or the process
crashes. Cayu never hands a started task to a second worker, because the first
one may still be acting. Once the lease expires the task is fenced until the
application settles it. This page shows how to let a lost task run again and
how to make the rerun cheap: it should reuse every step that already finished
and repeat only the work that never did.

There are four parts. Each is optional and works on its own:

1. Mark the task restartable, and let the worker requeue it after the old worker
   is confirmed gone.
2. Give each agent session a deterministic id and run it with `run_once`, so a
   finished session returns its stored outcome without a model call.
3. Save the result of each step that isn't a session with `task_steps`.
4. Make external effects safe to repeat, because the worker can be lost after an
   effect and before Cayu records it.

## Restart the task after the worker is lost

Set `restartable=True` on an ordinary root queue job whose handler can start
again from the top:

```python
from cayu.tasks import TaskCreate

await app.create_task(
    TaskCreate(
        type="scout_run",
        restartable=True,
        max_restarts=2,
        input={"repository": "example"},
    )
)
```

A restartable task is already requeued when the worker shuts down cleanly and
the handler stops within the shutdown grace. A crash, an OOM kill or a handler
that runs past the grace leaves the task fenced instead. To requeue it, tell the
worker that it is the only worker, so a task held by another worker id whose
lease expired has no live owner:

```python
await run_task_worker(
    app,
    app.task_store,
    handle_task,
    # A fresh id per process: a restarted container can reuse its hostname and pid.
    worker_id=f"jobs-{socket.gethostname()}-{uuid.uuid4().hex[:12]}",
    reconcile_abandoned_after_s=120,
    restart_abandoned=True,
    stop=stop,
)
```

`cayu worker --reconcile-abandoned-after-seconds 120 --restart-abandoned` does
the same for a project worker. Restart is refused without the single-worker
declaration. Don't use this when several workers claim the
same tasks: a slow worker can still be running a task whose lease expired.
With several workers, settle fenced tasks yourself once you know which workers
are live, for example from your platform's record of stopped containers:

```python
from cayu.tasks import reconcile_abandoned_tasks

await reconcile_abandoned_tasks(
    task_store,
    live_worker_ids=live_worker_ids,
    validator_id="my-service.worker-registry",
    validator_version="1",
    settlement="restart",
)
```

`reconcile_abandoned_tasks` restarts each restartable fenced task held by a
worker outside `live_worker_ids` and cancels the others as before.
`restart_abandoned_task(task_store, task_id, validator_id=...,
validator_version=..., evidence_id=...)` restarts one task when you have
evidence for that task alone.

A restarted task goes back to `pending` with the same id and input,
`restart_count` one higher, a `TaskRestartRecord` (reason `WORKER_LOST`)
appended to `task.restart_history`, and `status_payload["restart"]` naming the
previous worker, the evidence and the previous session id. Restarts after a loss
are capped so a handler that kills its worker every time (an OOM, say) doesn't
loop forever. The cap is `TaskCreate(max_restarts=...)` when the task sets it,
otherwise the caller's `max_restarts` (or
`run_task_worker(max_abandoned_restarts=...)`, `cayu worker
--max-abandoned-restarts`), and 2 by default. After the last allowed restart the
next loss cancels the task with evidence `restart-limit-reached:...`, and its
restart history stays on the cancelled task (`restart_abandoned_task` raises
`TaskRestartLimitReached` instead). A clean shutdown requeue still raises
`restart_count` but doesn't count toward the cap. Tasks that aren't restartable,
session-attached tasks and retry series keep the cancellation settlement, and a
task someone asked to cancel is never restarted.
`cayu.tasks.abandoned.abandoned_task_settlement(task)` says why a task can't be
restarted.

The handler can tell a restart from a first run by `task.restart_count`. It
doesn't have to: with the next two parts, a rerun from the top skips the work
that finished.

## Reuse finished sessions with `run_once`

Name each session after the piece of work it does, so a rerun computes the same
id, and run it with `run_once`:

```python
from cayu import Message, RunRequest
from cayu.context import StructuredOutputSpec
from cayu.sessions import RunOnceSessionNotFinished, run_once

VERIFICATION = StructuredOutputSpec(
    name="verification",
    json_schema={
        "type": "object",
        "properties": {"confirmed": {"type": "boolean"}},
        "required": ["confirmed"],
        "additionalProperties": False,
    },
)


async def verify(app, task, finding_id):
    for attempt in (1, 2, 3):
        request = RunRequest(
            agent_name="verifier",
            session_id=f"{task.id}-{finding_id}-verify-{attempt}",
            messages=[Message.text("user", f"Verify finding {finding_id}.")],
            structured_output=VERIFICATION,
        )
        try:
            outcome = await run_once(app, request)  # A finished attempt: no model call.
        except RunOnceSessionNotFinished:
            continue  # Lost with its worker and not settled yet.
        if outcome.ok:
            return outcome
        if outcome.resumable:
            # Paused for an approval or an answer: not lost, so keep this id.
            raise RuntimeError(f"{request.session_id} is waiting on a person.")
        # Failed, or a lost attempt that session recovery settled later:
        # move on to the next id rather than stopping here.
    raise RuntimeError(f"{finding_id}: no verification attempt completed.")
```

When no session has that id, `run_once` runs it. When a finished session has it,
`run_once` returns that session's `RunOutcome` from the store (status, final
text, structured output, usage and error) and calls no model. It raises
`RunOnceMismatch` if the stored session used another agent or structured-output
spec, and `RunOnceSessionNotFinished` if the session is still pending or
running. That session belonged to the lost worker, so start a new attempt under
a new id and leave the old one to the session recovery APIs. Once recovery
settles it, the old id returns a failed or interrupted outcome, so the loop
keeps going until it finds an attempt that completed. This recipe is for
agents that don't pause for approvals or user input. A session paused that way
isn't lost: `outcome.resumable` is true, and it has to be resumed under the same
id once the person answers (`cayu guide long-running`), not replaced by a new
attempt. Messages are not
compared: give each distinct piece of work its own id. Don't reuse the task's
`restart_count` in these ids, or a restart would rerun every session.

## Save other steps with `task_steps`

Not every step is a session: a git checkout, a file written to a volume, a
proposal filed with another service. Save their results in a step store keyed
by the task id and a step name:

```python
from cayu.storage import SQLiteTaskStepStore
from cayu.tasks import task_steps

step_store = SQLiteTaskStepStore("app.db")  # the task store's database


async def handle_task(app, task, worker_id):
    steps = task_steps(step_store, task)
    checkout = await steps.run("checkout", lambda: clone(task.input["repository"]))
    findings = await steps.run("rank", lambda: rank(checkout))
    ...
```

`steps.run(name, fn)` returns the saved result when the step has one, and
otherwise calls `fn` (sync or async), saves what it returns and returns it. The
first saved result for a step wins. Results are JSON values of at most 256 KiB;
save a reference such as an artifact id or a commit sha for anything bigger.
`PostgresTaskStepStore` does the same on PostgreSQL, and
`InMemoryTaskStepStore` is for tests. `steps.forget(name)` makes the next run
repeat one step, and `steps.clear()` drops a task's saved steps. Clear them
only after the task is complete, so a loss during cleanup doesn't throw away
the progress a restart would reuse.

Save a step's result as soon as the step finishes, not at the end of the
handler. A handler that keeps results in memory until it returns loses all of
them with its worker.

## Make external effects safe to repeat

A saved step or a finished session proves that the step finished. It can't
prove that a step which was running when the worker died did nothing: the
worker can be lost after the effect and before the save. So any step that
changes another system has to tolerate running twice:

- Pass a stable idempotency key to the other system, built from the task id and
  the step name (`f"{task.id}:propose:{finding_id}"`), not from the attempt.
- Or check first: look up the commit, pull request or ticket you would create
  and reuse it if it exists.
- Inside agent sessions, declare tool effects and use
  `ToolContext.idempotency_key`; Cayu's effect settlement decides whether an
  interrupted tool call in that session may run again. The key includes the
  session id, so it doesn't deduplicate a new attempt under a new session id.
  A tool that changes another system also needs a key scoped to the task and
  operation, or a lookup that finds the earlier result, before another attempt
  repeats it. See `cayu guide tool-effects` and `cayu guide durable-operations`.

## Putting it together

```python
async def handle_task(app, task, worker_id):
    steps = task_steps(step_store, task)
    checkout = await steps.run("checkout", lambda: clone(task.input["repository"]))
    findings = await steps.run("rank", lambda: rank(checkout))
    proposed = []
    for finding in findings:
        outcome = await verify(app, task, finding["id"])  # completed, via run_once
        if outcome.structured_output.output["confirmed"]:
            key = f"{task.id}:propose:{finding['id']}"
            proposed.append(
                await steps.run(
                    f"propose:{finding['id']}",
                    lambda: propose(finding, idempotency_key=key),
                )
            )
    await complete_managed_task(app.task_store, task, worker_id, {"proposed": proposed})
    await steps.clear()  # only after the task is complete
```

The task is created with `restartable=True` and the worker runs with
`restart_abandoned=True`. If the worker dies after three verifications, the
next worker claims the same task, the checkout and ranking come back from the
step store, the three finished verifications come back from the session store,
and only the rest calls the model.

Test it the way it will fail: `cayu guide restart-tests` shows how to stop an
application and rebuild it on the same stores.
