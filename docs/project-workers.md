# Project worker processes

`cayu worker` boots the application factory declared in the nearest Cayu
project. The command follows the
[application construction contract](../src/cayu/guides/application-anatomy.md):
one synchronous, zero-argument factory call creates one process-scoped
`CayuApp`. Durable stores coordinate separate processes.

## Run a named worker

Projects can configure several worker entrypoints without changing the default
application factory:

```toml
[tool.cayu]
factory = "app:build_app"

[tool.cayu.workers]
dispatch = "workers:run_dispatch"
fresh = "workers:run_fresh_tasks"
```

Run exactly one:

```bash
cayu worker dispatch
cayu worker fresh --shutdown-grace-seconds 45
```

Every target has one contract: an `async def` callable with two required
positional parameters, `(app, stop)`. `app` is the freshly constructed
`CayuApp`; `stop` is an `asyncio.Event` set by SIGINT or SIGTERM. Invalid
signatures and synchronous targets fail before application construction or a
long-running loop begins. An entrypoint may also set the event when it decides
to finish its own coordination; normal completion without a process signal
still exits `0`.

Entrypoints choose the worker shape and remain thin. A
`TaskStoreDispatcher.run_worker` entrypoint resumes existing sessions through
the configured dispatcher:

```python
import os
import socket

from cayu import CayuApp
from cayu.tasks import TaskStoreDispatcher


def worker_id(role: str) -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{role}"


async def run_dispatch(app: CayuApp, stop) -> None:
    if not isinstance(app.dispatcher, TaskStoreDispatcher):
        raise RuntimeError(
            "dispatch worker requires app.dispatcher to be a TaskStoreDispatcher"
        )
    await app.dispatcher.run_worker(
        app,
        worker_id=worker_id("dispatch"),
        stop=stop,
    )
```

A fresh-task worker calls the complementary public loop. Its handler owns the
domain-specific `RunRequest` and terminal task behavior:

```python
from cayu import CayuApp
from cayu.tasks import Task, TaskQuery, complete_managed_task, run_task_worker


async def handle_task(app: CayuApp, task: Task, claimed_by: str) -> None:
    # Consume app.run(RunRequest(task_id=task.id,
    #                            task_worker_id=claimed_by,
    #                            task_lease_expires_at=task.lease_expires_at,
    #                            ...)) here.
    # If this handler owns terminalization instead of app.run(), use the
    # managed boundary so heartbeat renewal cannot race a stale lease snapshot:
    # await complete_managed_task(app.task_store, task, claimed_by, {"ok": True})
    ...


async def run_fresh_tasks(app: CayuApp, stop) -> None:
    if app.task_store is None:
        raise RuntimeError("fresh worker requires app.task_store")
    await run_task_worker(
        app,
        app.task_store,
        handle_task,
        worker_id=worker_id("fresh"),
        query=TaskQuery(type="domain_job"),
        # Bounds each handler call; a hung or endlessly retrying job fails.
        execution_timeout_s=600,
        stop=stop,
    )
```

`app.task_store` and `app.knowledge_store` are read-only properties holding the
stores the factory built the app with, so a worker target needs only the app.

A handler that returns without settling its task fails it with "Task handler
returned without completing or failing the task." A run bound to the task, as
in the comment above, settles it when its session ends. A handler that runs several sessions,
or does its work without one, settles the task itself:

```python
from cayu import run_to_completion
from cayu.tasks import complete_managed_task, fail_managed_task


async def handle_hunt(app: CayuApp, task: Task, claimed_by: str) -> None:
    outcomes = [await run_to_completion(app, request) for request in hunt_requests(task)]
    if all(outcome.ok for outcome in outcomes):
        sessions = [outcome.session_id for outcome in outcomes]
        await complete_managed_task(app.task_store, task, claimed_by, {"sessions": sessions})
    else:
        await fail_managed_task(app.task_store, task, claimed_by, {"error": "a hunt failed"})
```

The CLI does not infer a handler, query, task store, abandoned-session
boundary, or recovery policy. Configure those in the factory or named
entrypoint. In particular, `cayu worker` performs no implicit incomplete-session
recovery.

## A deploy during a long task

Stopping a worker while a non-restartable task is running leaves that task
fenced. `cayu worker` waits `--shutdown-grace-seconds` (30 by default, the
same as ECS and Kubernetes) for the handler to return. When the grace runs out
first, or the process is killed, the task stays `claimed` with
`status_reason` `cancellation_requested` and no worker runs it again until it
is reconciled, because Cayu can't prove the old handler stopped. The next
worker that looks for expired claims logs a warning naming the task, and
`cayu inspect --fenced-tasks` lists every such task. Restartable tasks are
requeued only when their handler stops within the grace. On Cayu Cloud,
`[worker] stop_grace_seconds` gives the worker up to 120 seconds and raises
`--shutdown-grace-seconds` to match
([how the worker stops](cayu-cloud.md#how-the-worker-stops)), and
`cayu cloud service status` reports fenced tasks, and `cayu cloud deploy` lists the
tasks the current workers are running before it uploads
([how](cayu-cloud.md#how-the-worker-stops)).

On Cayu Cloud a worker can avoid the stop altogether. When its Release declares
`[worker] drain_seconds` and the Agent has its provisioned database, a deploy
keeps the old worker running beside the new one for up to that long. `cayu
worker` asks Cloud every 30 seconds whether its Release is draining (through
`CAYU_WORKER_DRAIN_URL`, with `CAYU_ATTENTION_HANDOFF_TOKEN` and
`CAYU_RELEASE_ID`). Once it is, the worker sets the same `stop` event a SIGTERM
sets, without the shutdown grace: `run_task_worker` claims nothing new, the
tasks it holds finish, and the process exits with status 0. The new worker
takes every new task meanwhile, and Cloud stops whatever is still running at
the deadline. Outside Cayu Cloud, or with a control plane that doesn't answer
the question, nothing is asked. A worker run by another command can ask the same
endpoint itself.

A worker that `[tool.cayu.evals].worker` names also runs the project's declared
eval targets' runs, and drains them the same way: it finishes the run it holds
and takes no new one ([Evals](evals.md)).

When this is the only worker process for its tasks, say so and it settles them
itself:

```bash
cayu worker dispatch --reconcile-abandoned-after-seconds 120
```

A task fenced under another worker id is then settled as cancelled once the
worker has seen it fenced for 120 seconds, with the declaration recorded as the
validator; create a new task if the work should run again. The flag sets
`CAYU_WORKER_RECONCILE_ABANDONED_AFTER_S`, which `run_task_worker` reads when the
entrypoint doesn't pass `reconcile_abandoned_after_s` itself. It is off by
default and refused with `--processes`, because with several workers an expired
lease can belong to one that is slow rather than gone. Give each process its
own worker ID (as in the example above), so a restarted worker doesn't share
the dead one's ID. Otherwise settle fenced tasks yourself once you know the old
process exited; see [when a worker dies mid-task](triggering-runs.md#when-a-worker-dies-mid-task).

To run a task created with `TaskCreate(restartable=True)` again instead of
cancelling it, add `--restart-abandoned`:

```bash
cayu worker dispatch --reconcile-abandoned-after-seconds 120 --restart-abandoned
```

The task goes back to `pending` with `restart_count` one higher and a record in
`restart_history`, and the next handler sees `task.status_payload["restart"]`.
After two restarts after worker loss (`--max-abandoned-restarts N`, or the task's
own `max_restarts`), the next loss cancels it as before, so a handler that
crashes its worker every time doesn't loop. Tasks that aren't restartable are
still cancelled. The flags set `CAYU_WORKER_RESTART_ABANDONED` and
`CAYU_WORKER_MAX_ABANDONED_RESTARTS`, which `run_task_worker` reads when the
entrypoint doesn't pass `restart_abandoned` and `max_abandoned_restarts` itself.

## Inspect tasks

`cayu task` reads tasks without starting the application. It opens the project's store
the way `cayu session` does (`--sqlite PATH`, `--postgres DSN`, `CAYU_DATABASE_URL`, then
the project's configured or canonical store), read-only, or reads a Cayu Cloud Agent with
`--application NAME` ([how](cayu-cloud.md#reading-an-agents-state-from-the-cli)):

```bash
cayu task list --status running
cayu task list --application my-agent --type scan --limit 20
cayu task show TASK_ID --json
```

`list` prints the newest-updated tasks first with their status, the reason for it, the
assigned agent, the retry attempt and the worker holding the claim; add `--offset` for
the next page. Like `GET /api/tasks`, its rows leave out each task's input and result.
`show` adds the input, result or error, a `cancellation` summary (whether cancelling was
asked for or done, the reason, and for a task cancelled after its worker was lost, the
reconciliation outcome such as `quiescent`), and every attempt of the task's retry series
found through its predecessor and successor links. Table output is the default;
`--json` (and `--jsonl` for `list`) prints the same data with a `schema_version`.

## Run due tasks and exit

A host that starts the worker only when work is due (cron, a function, a
platform that wakes the application on a timer) runs it with `--until-idle`:

```bash
cayu worker fresh --until-idle
```

The entrypoint learns about the flag from `worker_until_idle_requested()` and
returns the result of `run_task_worker_until_idle`, which takes the same
arguments as `run_task_worker`:

```python
from cayu.tasks import run_task_worker, run_task_worker_until_idle, worker_until_idle_requested


async def run_fresh_tasks(app: CayuApp, stop):
    run = run_task_worker_until_idle if worker_until_idle_requested() else run_task_worker
    return await run(
        app,
        app.task_store,
        handle_task,
        worker_id=worker_id("fresh"),
        query=TaskQuery(type="domain_job"),
        stop=stop,
    )
```

The store must report scheduled task times (`supports_task_scheduling`; every
built-in store does); otherwise the call fails before claiming anything.

The worker reclaims every expired claim, finishes a whole pass of expired
interrupted-handoff recovery and, with a `recovered_interrupted_task_handler`,
one continuation scan, scanning again if recovery hands off more sessions. It
then handles tasks as usual and returns once a claim finds nothing and no
handler is running; a handler may create a task that is due at once, so with
`concurrency` above one, slots wait for each other. `max_tasks` still caps the
run. The command closes the app, then prints one JSON line and exits `0`; read
the last line of stdout:

```json
{"handled":2,"idle":true,"next_available_at":"2027-01-15T12:00:00Z","next_expiry_at":null,"maintenance_required":false,"continuations_pending":false}
```

Start the worker again at `next_available_at` (a scheduled task becomes
claimable) or `next_expiry_at` (a scheduled task stops being admissible and a
worker records it as expired), whichever is earlier. Start it again at once when
`idle` is false (`max_tasks` or a stop ended the run, so work may still be due),
`maintenance_required` is true, or `continuations_pending` is true. Both times
are `null` when nothing is scheduled. They come from the store just before the
claim that found nothing, so a task that becomes due while the worker exits is
never missed. They cover only that worker's `query`: with several worker names,
start each at its own time, or one host timer at the earliest.

The continuation scan runs once per start, in one slot, so a handed-off session
that waits on a person doesn't keep the worker running or get continued after
every task. A handled task may hand off its session after that scan, so a run
that handled a task with a `recovered_interrupted_task_handler` reports
`continuations_pending`. Nothing else in the result covers work that becomes
due later without a scheduled time; the host must start the worker for it
itself:

- tasks created after the run, for example by the web process;
- a session handed off later, for example when a person answers in the web
  process, and attached handoffs whose owner's lease is still live;
- claims of a run that was killed or failed, which expire after
  `lease_seconds` (300 by default); start the worker again after a non-zero exit;
- tasks fenced after worker loss, which `reconcile_abandoned_after_s` settles
  only after a worker has seen them fenced for that long;
- tasks blocked by a concurrency key or a dependency held by work in another
  process;
- event retry deadlines of pending and paused tasks, enforced only while a
  worker runs.

The persisted event side-effect sweep runs in `cayu serve`, not in a worker.

An entrypoint that returns anything else fails with exit `1` and a message
naming the flag. `--until-idle` can't be combined with `--processes`; use
`concurrency` instead. The long-running preset's `cayu worker timers` supports
it; it creates timers only for pauses it sees when it starts, so also start it
after a session pauses.

## One process is one CPU core

A Python process runs all of its asyncio work on one CPU core. Model calls and
commands wait on the network or another process, but the runtime's own work
between them (preparing the next request, recording tool results and
evidence, store writes, environment setup) runs on that one core, shared by
every session the process is running. Adding `concurrency` to
`run_task_worker` or starting more sessions in the same process adds work to
that core, not cores.

One service that ran every review session inside its web process (it embedded
`run_task_worker` there) measured this across 4,349 sessions on a 4-vCPU host.
Medians in seconds, by how many sessions were running at the time:

| Segment | 1 running | 2–3 | 4–7 | 8+ |
|---|---|---|---|---|
| session start → first model call | 4.07 | 4.64 | 7.62 | 29.59 |
| `model.completed` → `tool.call.started` | 0.51 | 0.49 | 0.85 | 2.53 |
| tool done → `tool.exposure.recorded` | 0.40 | 0.38 | 0.61 | 2.18 |
| `tool.exposure.recorded` → `request.footprint.recorded` | 0.26 | 0.25 | 0.36 | 1.57 |
| `request.footprint.recorded` → `model.started` | 0.06 | 0.07 | 0.13 | 0.82 |
| `read_file` call | 0.09 | 0.10 | 0.23 | 1.46 |
| `search_text` call (the command itself takes about 0.17) | 0.30 | 0.33 | 0.83 | 4.92 |
| `list_files` call | 2.53 | 2.02 | 4.56 | 17.61 |

The runtime's own time between model calls went from about 1.2 s per step
with one session to about 7 s with eight or more, and tool calls took 5 to 16
times longer. None of the slow segments include model or command time. When
20 sessions started within one minute, 18 of them waited 100–120 s before
their first model call, with each environment lifecycle phase taking 10–20 s
instead of 0.1–0.6 s.

So the ceiling is a few sessions per process: slowdown starts at about four
concurrent sessions and is severe at eight. Past that, run the task worker in
more processes, which the store already coordinates:

- `cayu worker NAME --processes N` in a Cayu project (below). One process per
  core is a reasonable start.
- [`supervise_process_workers`](#start-worker-processes-from-your-own-code)
  from an application's own code, for example a web service that embeds the
  worker today.
- Separate containers or hosts, each running one or more worker processes.

Every process must use the same durable stores. SQLite allows one writer at a
time, so sustained writes from several processes want PostgreSQL. A web
process that only creates tasks and serves reads can stay a single process;
it's the processes that run sessions that need to scale out.

## Watch for saturation

`run_task_worker` and `TaskStoreDispatcher.run_worker` measure event-loop
lag while they run: one timer callback every half second records how late it
ran. In a healthy process that's a few milliseconds. When it reaches one
second, the worker logs a warning from the `cayu.observability.event_loop_lag`
logger, at most once a minute per process, saying that runtime work in this
process is queuing and to run more worker processes.

A `DurableWorkerMetrics` passed to the worker receives every sample in its
snapshot, next to the active handler count, for an application's own metrics
or alerts:

```python
from cayu.runtime import DurableWorkerMetrics

metrics = DurableWorkerMetrics(configured_handler_capacity=4)
# run_task_worker(..., concurrency=4, metrics=metrics)

snapshot = metrics.snapshot()
snapshot.current_event_loop_lag_s  # the latest sample
snapshot.event_loop_lag_max_s  # the worst since the metrics were created
snapshot.event_loop_lag_total_s / max(snapshot.event_loop_lag_samples, 1)  # mean
snapshot.active_handlers  # sessions this worker is running now
```

Lag that rises with `active_handlers` is the one-process ceiling. A process
that runs sessions without a task worker can watch the same thing with
`EventLoopLagMonitor` on its own:

```python
from cayu.observability import EventLoopLagMonitor

monitor = EventLoopLagMonitor(on_sample=lag_histogram.observe)
monitor.start()  # inside the running event loop, for example at app startup
...
monitor.snapshot().maximum_lag_s
monitor.stop()
```

`interval_s` (0.5 by default), `warning_threshold_s` (1.0; `None` turns the
warning off) and `warning_interval_s` (60) are keyword options.

## Use multiple CPU cores

```bash
cayu worker dispatch --processes 16 --shutdown-grace-seconds 45
```

On POSIX hosts, `--processes` starts independent Python interpreters, each with
its own application factory call and event loop. This allows Python runtime
work to use multiple cores. The default remains one process; the explicit
range is 1–256. The supervising process does not construct an application.
Workers receive zero-based `CAYU_WORKER_INDEX` and total `CAYU_WORKER_COUNT`
environment variables for process-specific configuration.

Configure all workers to use the same durable stores and unique worker IDs
(as in the example above). Existing task claims and lease fences coordinate
ownership. In-memory stores are separate in each process. Starting more
processes does not automatically partition an arbitrary worker entrypoint:
an entrypoint that unconditionally launches the same request will do so in
every process. Use the existing dispatcher or fresh-task loop for claimed work.
Concurrency limits configured inside an entrypoint apply per process.

A failed child stops its siblings. Signals request cooperative shutdown;
children that exceed the grace period are killed and reaped. Child processes
also watch for supervisor loss, including during project startup. Workers are
never automatically restarted: durable recovery remains an explicit store and
entrypoint policy. The supervisor controls process lifecycle, not external
container cleanup or sandbox isolation. Windows process mode is unsupported.

## Start worker processes from your own code

`cayu worker --processes` is built from a process supervisor that an
application can use without a Cayu project or the CLI. It starts each command
in its own process group, stops the others when one fails, and on a stop or
signal sends SIGTERM to every group, waits the grace, then SIGKILLs what is
left and reaps everything before it returns. It never restarts a child.

```python
# run_workers.py
import asyncio
import os
import sys
from pathlib import Path

from cayu.runtime import ProcessWorkerCommand, process_worker_environment, supervise_process_workers


async def main() -> int:
    count = os.cpu_count() or 1
    outcome = await supervise_process_workers(
        [
            ProcessWorkerCommand(
                argv=(sys.executable, "-m", "reviews.worker"),
                cwd=Path.cwd(),
                environment=process_worker_environment(index, count),
            )
            for index in range(count)
        ],
        # A little longer than each worker's own shutdown.
        shutdown_grace_seconds=46,
    )
    return outcome.exit_code


raise SystemExit(asyncio.run(main()))
```

`process_worker_environment(index, count)` copies this process's environment
and adds `CAYU_WORKER_INDEX`, `CAYU_WORKER_COUNT` and `CAYU_SUPERVISOR_PID`.
Each child is an ordinary program that builds its own app and stops on SIGTERM
or when the supervisor goes away:

```python
# reviews/worker.py
import asyncio
import os
import signal
import socket

from cayu.runtime import supervisor_watchdog, watch_supervisor


async def main() -> None:
    # Imported here, after the watchdog below has started, so a supervisor that
    # dies while this process is still importing application code stops it too.
    from cayu.tasks import TaskQuery, run_task_worker

    from reviews.app import build_app, handle_review

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    watcher = asyncio.create_task(watch_supervisor(lambda _signum: stop.set()))
    app = build_app()
    try:
        await run_task_worker(
            app,
            app.task_store,
            handle_review,
            worker_id=f"{socket.gethostname()}-{os.getpid()}-reviews",
            query=TaskQuery(type="review"),
            stop=stop,
        )
    finally:
        watcher.cancel()
        await app.aclose()


# Stops the child if the supervisor dies, including during the imports and
# build_app() in main().
with supervisor_watchdog(45):
    asyncio.run(main())
```

By default the supervisor installs SIGINT and SIGTERM handlers, so it must run
on the main thread, and puts the previous handlers back when it returns. A
host that handles signals itself, such as a web server starting its workers
at startup, passes `handle_signals=False` and its own `stop` event:

```python
stop = asyncio.Event()
supervisor = asyncio.create_task(
    supervise_process_workers(commands, shutdown_grace_seconds=46, stop=stop, handle_signals=False)
)
# ... at shutdown:
stop.set()
outcome = await supervisor
```

`outcome.exit_code` is `0` when every child exited `0` or was stopped through
`stop`, the first failing child's code otherwise, `128 + signal` after SIGINT
or SIGTERM, and `124` when a child had to be killed. Cancelling the call also
stops and reaps the children. POSIX only.

## How `cayu worker` stops and exits

SIGINT and SIGTERM set the cooperative event and allow the configured grace
period. When the target returns, the CLI closes the app with `app.aclose()`,
using what remains of that grace after a signal, and prints a warning if the
shutdown did not settle. Normal completion exits `0`, SIGINT after cooperative
shutdown exits `130`, SIGTERM exits `143`, validation or startup failure exits
`1`, and a shutdown timeout exits `124` after cancelling the local worker task. Exit `124`
is a hard process boundary so even a cancellation-resistant target cannot
extend the configured grace period. Project import, factory, and entrypoint
`SystemExit` failures are converted into labeled CLI errors rather than escaping
without context. Existing SIGINT and SIGTERM handlers are restored exactly when
the worker returns.
