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
requeued only when their handler stops within the grace.

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
