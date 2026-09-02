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

from cayu import CayuApp, TaskStoreDispatcher


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
from cayu import CayuApp, Task, TaskQuery, complete_managed_task, run_task_worker


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
        stop=stop,
    )
```

The CLI does not infer a handler, query, task store, abandoned-session
boundary, or recovery policy. Configure those in the factory or named
entrypoint. In particular, `cayu worker` performs no implicit incomplete-session
recovery.

SIGINT and SIGTERM set the cooperative event and allow the configured grace
period. Normal completion exits `0`, SIGINT after cooperative shutdown exits
`130`, SIGTERM exits `143`, validation or startup failure exits `1`, and a
shutdown timeout exits `124` after cancelling the local worker task. Exit `124`
is a hard process boundary so even a cancellation-resistant target cannot
extend the configured grace period. Project import, factory, and entrypoint
`SystemExit` failures are converted into labeled CLI errors rather than escaping
without context. Existing SIGINT and SIGTERM handlers are restored exactly when
the worker returns.
