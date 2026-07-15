# Cayu application anatomy

A Cayu application has one explicit boot contract: a project declares an
application factory, and every process calls that factory to construct its own
application graph. The Python object is process-local. Configured durable stores
are the coordination boundary between processes.

## Project and factory

A **Cayu project** is a directory rooted by project configuration that declares a
synchronous application factory:

```toml
[tool.cayu]
factory = "app:build_app"
```

The target is a synchronous callable that accepts a zero-argument call and
returns a fresh `CayuApp`:

```python
from cayu import CayuApp


def build_app() -> CayuApp:
    app = CayuApp()
    # Register this process's providers, environments, tools, and agents.
    return app
```

Optional dependency-injection arguments are useful public test seams, provided a
normal `build_app()` call remains valid. Importing the project module must not
call the factory, connect to external services, run migrations, start background
activity, or invoke models or tools.

## Process-scoped application graph

Each factory call returns a distinct, process-scoped `CayuApp`. The app is not a
framework singleton, global registry, or durable coordination mechanism. A
console, server integration, worker integration, script, and test each construct
their own graph, even when all of them point at the same storage configuration.

Configured session, task, knowledge, artifact, watcher, budget, and other durable
stores carry shared state across processes. In-memory stores and Python object
identity do not. Registration and live resource ownership remain local to the
process that constructed the app.

## Application lifecycle boundaries

Keep these four responsibilities conceptually separate so a host makes its
operational effects explicit:

| Boundary | Meaning | What it does not imply |
| --- | --- | --- |
| Application construction | Call the factory and compose the process-local graph. | Active work has started or another process shares the app object. |
| Resource acquisition | Open or attach owned database, network, sandbox, or host resources. | Schemas are migrated or active services are running. |
| Administrative initialization | Explicitly perform migrations, recovery selection, seeding, or maintenance. | A long-running service owns the process. |
| Active-service startup | Explicitly start a server lifespan, worker loop, watcher, scheduler, or other active integration. | Other processes share this app object or its lifecycle. |

Cayu does not yet impose a general `CayuApp` lifecycle protocol, so a configured
component can perform more than one responsibility in its constructor. In the
generated local project, for example, SQLite store constructors open their files
and ensure their schemas; `cayu console`, `cayu inspect`, and `cayu check` call
the factory and therefore exercise that configured behavior. Keep module imports
inert, keep constructor effects bounded and documented, and leave active-service
startup and cleanup under the explicit host or process entrypoint that owns them.

## Process roles

| Role | Application ownership | Shared state | Automatic active services |
| --- | --- | --- | --- |
| One-off script | Calls the factory for its process. | Configured durable stores. | Only behavior explicitly invoked by the script. |
| Interactive console | Calls the factory once for the console process. | Configured durable stores. | None. |
| Server integration | Calls the factory for the server process. | Configured durable stores. | Only lifecycle behavior explicitly owned by the server integration. |
| Worker integration | Calls the factory for the worker process. | Configured durable stores. | Only worker behavior explicitly started by that process. |
| Test | Calls the factory for the test or fixture scope. | Test-selected stores. | None unless the test starts it. |

`cayu console`, `cayu inspect`, and `cayu check` use the declared factory today.
Server, script, and worker integrations should follow the same contract, but
`cayu server`, `cayu script`, and `cayu worker` are roadmap work tracked by
issues #233, #234, and #236; they are not shipped commands.

## Console contract

`cayu console` constructs one console-local app and binds it as `app`. That name
is not a framework registry or singleton. Opening the console does not start a
server lifespan, recovery, workers, watchers, schedulers, sessions, models, or
tools. Operations requested interactively may affect the same durable backends
used by other processes.

File-backed Cayu SQLite stores use WAL and a busy timeout, but SQLite remains
single-writer. Another process can still contend with a console write and raise
`database is locked`. Prefer a deployment-appropriate shared store such as
PostgreSQL when multiple active processes need sustained write concurrency.

## Dependency boundary

Generated production dependencies use base `cayu`. Interactive console support
is an explicit development extra such as `cayu[console]`; a production process
does not need to install REPL tooling merely because the project declares a
factory.

## Anti-patterns

Avoid these shapes:

- a module-global `CayuApp` used as shared application state;
- calling `build_app()` during module import;
- starting migrations, recovery, workers, watchers, schedulers, models, or tools
  as an import side effect; and
- treating the console-local `app` binding as a framework-wide registry.

## Verify the contract

For a generated project, verify the public boundary rather than exact prose:

```bash
cayu inspect --json
cayu check --json
pytest
```

Project tests should prove that importing `app.py` does not construct the app,
two factory calls return distinct `CayuApp` objects, and `[tool.cayu].factory`
resolves to that callable. These checks prove structure and deterministic runtime
behavior; they do not claim live provider, environment, or service verification.
