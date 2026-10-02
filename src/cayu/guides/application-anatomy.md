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
runtime singleton, global registry, or durable coordination mechanism. A
console, server integration, worker integration, script, and test each construct
their own graph, even when all of them point at the same storage configuration.

Configured session, task, knowledge, artifact, watcher, budget, and other durable
stores carry shared state across processes. In-memory stores and Python object
identity do not. Registration and live resource ownership remain local to the
process that constructed the app.

## Resume versus anatomical succession

Reconstructing an app and calling `resume(...)` keeps the same durable session.
The current application graph supplies the registered tools and policies, but
the session's existing transcript remains authoritative—including its original
system prompt. Changing `AgentSpec.system_prompt` does not silently rewrite that
history.

When a new agent body must inherit conversation history but install its current
prompt anatomy, create an explicit descendant instead:

```python
from cayu import (
    ExecutionProfileAdoptionIntent,
    ForkExecutionProfileSelection,
    ForkSessionRequest,
    ForkSystemPromptPolicy,
    ResolutionActor,
    ResolutionActorSource,
)

events = [
    event
    async for event in app.fork_session(
        ForkSessionRequest(
            source_session_id="agent-v1-session",
            session_id="agent-v2-session",
            agent_name="agent-v2",
            copy_checkpoint=False,
            system_prompt_policy=ForkSystemPromptPolicy.CURRENT_AGENT,
            execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
            profile_adoption=ExecutionProfileAdoptionIntent(
                idempotency_key="agent-v2-session-profile",
                reason="Install the registered agent-v2 execution profile.",
                requested_by=ResolutionActor(
                    subject="release-operator",
                    source=ResolutionActorSource.REQUEST,
                ),
            ),
        )
    )
]
```

This opt-in fork renders the current registered agent prompt and current
workspace instructions, atomically replaces inherited system messages, and
preserves the selected non-system history. It records a durable
`PromptAnatomyTransitionReceipt` containing source/child prompt SHA-256 digests
and transition identities, plus source/child model identities and the successful
portability check, never prompt text. Before descendant creation the source
checkpoint records a durable transition intent, so a process restart can
continue the exact request. The source transcript stays unchanged. Exact
retries reuse the caller-selected destination ID when supplied; otherwise Cayu
derives one opaque parent-and-request-scoped ID. Both forms converge on the same
descendant and receipt.

Selecting `CURRENT_CHILD` is an explicit execution-profile adoption request.
The application's `ExecutionProfilePolicy` must authorize the current
registration's decision-bearing authority before the child is created, including
environment, hooks, tool policy, runner, credential, network, and external-effect
authority that is not part of the structural profile fingerprint. A provider
change always projects and preflights the complete portable transcript, even when
the model name is unchanged. Omitting `execution_profile_selection` instead
inherits the parent's effective durable profile and does not consult mutable
current registrations to choose a different child baseline.

Prompt succession currently requires `copy_checkpoint=False` and a concrete
registered environment. This is deliberate: a checkpoint or an unmaterialized
environment factory can contain body-specific execution state that the runtime
cannot prove compatible with the new prompt anatomy. Ordinary forks continue to
inherit the source prompt by default.

For a body upgrade that hopes to keep the exact session ID, compare the
persisted transcript's `system_prompt_messages_sha256(...)` with
`await app.current_prompt_anatomy_sha256(...)`. Only an equal digest proves the
new application graph would install the same prompt anatomy; otherwise fork.

## Application lifecycle boundaries

Keep these five responsibilities conceptually separate so a host makes its
operational effects explicit:

| Boundary | Meaning | What it does not imply |
| --- | --- | --- |
| Application construction | Call the factory and compose the process-local graph. | Active work has started or another process shares the app object. |
| Resource acquisition | Open or attach owned database, network, sandbox, or host resources. | Schemas are migrated or active services are running. |
| Administrative initialization | Explicitly perform migrations, recovery selection, seeding, or maintenance. | A long-running service owns the process. |
| Active-service startup | Explicitly start a server lifespan, worker loop, watcher, scheduler, or other active integration. | Other processes share this app object or its lifecycle. |
| Shutdown | `await app.aclose()` refuses new execution, waits for operations in flight, and drains this app's subsystems under one deadline. | Caller-supplied stores or providers are closed, or other processes have stopped. |

Construction and resource acquisition have no general protocol, so a configured
component can perform more than one responsibility in its constructor. In the
generated local project, for example, SQLite store constructors open their files
and ensure their schemas, while PostgreSQL stores connect lazily and only validate
the schema that `cayu storage migrate` applied; `cayu console`, `cayu inspect`, and
`cayu check` call the factory and therefore exercise that configured behavior. Keep module imports
inert, keep constructor effects bounded and documented, and leave active-service
startup and cleanup under the explicit host or process entrypoint that owns them.

Shutdown is one call. The host that built the app closes it, and hands over any
resources it wants closed with it:

```python
from contextlib import aclosing
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    ModelProvider,
    RunRequest,
    configured_database_url,
    open_application_stores,
)


async def main(provider: ModelProvider) -> None:
    stores = open_application_stores(
        configured_database_url(), sqlite_path=Path("state/app.sqlite3").resolve()
    )
    # The app closes the stores, but only after every subsystem settled.
    app = CayuApp(
        session_store=stores.session_store,
        task_store=stores.task_store,
        owned_resources=(stores,),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="your-model"))
    async with app:
        request = RunRequest(agent_name="assistant", messages=[Message.text("user", "Hi")])
        async with aclosing(app.run(request)) as events:
            async for event in events:
                print(event.type)
    outcome = app.shutdown_outcome
    if outcome is not None and not outcome.settled:
        print(f"shutdown {outcome.status}; stores kept open, retry app.aclose()")
```

`async with app:` calls `aclose()` on exit. `aclosing(...)` closes a stream you
stop reading early, so shutdown does not wait for it. Without `owned_resources`,
close your own stores only once `app.shutdown_outcome.settled` is true. Do not
nest the app inside `async with stores:`, which closes them unconditionally. The
order, deadline, and outcome are specified in the runtime contracts.

## Process roles

| Role | Application ownership | Shared state | Automatic active services |
| --- | --- | --- | --- |
| One-off script | Calls the factory for its process. | Configured durable stores. | Only behavior explicitly invoked by the script. |
| Interactive console | Calls the factory once for the console process. | Configured durable stores. | None. |
| Server integration | Calls the factory for the server process. | Configured durable stores. | Only lifecycle behavior explicitly owned by the server integration. |
| Worker integration | Calls the factory for the worker process. | Configured durable stores. | Only worker behavior explicitly started by that process. |
| Test | Calls the factory for the test or fixture scope. | Test-selected stores. | None unless the test starts it. |

`cayu console`, `cayu inspect`, and `cayu check` use the declared factory for
operator workflows. `cayu serve` and `cayu worker` are shipped process-role
adapters that construct the same application through its declared factory.
One-off scripts should call that factory directly and own the lifecycle of any
active services they start; Cayu does not ship a `cayu script` command.

## Durable data

Generated projects build their session, task, and knowledge stores with
`open_application_stores(configured_database_url(), sqlite_path=...)` in
`configuration/storage.py`. The same code runs locally and in a deployment:

| Variable | Effect |
| --- | --- |
| `CAYU_DATABASE_URL` | `postgres://` or `postgresql://` selects PostgreSQL; an absolute `sqlite:///` URL selects that file. Unset selects SQLite at the project's absolute `sqlite_path` (`data/cayu.db`). |
| `CAYU_DATABASE_POOL_MAX` | Maximum connections in the one pool the PostgreSQL stores share (default 5). The Evals store that `cayu serve` and `cayu check` open uses its own pool of the same size. |
| `CAYU_DATABASE_DIRECT_URL` | Optional direct server address for the one task-admission `LISTEN` connection, kept outside the pool. Set it when `CAYU_DATABASE_URL` points at a transaction-pooling proxy such as PgBouncer, where `LISTEN` does not work. |
| `CAYU_REQUIRE_POSTGRES` | `1` makes every Cayu SQLite store raise at construction, so a deployment missing `CAYU_DATABASE_URL` fails at startup instead of writing local files. |

One PostgreSQL-backed application process opens at most `CAYU_DATABASE_POOL_MAX`
pooled connections plus the listener, and the Evals store's pool when a Cayu
command serves the project. Cayu CLI commands resolve `CAYU_DATABASE_URL` before
`[tool.cayu.session_store]`, so they inspect the same database the app uses.

Keep application-owned durable records in the configured database. Application
tables may share it with their own table prefix; Cayu reserves `cayu_` for its
tables (ADR 0001). `data/` locally and `/data` in a deployment hold files such as
artifacts, uploads, and fixtures. Do not open SQLite files there for durable
application state: a deployment's `/data` may be a network filesystem, where
SQLite is slow and its WAL mode is unsupported. Deployments of applications with
automatic memory must set `CAYU_MEMORY_EVIDENCE_KEY`, because the local
`data/memory-evidence.key` is git-ignored and never reaches a deployment.

## Console contract

`cayu console` constructs one console-local app and binds it as `app`. That name
is not a runtime registry or singleton. Opening the console does not start a
server lifespan, recovery, workers, watchers, schedulers, sessions, models, or
tools. Operations requested interactively may affect the same durable backends
used by other processes.

File-backed Cayu SQLite stores use WAL and a busy timeout, but SQLite remains
single-writer. Another process can still contend with a console write and raise
`database is locked`. Prefer a deployment-appropriate shared store such as
PostgreSQL when multiple active processes need sustained write concurrency.

## Dependency boundary

Generated production dependencies use `cayu[postgres]`, so any project can switch
to PostgreSQL through `CAYU_DATABASE_URL`. Interactive console support
is an explicit development extra such as `cayu[console]`; a production process
does not need to install REPL tooling merely because the project declares a
factory.

## Usage and cost

Read token usage and cost from Cayu instead of computing them in the
application. In Python, call `await app.get_session_usage(session_id)` and
`await app.get_session_cost(session_id, pricing)`. Over HTTP, use
`GET /api/sessions/{session_id}/usage`, `GET /api/sessions?include=usage` for a
page of sessions, and `POST /api/sessions/{session_id}/cost`; `GET /api/contract`
lists these paths under `accounting`. Usage reads are incremental in every
built-in session store, and the HTTP usage endpoint supports `ETag` and
`If-None-Match`, so a UI can poll it cheaply.

Do not fold raw `model.completed` events to compute tokens or cost. That misses
cache counters, auxiliary provider attempts, hosted-tool usage, and price-book
rules, and it rereads the whole event history on every refresh.

## Anti-patterns

Avoid these shapes:

- a module-global `CayuApp` used as shared application state;
- calling `build_app()` during module import;
- starting migrations, recovery, workers, watchers, schedulers, models, or tools
  as an import side effect;
- summing `model.completed` event payloads to show tokens or cost; and
- treating the console-local `app` binding as a runtime-wide registry.

## Verify the contract

For a generated project, verify the public boundary rather than exact prose:

```bash
cayu inspect --json
cayu check --fail-on warning --json
pytest
```

Project tests should prove that importing `app.py` does not construct the app,
two factory calls return distinct `CayuApp` objects, and `[tool.cayu].factory`
resolves to that callable. These checks prove structure and deterministic runtime
behavior; they do not claim live provider, environment, or service verification.
Use `cayu guide applications` for the generated placement convention,
normalized plan, generator compatibility, and declared-layout diagnostics.
Use `cayu guide app-ui` ([source](app-ui.md)) before building a browser UI
over sessions: it covers the served `client.js`, polling rules, read-only GET
handlers, and the idle-tab budget for the server integration role.
