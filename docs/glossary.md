# Glossary and naming notes

A few cayu names collide with common Python terms or with each other. This
disambiguates them so a name never sends you down the wrong path.

## Names that collide with Python

- **`Task` (cayu) vs `asyncio.Task`.** `cayu.Task` is a durable unit of work in a
  `TaskStore` (created with `TaskCreate`, claimed by a worker). `asyncio.Task`
  wraps a coroutine. They co-occur — a worker often runs an `asyncio.Task` that
  heartbeats a cayu `Task` — so name your locals accordingly (`bg_task`,
  `heartbeat_handle`), not `task`.

- **`Environment` vs OS environment variables.** A cayu `Environment` is an agent's
  execution context — workspace, runner, vault, credential proxy, MCP servers. It
  is *not* OS environment variables; those are the runner's `env` / `inherit_env`
  settings (see `LocalRunner`).

- **`Runner` runs commands, not agents.** A `Runner` executes shell/process
  commands inside a workspace (what `exec_command` uses). The *agent* loop is
  `app.run(...)`. A `Runner` never "runs the agent."

## Names that collide with each other

- **`app.run` vs `app.resume` vs `app.dispatch` vs task claiming vs `SubagentTool`.**
  These are different ways to *start or continue* a run. See
  [triggering-runs.md](triggering-runs.md) for a decision table.

- **`resume` (session) vs task lifecycle.** `app.resume(ResumeRequest)` appends
  messages to an existing session's durable transcript. Task workers use
  `claim_task` / `complete_task` / `fail_task` — a separate lifecycle, not
  "resuming a task."

- **`*Spec` is a suffix with a consistent meaning, one exception.** `AgentSpec`,
  `EnvironmentSpec`, `WorkflowSpec` are the portable, serializable *core* of a
  declaration; live objects (tools, workspaces, runners) attach at construction or
  registration, not on the spec. `ToolSpec` is the odd one out ergonomically: it is
  set as a class attribute on a `Tool` subclass rather than passed to a constructor.

- **Eval assertions vs runtime events.** `SessionCompleted` / `SessionFailed` /
  `SessionInterrupted` are eval *assertions* (`cayu.evals`) that check a run's
  outcome. `EventType.SESSION_COMPLETED` / `SESSION_FAILED` / `SESSION_INTERRUPTED`
  are the runtime *events* those assertions inspect. If autocomplete offers
  `SessionStatusIs`, that too is an assertion, not an event.

- **`Session` (the run) vs `SessionStore` (its persistence).** A `Session` is one
  run of an agent in an environment; a `SessionStore` (in-memory / SQLite /
  Postgres) persists sessions, events, and transcripts.

## A few load-bearing terms

- **Knowledge vs memory.** Knowledge is durable canonical semantic material
  (entries, chunks, source identity, lifecycle, and eventually immutable
  revisions/evidence). Memory is the broader recall and context system that may
  retrieve permitted knowledge, transcript episodes, and artifact-derived
  documents. A `KnowledgeStore` is one memory source, not the whole memory
  system. See [Memory foundation contracts](memory-foundation.md).
- **Model-provider credential.** Host/control-process authority used by a
  `ModelProvider` or its refresh transport to make model requests. It is
  non-delegable by default; provider registration never makes it a runner,
  tool, workspace, vault, or virtual-egress credential.
- **Workload credential.** Authority deliberately granted to a tool, runner,
  broker, or integration through Cayu's workload credential contracts. Its
  `CredentialMode` determines whether the real value is agent-readable,
  trusted-tool-only, or represented by a virtual credential.
- **Workspace.** The mutable filesystem an agent's tools read and write during a run.
  Whether changes persist past the run depends on the `WorkspaceBinding` (e.g. a
  `SyncBinding` copies changes back; `GitRepositoryBinding` checks out a repo).
- **Binding.** The bridge between a durable source and the run's live workspace —
  it prepares the workspace before the run and finalizes it after.
- **Environment factory.** A callable that builds a fresh `Environment` per session
  (keyed on `session_id` / `agent_name`), instead of one static `Environment`.
