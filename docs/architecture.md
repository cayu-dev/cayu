# Cayu Architecture

This is a design/maintainer document for the current framework foundation. It records architecture decisions and intended direction; it is not a complete end-user guide.

Cayu is a backend/runtime-first Python framework for building long-running agents, multi-agent workflows, and sandboxed tool runtimes.

The framework should run locally, on a VPS, in Docker, in ECS, or behind any other normal runtime. Hosted deployments should be adapters around the framework, not a requirement for using it.

## Core Decisions

- Repo/package/CLI name: `cayu`
- Language: Python for v1
- Framework repo structure: horizontal by subsystem
- Generated user projects: Rails-like default layout, vertical domain modules allowed
- CLI: developer/admin utility, not the primary product interface
- Dashboard: optional viewer over runtime events and session storage
- MCP: interoperability layer, not the required custom tool model
- Runtime model: separate agent, environment, and session concerns

## Dependency Direction

```text
core
  providers -> core
  runners -> core
  workspaces -> core
  storage -> core
  vaults -> core
  mcp -> core
  environments -> workspaces + runners + vaults + mcp
  runtime -> core + providers + runners + workspaces + storage + vaults + mcp
  cli -> runtime + project scaffolding
  dashboard -> runtime API / event store
```

`core` should stay small and stable. It defines events, messages, agents, tools, workflows, and shared value objects.

## Runtime Shape

```text
RunRequest
  -> SessionStore creates session
  -> Environment provides execution context
  -> Agent runtime streams provider/tool/workflow events
  -> EventSink emits to terminal/dashboard/webhook
  -> SessionStore persists append-only event log
```

Every important action should produce an event. Events are the shared contract for debugging, dashboards, hosted integrations, replay, and tests.
Event identity fields such as agent, environment, workflow, and tool should be top-level event fields so event stores and dashboards can index them without parsing payload JSON.

Runtime inputs are copied at framework boundaries. Framework code should depend on explicit registration and validated contract objects, not on later mutation of user-owned Python objects.

JSON-like contract fields should remain portable across local, hosted, and remote execution. They should contain JSON-compatible values only, without Python-specific object identity, circular references, or special numeric values such as NaN and Infinity.

## Multi-Agent Shape

Cayu must support systems where multiple agents collaborate through shared state.

```text
Agent A
  -> writes record/task/event to SharedState
  -> trigger starts Agent B
  -> Agent B claims task and writes result
  -> orchestrator reviews, retries, escalates, or delegates
```

This requires both deterministic orchestration and LLM orchestrator agents.

## Workspace, Runner, Sandbox

Cayu follows an agent/environment/session separation:

- `Agent`: model, system prompt, tool declarations, and metadata.
- `Environment`: workspace, runner, vault, MCP servers, and execution metadata.
- `Session`: one run of an agent in an environment, with messages, status, events, and checkpoints.

- `Workspace`: files/artifacts an agent can work with.
- `Runner`: executes explicit `ExecCommand` values in a workspace or sandbox.
- `Sandbox`: isolated workspace plus runner plus lifecycle and limits.

`LocalRunner` is not a sandbox. It is only a development or already-disposable-environment execution backend.

Process commands should use argv form. Shell execution should be an explicit mode, because hosted runners need to enforce quoting, limits, logging, and security consistently.

## Storage and Memory

Files are good source-of-truth for prompts, instructions, workflows, manuals, skills, and human-reviewed memories.

Databases/indexes are better for sessions, event logs, high-volume memories, permissions, embeddings, search, and hosted multi-user state.

Default local strategy:

```text
files for human-readable source
SQLite for sessions, append-only events, transcripts, checkpoints, and indexes
SQLite FTS/BM25 for default keyword retrieval
optional vector search later
```

The local durable session store is `SQLiteSessionStore`. It keeps the event log append-only, but stores indexed identity columns beside the JSON event payload so dashboards and replay tools do not have to scan transcript files. It also stores the provider-neutral transcript messages needed for future resume and compaction APIs. Storage APIs support filtered session listing, filtered event queries with durable sequence cursors, transcript loading, and atomic batched event appends. JSONL is better treated as an export/debug format than as Cayu's primary runtime store.

Context policies are runtime projections over transcript messages, not storage. They let applications customize the model-facing conversation history by trimming, compacting, replacing bulky tool results, or injecting retrieved context while preserving the raw durable transcript for audit, debugging, resume, and future compaction.

Tasks are optional durable work items, not a required execution model. A simple agent can run with only sessions and events. A background job, orchestrated multi-agent app, webhook processor, or dashboard-visible queue can use `TaskStore` to track work status, inputs, outputs, errors, ownership, and parent/child relationships. `SQLiteTaskStore` is the local durable task implementation.
