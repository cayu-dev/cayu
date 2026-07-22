# Cayu Architecture

This is a design/maintainer document for Cayu's runtime framework. It records architecture decisions and intended direction; it is not a complete end-user guide.

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

MCP tools should enter the runtime as normal Cayu tools through adapters. That keeps
external servers under the same policy, approval, event, and transcript model as native
Python tools while preserving MCP as an interoperability boundary.

## Dependency Direction

```text
core
  providers -> core
  artifacts -> core
  runners -> core
  workspaces -> core
  storage -> core
  vaults -> core
  proxies -> vaults + core
  egress -> proxies + vaults + core   (Docker adapter also uses runners)
  mcp -> core
  environments -> artifacts + workspaces + runners + vaults + proxies + mcp
  runtime -> core + providers + artifacts + runners + workspaces + storage + vaults + proxies + mcp
  workflows -> core + runtime
  cli -> runtime + project scaffolding
  dashboard -> runtime API / event store
```

`core` should stay small and stable. It defines events, messages, agents, tools, the abstract workflow contract, and shared value objects.
`workflows` contains orchestration-as-code helper primitives layered above the runtime; it may depend on runtime session/event contracts, but runtime should not depend on workflow helpers.

## Runtime Shape

```text
RunRequest
  -> SessionStore creates session
  -> Environment provides execution context
  -> Agent runtime streams provider/tool events
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

- `Agent`: an `AgentSpec` — model, system prompt, and metadata. Tools and tool policy are attached separately at `register_agent(spec, tools=..., tool_policy=...)`.
- `Environment`: workspace, artifact store, runner, vault, credential proxy, MCP servers, and execution metadata.
- `Session`: one run of an agent in an environment, with messages, status, events, and checkpoints.

The `*Spec` types (`AgentSpec`, `EnvironmentSpec`, …) are the portable, serializable core of a declaration; live objects — tools, workspaces, runners, providers — are attached at construction or registration, not stored on the spec.

- `Workspace`: active filesystem an agent can work with, such as a target repo or working directory.
- `ArtifactStore`: uploaded/generated durable file references scoped to a session or environment.
- `Runner`: executes explicit `ExecCommand` values in a workspace or sandbox.
- `Sandbox`: isolated workspace plus runner plus lifecycle and limits.

Workspaces and artifacts are separate on purpose:

- Use the workspace for mutable work: cloned repositories, temporary files, generated outputs, editable documents, test results, and command-line processing.
- Use the artifact store for durable file objects: original uploads, stable snapshots, final outputs, evidence, attachments, and files that must survive replay/fork/resume independently of the current workspace state.
- Use Git inside the workspace for code repositories when the Git remote is the source of truth. A coding agent should usually clone into the sandbox workspace, edit there, and only commit/push through explicit user policy.
- Copy artifacts into the workspace only when a tool or script needs a path-backed mutable file. Store workspace files back as artifacts only when a generated/edited result should become durable output.

There is no implicit bidirectional sync between artifacts and the workspace. Copies are explicit one-way operations.

Model-facing file reads should persist artifact references, not provider-specific file payloads. The runtime resolves those references from the active artifact store immediately before provider calls, and provider adapters translate them into Anthropic/OpenAI/etc. native file/image/document content. Built-in context policies strip older native attachment references from provider-facing history while keeping transcript summaries, so file-heavy sessions do not resend the same bytes indefinitely. This keeps transcripts portable while still allowing multimodal providers to inspect images and PDFs.

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
provider-neutral embeddings plus in-memory semantic retrieval for demos/tests
backend-specific durable vector indexes later
```

The local durable session store is `SQLiteSessionStore`. New projects conventionally share `data/cayu.db` across Cayu's SQLite-backed runtime stores; applications may select another path explicitly. It keeps the event log append-only, but stores indexed identity columns beside the JSON event payload so dashboards and replay tools do not have to scan transcript files. Session records also persist provider, active model, runtime, agent, and environment identity so resume does not silently follow changed application defaults. New sessions start from the agent's default model; resume can durably update the session's active model within the stored provider. The store also keeps the provider-neutral transcript messages used by explicit session resume and checkpoint-backed context compaction. Storage APIs support filtered session listing, filtered event queries with durable sequence cursors, transcript loading, atomic status transitions, active model updates, and atomic batched event appends. JSONL is better treated as an export/debug format than as Cayu's primary runtime store.

Context policies are runtime projections over transcript messages, not storage. They let applications customize the model-facing conversation history by trimming, compacting, replacing bulky tool results, or injecting retrieved context while preserving the raw durable transcript for audit, debugging, resume, and future compaction.

Tasks are optional durable work items, not a required execution model. A simple agent can run with only sessions and events. A background job, orchestrated multi-agent app, webhook processor, or dashboard-visible queue can use `TaskStore` to track work status, inputs, outputs, errors, ownership, and parent/child relationships. `SQLiteTaskStore` is the local durable task implementation.
