# Cayu

Cayu is an open-source runtime for production AI agents.

Cayu is built for agents that need to survive real work: long-running sessions,
tool calls, human approval, context limits, retries, budget enforcement, durable
state, evals, and recovery.

Cayu is not a prompt-chain DSL or an orchestration framework. It is the runtime
layer you build on directly: sessions, tools, storage, context management,
provider routing, observability, and human-in-the-loop control.

You do not need LangChain, LangGraph, CrewAI, or another agent framework to build
production agents with Cayu. Cayu owns the loop those frameworks usually hide:
model calls, tool execution, transcript state, approvals, budgets, context
projection, recovery, and events.

## Why Cayu

Most agent frameworks help you assemble prompts, tools, and graphs. Production
agent systems usually fail somewhere else:

- The model calls the right tool with the wrong authority.
- A tool needs human approval halfway through a run.
- A process crashes after a side effect but before the transcript is updated.
- Context grows until the provider rejects the request.
- A team cannot explain why one task spent too much money.
- Memory becomes unreviewed model-written state.
- Eval coverage lives outside the runtime and misses real behavior.

Cayu treats those as runtime problems, not application afterthoughts.

Frameworks are useful for getting a prototype moving. Cayu is for the point
where the agent starts doing consequential work and you need to own the runtime
instead of debugging through framework abstractions.

## What Cayu Provides

| Problem | Cayu primitive |
| --- | --- |
| Long-running work | Durable sessions, transcripts, events, resume, fork |
| Tool safety | Tool policies, approvals, taint-aware controls |
| Human-in-the-loop | `ask_user`, approval checkpoints, recovery APIs |
| Context pressure | Counting, estimation, compaction, overflow recovery |
| Knowledge | Reviewed memory, indexing, recall tools, pgvector support |
| Cost control | Usage events, budget policies, causal budget summaries |
| Provider flexibility | OpenAI, Anthropic, Amazon Bedrock, Vertex, OpenAI-compatible providers |
| Sandboxed commands | Docker, E2B, AWS Lambda MicroVM, and deny-by-default Microsandbox runners |
| Evals | Runtime trajectories, assertions, replay, LLM judges |

## Environments Are The Execution Boundary

In Cayu, an agent is separated from the environment it runs in.

`AgentSpec` describes who the agent is: model, instructions, defaults, and
runtime policies.

`Environment` describes what the agent can touch: workspace, artifact store,
runner, vault, credential proxy, knowledge store, MCP servers, and workspace
binding.

That split lets the same agent run locally, in Docker, in a microVM sandbox, in
E2B, or behind your own hosted infrastructure without rewriting the agent.

```text
AgentSpec
  -> identity, model, instructions, policies

Environment
  -> workspace, runner, artifacts, vault, proxy, knowledge, MCP

Session
  -> durable transcript, events, provider/model identity, checkpoints

ToolContext
  -> runtime services exposed to a tool for one run
```

This is why Cayu treats tools as runtime work, not just model schemas. A command
tool needs a runner. A file tool needs a workspace. A document tool may need an
artifact store. A production tool may need scoped secrets, approval, proxying,
and durable recovery.

Those services come from the active environment, not from global process state.

```python
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    LocalArtifactStore,
    LocalRunner,
    LocalWorkspace,
)

app = CayuApp()
root = Path("./workspace")
root.mkdir(parents=True, exist_ok=True)

app.register_environment(
    Environment(
        EnvironmentSpec(name="local-dev"),
        workspace=LocalWorkspace(root, workspace_id="local"),
        artifact_store=LocalArtifactStore("./.cayu/artifacts"),
        runner=LocalRunner(root),
    ),
    default=True,
)

app.register_agent(
    AgentSpec(
        name="builder",
        model="gpt-5.6",
        system_prompt="You are a careful coding agent.",
    )
)
```

`default=True` makes the environment available to runs that omit
`RunRequest.environment_name`. Without it, the environment remains available
only by name; even the first registration is never selected implicitly.

### Fresh Environments Per Session

Static environments work for local development and trusted single-process apps.
For production, Cayu can register an `EnvironmentFactory` that creates or
reattaches the concrete environment for each session.

Factories receive durable session context and can return reconnect metadata,
such as a sandbox id, region, image, or attach handle. On resume, approval
continuation, or recovery, Cayu passes that metadata back so the runtime can
reattach instead of guessing.

## Contents

- [Quickstart](#quickstart)
- [Application console](#application-console)
- [Featured example: cloud PR reviewer](#featured-example-cloud-pr-reviewer)
- [Advanced runtime examples](#advanced-runtime-examples)
- [Further reading](#further-reading)
- [Design goals](#design-goals) · [Scope](#scope) · [Contract rules](#contract-rules) · [Repository layout](#repository-layout)
- [Development](#development) · [Usage and cache metrics](#usage-and-cache-metrics) · [Thinking and reasoning](#thinking-and-reasoning) · [Examples](#example)

## Design Goals

- Build real agent applications, not just hosted prompt/config definitions.
- Support multiple agents collaborating through shared state, channels, tasks, triggers, and workflows.
- Treat tool execution as a first-class runtime concern with explicit workspace, runner, vault, and sandbox contracts.
- Store every important run action as structured events for CLI output, dashboard inspection, webhooks, and replay.
- Run locally, in containers, on hosted infrastructure, or behind an application server.
- Make MCP an interoperability layer, not the only custom tool model.

## Quickstart

### Coding-agent path

Start from the generated project when a coding agent is building with Cayu:

```bash
pip install 'cayu[console]' pytest
cayu new myagent
cd myagent
cayu guide anatomy
cayu guide authoring
cayu inspect --json
cayu check --json
pytest
cayu eval run evals.assistant:build_eval
```

`cayu new` emits a concise vendor-neutral `AGENTS.md`, a process-scoped
application factory, a safe explicit-effect tool, and credential-free runtime
test/eval. The package-shipped [coding-agent authoring guide](src/cayu/guides/authoring.md)
starts from desired behavior and explains when to omit unneeded workflow, task,
environment, approval, memory, server, or multi-agent machinery. The canonical
[application-anatomy guide](src/cayu/guides/application-anatomy.md) defines the
factory, process-role, durable-state, and lifecycle boundaries that every
project follows. Use
`cayu generate slice NAME --tool TOOL --effect EFFECT --dry-run --json` to plan
one reviewable vertical slice without importing or changing the project. Apply
revalidates the plan's preimages and symlink-free project containment, stages
every output, and rolls back the whole slice if any filesystem commit fails.

The default proof distinguishes structural inspection, hermetic runtime tests,
process-boundary checks, and optional live checks. It never treats successful
construction or scripted providers as live provider/environment verification.

### Hand-wired minimal runtime

The minimal ceremony is four calls — create an app, register a provider, register
an agent, and run it. An `Environment` is optional; you only need one when your
agent uses tools that touch a workspace, runner, or sandbox.

```bash
pip install cayu
```

Scaffold a ready-to-edit project with `cayu new myagent`, or use the following
module-global snippet only to understand the four primitive calls. Real projects
should use the generated synchronous `build_app()` factory so imports stay inert
and each process owns a fresh application graph:

```bash
pip install 'cayu[console]'
cayu new myagent
cd myagent
cayu console
```

```python
import asyncio

from cayu import AgentSpec, CayuApp, Message, OpenAIProvider, RunRequest

app = CayuApp()
app.register_provider(OpenAIProvider(), default=True)  # reads OPENAI_API_KEY
app.register_agent(AgentSpec(name="assistant", model="gpt-5.6-luna"))


async def main() -> None:
    async for event in app.run(
        RunRequest(
            agent_name="assistant",
            session_id="demo",
            messages=[Message.text("user", "Say hello in one sentence.")],
        )
    ):
        print(event.type, event.payload)


asyncio.run(main())
```

For a version that runs with **no API key**, see
[`examples/echo_tool_runtime.py`](examples/echo_tool_runtime.py) (a scripted
provider plus a custom tool). To add tools, a filesystem workspace, and command
execution, register an `Environment` — see
[`examples/local_environment_runtime.py`](examples/local_environment_runtime.py).

## Application console

`cayu console` opens an IPython session around a freshly built `CayuApp`, similar
to `rails console`. A project declares its synchronous, zero-argument factory in
`pyproject.toml`:

```toml
[tool.cayu]
factory = "app:build_app"
```

Run the command anywhere below that project root. The console exposes `app`,
`sessions`, `tasks`, `knowledge`, and `cayu`; these are live objects connected to
the application's configured stores. You can also bypass discovery explicitly:

```bash
cayu console package.module:build_app
```

Install the optional dependency with `pip install 'cayu[console]'` or
`uv add 'cayu[console]'`. See [the console guide](docs/console.md) for factory
rules, async behavior, and safety details.

## Getting the result and handling failure

`app.run(...)` returns an event stream and never raises on a model or tool failure —
a failed run ends in a terminal `session.failed` event. To run to completion and get
the answer (or the error) without hand-inspecting events, use `run_to_completion`:

```python
from cayu import run_to_completion

outcome = await run_to_completion(app, request)
if outcome.ok:
    print(outcome.final_text)             # the agent's final answer
else:
    print("run failed:", outcome.error)   # status == "failed"
```

`RunOutcome` carries `status` (a `SessionStatus` with values like
`"completed"` / `"failed"` / `"interrupted"`), `final_text`, `error`, and the
full `events` tuple. If you already hold a loaded transcript,
`final_output_text(transcript)` returns the last persisted assistant text in
that transcript.

## Featured example: cloud PR reviewer

For the full long-running-agent shape — durable triggers, a per-task review
workspace, tool egress, and QA — the
[**cloud PR-reviewer recipe**](docs/recipes/pr-reviewer.md) composes it
end-to-end: a pull request triggers a durable task, a worker checks the PR out
into a fresh workspace, the agent QAs the change by running the test suite, and
it posts one review comment back. Runnable code lives in
[`examples/github_pr_reviewer/`](examples/github_pr_reviewer/); run the no-key demo
with `PYTHONPATH=src python examples/github_pr_reviewer/pr_reviewer.py`.

## Advanced runtime examples

Cayu's advanced examples are executable runtime specifications for behavior that
depends on programmable sessions, checkpoints, forks, authority, recovery, and
durable evidence—not exact model prose.

| Example | Runtime capability |
| --- | --- |
| [Prompt-cache compaction](examples/prompt_cache_compaction/) | Compare a cache-aware compaction with a bounded control from the same compactable source, then keep later compactions bounded to the checkpoint delta. |
| [Cache-aware research council](examples/cache_aware_research_council/) | Compact shared context once, fork several strategies, evaluate them from another branch, and repair the selected result while measuring provider-reported input tokens. |
| [Counterfactual approval](examples/counterfactual_approval/) | Compute authority-free approved and denied futures during a human wait, reject stale state, select one analysis as advisory continuation context, and recover exactly one protected mutation after `CayuApp` reconstruction. |
| [Repository maintainer tournament](examples/repo_maintainer_tournament/) | Replay competing repairs in isolated worktrees, reject test weakening, select one winner, and optionally push exactly one verified PR to a real repository. |
| [Tainted incident response](examples/tainted_incident_response/) | Carry origin-based taint through a generic fork and `CayuApp` reconstruction, block protected authority, and transfer only a sanitized artifact to a clean session. |

Start with the [advanced example developer index](examples/ADVANCED_RUNTIME_EXAMPLES.md)
for commands, evidence contracts, and verification lanes. See
[Advanced runtime strategies](docs/advanced-runtime-examples.md) for the product
story, measured observations, and explicit proof boundaries.
For the optimization portfolio, paired-evidence standard, and enterprise budget
controls, see [Cost optimization and governance](docs/cost-optimization.md).
The current [live Anthropic Haiku benchmark](docs/anthropic-haiku-cost-savings-results.md)
measured 66.17% aggregate savings for prompt-cache compaction and 48.56% across
paired compacted research branch calls, with six of six final trials verified.

## Further reading

User guides:

- [Advanced runtime strategies](docs/advanced-runtime-examples.md) — programmable forks, speculative approval, evaluator branches, real-repository promotion, and taint-aware isolation with measured evidence.
- [Cost optimization and governance](docs/cost-optimization.md) — workload-specific savings strategies, paired evidence, pricing provenance, causal budgets, reservations, and failure cases.
- [Live Anthropic Haiku cost-savings results](docs/anthropic-haiku-cost-savings-results.md) — six paired live trials, exact denominators, quality gates, and current pricing provenance.
- [PR-reviewer recipe](docs/recipes/pr-reviewer.md) — the featured end-to-end example.
- [Business approvals recipe](docs/recipes/business-approvals.md) — multi-tier approve/condition/decline on top of the binary approval primitive.
- [Triggering runs](docs/triggering-runs.md) — which start verb (run / dispatch / task worker / subagent / event watcher) fits your trigger.
- [Environment factories](docs/environment-factories.md) — per-session workspaces, runners, and bindings.
- [Build a runner](docs/build-a-runner.md) — run commands on your own platform.
- [Git command policy](docs/git-command-policy.md) — authorize a narrow local coding-agent workflow.
- [Evals](docs/evals.md) — assertions, trajectories, and reports.

Reference and design docs (deeper, maintainer-facing):

- [Runtime contracts](docs/runtime-contracts.md) — the exhaustive contract reference.
- [Release notes](docs/release-notes.md) — prerelease behavior changes and migrations.
- [Architecture](docs/architecture.md) and [project layout](docs/project-layout.md).
- [Glossary](docs/glossary.md) — naming notes and disambiguations (`Task` vs `asyncio.Task`, `Runner`, the `*Spec` convention…).

## Scope

Cayu's runtime core was extracted from a production agent system used at multiple
mid-size and enterprise companies. The public package includes core contracts,
environment registration, local workspace/runner/artifact-store implementations,
runtime-native file, artifact, command, knowledge recall, and stdio and
Streamable HTTP MCP tool adapters, first-class tool policies for scoped
authority and durable tool approvals, in-memory, SQLite, and Postgres
session/event/transcript stores, explicit session resume, resumable session
interruption, session-level usage/cache summaries, hard token/tool/time run
limits, and session fork with persisted provider/model identity, in-memory,
SQLite, and Postgres task stores, in-memory/SQLite/Postgres knowledge stores,
deterministic knowledge indexing, event sinks and structured runtime logging,
model-provider contracts, model-facing context policies, checkpoint-backed
context compaction, Anthropic Messages API and OpenAI Responses API providers
with certifi-backed TLS verification, structured message/tool-call handling,
tool execution, tool-result feedback to the model, max-step protection,
validation for runtime boundary data, and an optional FastAPI server with a
packaged dashboard for inspecting runs, sessions, tasks, transcripts, events,
and pending knowledge review.

The current public scope is the runtime and integration layer. Hosted deployment
adapters, durable production vector indexes, and higher-level task orchestration
are expected to live in companion packages or application code.

## Contract Rules

Cayu treats payloads, metadata, tool arguments, tool results, model options, checkpoints, task data, and event data as JSON data. These fields must contain JSON-compatible values: objects, arrays, strings, integers, finite floats, booleans, and null. Tuples, arbitrary Python objects, non-string object keys, circular references, NaN, and Infinity are rejected. Task input, result, error, and metadata fields are top-level JSON objects with JSON-compatible nested values.

Runtime objects are copied at runtime boundaries. Mutating an agent, environment, or tool object after registration is not part of the public contract. To change a registered declaration, register a new configuration or use an explicit update API once one exists.

Runtime-native tools receive services through `ToolContext`: workspace, artifact store, runner, vault, credential proxy, knowledge store, and MCP server specs. Those service references are runtime-only and are excluded from serialized context data. Runtime-executed tool calls also receive a stable `ToolContext.idempotency_key` that side-effecting tools can pass to downstream idempotent APIs. `ToolSpec.effect` declares side-effect semantics (`none`, `idempotent`, or `external`) for policy, audit, and future retry decisions; it does not authorize a call by itself.

Tool policies authorize registered tool calls before execution. Denied calls emit `tool.call.blocked`, do not run the tool, and are returned to the model as error tool results so the session can continue. This `tool.call.blocked` contract covers the app-level `ToolPolicy` gate; a tool's *own* internal policy — such as an `ExecCommandTool(policy=...)` `CommandPolicy` denial — instead surfaces as `tool.call.failed` with a structured error, so observability that watches only `tool.call.blocked` will miss command denials.

## Repository Layout

```text
src/cayu/
  core/        framework primitives: events, messages, agents, tools
  workflows/   orchestration-as-code helper primitives
  artifacts/   uploaded/generated file storage contracts
  environments/ execution context contracts
  runtime/     app runtime, sessions, event sinks
  runners/     command execution backends
  workspaces/  filesystem workspace contracts
  storage/     storage contracts and SQLite implementations
  providers/   model provider contracts
  mcp/         MCP client/server integration contracts
  vaults/      secrets access contracts
  proxies/     credential proxy contracts
  cli/         developer/admin CLI
```

`LocalRunner` is for trusted local development. `MicrosandboxRunner` is available
behind the optional `cayu[microsandbox]` extra for local microVM-backed command
execution. It denies guest networking by default; unrestricted networking
requires an explicit Microsandbox policy. The prerelease integration is pinned
to Microsandbox 0.6.6; Cayu does not support older 0.5.x SDK/runtime
combinations. `E2BRunner` and `E2BWorkspace` are available behind the optional
`cayu[e2b]` extra for E2B cloud sandbox execution and native E2B filesystem
access. `LambdaMicroVMRunner` is available behind `cayu[aws]` for AWS-native,
Firecracker-isolated, stateful execution and composes with `RunnerWorkspace`.
`PostgresSessionStore` and `PostgresTaskStore` are production-grade,
multi-worker durable stores behind the optional `cayu[postgres]` extra
(`pip install "cayu[postgres]"`); pass them to `CayuApp(session_store=..., task_store=...)`.

To run commands on your own platform, implement a custom `Runner`: see the
[Build a Runner guide](docs/build-a-runner.md) and the worked
`examples/modal_runner.py` example.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run pytest tests/providers/test_provider_conformance.py -q
uv run pytest tests/workspaces/test_workspace_conformance.py -q
uv run ruff check src/ tests/ examples/ scripts/ maintenance/
uv run ruff format --check src/ tests/ examples/ scripts/ maintenance/
uv run ty check src/cayu examples maintenance
```

The provider and workspace conformance commands are deterministic and
credential-free. They exercise every built-in adapter through its public
interface and make no paid API calls. Use the opt-in checks in
[`docs/nightly-verification.md`](docs/nightly-verification.md) when you need
evidence from real providers and sandboxes; those live checks prove a different
boundary than the CI conformance suites.

Install optional file readers when the built-in `read_file` tool should inspect images/PDFs,
resize oversized images, or extract selected PDF pages:

```bash
uv sync --extra dev --extra files
```

`ReadFileTool` keeps workspace text reads built in and lets applications add
artifact-specific readers, such as OCR, DOCX, or domain-specific invoice parsing, without
rewriting the whole tool. Use `extra_artifact_readers` to extend Cayu's defaults, or
`artifact_readers` when the full artifact-reader chain should be replaced.

Native image/PDF file attachments default to 8 MB per attachment, 32 MB total per
provider request, and 20 attachments per provider request. Override the model-facing
`read_file` caps on `ReadFileTool`, and override final provider-request caps on
`CayuApp`. Built-in context policies keep only the latest file-attachment tool result
provider-resolvable by default, so older attachments stay in the durable transcript as
references/summaries without being resent on every model call.

Configure an artifact store when an agent should inspect uploaded/generated files or
workspace PDFs/images:

```python
from cayu import Environment, EnvironmentSpec, LocalArtifactStore, LocalWorkspace

environment = Environment(
    EnvironmentSpec(name="local"),
    workspace=LocalWorkspace("./workspace", workspace_id="local"),
    artifact_store=LocalArtifactStore("./.cayu/artifacts", store_id="local-artifacts"),
)
```

Use the workspace for mutable work and the artifact store for durable file objects.
For a coding agent, clone the target repo into the sandbox workspace and let file
tools and command tools operate there. For a document or invoice agent, store the
original upload as an artifact, copy it into the workspace only when a path-based
tool or script needs to edit/process it, then store the generated output as a new
artifact:

```python
from cayu import copy_artifact_to_workspace, copy_workspace_file_to_artifact

await copy_artifact_to_workspace(
    artifact_store,
    workspace,
    artifact_id,
    "inputs/invoice.pdf",
)

output = await copy_workspace_file_to_artifact(
    workspace,
    artifact_store,
    "results/invoice-summary.json",
    session_id=session_id,
    agent_name="invoice-agent",
    environment_name="local",
)
```

These copy helpers are explicit one-way operations. There is no hidden sync between
artifact storage and the active workspace.

Attach a knowledge store and register the recall tools when an agent should
explicitly search durable facts, procedures, documents, skills, warnings, or other
reusable context:

```python
from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    ListKnowledgeTool,
    ReadKnowledgeTool,
    SQLiteKnowledgeStore,
    SearchKnowledgeTool,
)

app = CayuApp()
knowledge_store = SQLiteKnowledgeStore("knowledge.sqlite")
environment = Environment(
    EnvironmentSpec(name="local"),
    knowledge_store=knowledge_store,
)

app.register_environment(environment, default=True)
app.register_agent(
    AgentSpec(name="assistant", model="gpt-5.6"),
    tools=[ListKnowledgeTool(), SearchKnowledgeTool(), ReadKnowledgeTool()],
)
```

`list_knowledge` lets the agent discover active entries and facets such as kinds,
labels, aspects, namespaces, and source types before it knows the right search
terms. `search_knowledge` returns bounded previews and entry/chunk ids for active
knowledge; it accepts simple query text plus structured keyword fields
(`any`/`all`/`none`/`phrases`) without exposing backend query syntax.
For large stores, prefer `list_knowledge(group_by=["kind"])` for discovery, then
targeted `search_knowledge`, then `read_knowledge` to expand the selected entry.
The model-facing tool schema uses a list such as `["kind", "aspect", "label"]`;
direct runtime calls may also pass a single facet field string.
Grouped discovery omits entry previews by default; pass `include_entries=true`
only when a small entry sample is useful. `limit` also caps facets per group, and
`facets_truncated=true` means more buckets matched than were returned.
`preview_bytes` controls per-result snippet size; `max_bytes` caps the total tool
payload. Filters such as namespace, labels, kinds, aspects, and source ids are
retrieval hints; tenant/user/project isolation should be enforced by the app or
store wrapper.

`remember_knowledge` is optional. It lets an agent propose a new knowledge entry
through the same store/indexer path, but model-authored entries are stored as
`pending` by default and are excluded from normal recall until reviewed or until
the app registers `RememberKnowledgeTool` with a policy that explicitly allows
active writes. The policy owns the default namespace and required labels; model
inputs are limited to the knowledge text plus optional title, kind, and aspects.
If the app configures `allowed_kinds`, the registered tool schema exposes those
values as the `kind` enum so the model can choose one instead of guessing.
Use `KnowledgeReviewWorkflow` from app/operator code to list pending entries in
an app-owned namespace/label scope, approve them into normal recall, or reject
them as archived. The packaged server/dashboard exposes the same review flow
when the app is constructed with `knowledge_store=...`; configure
`knowledge_review_namespace` and `knowledge_review_labels` on `CayuApp` to limit
which pending entries the dashboard can review.
The tool creates new entries only and enforces an app-configured text-size cap;
edits, archival, deletion, and dedupe/rewrite workflows belong in stricter
app-owned or future tools. See
`examples/knowledge_remember_local.py` for a runnable local policy example.

For semantic recall, Cayu exposes a provider-neutral `TextEmbeddingProvider`
contract. `OpenAIProvider.embed_texts(...)` implements that contract against
OpenAI embeddings, and `InMemoryEmbeddingKnowledgeStore` can use any embedding
provider for opt-in `semantic`, `hybrid`, or `auto` search in tests, demos, and
small single-process apps. `PostgresEmbeddingKnowledgeStore` adds durable
pgvector-backed semantic search for Postgres deployments that install the
`vector` extension and opt into explicit embedding dimensions. Plain
`PostgresKnowledgeStore` and SQLite remain durable keyword stores. Existing
Postgres knowledge can be indexed deliberately with bounded
`backfill_embeddings(..., limit=N)` batches. Pgvector HNSW indexing is created
for dimensions up to 2000; larger vectors still work with exact pgvector search.
Use `examples/postgres_knowledge_embedding.py` for the durable Postgres path:
seed normal Postgres knowledge, create the pgvector-backed store, backfill
existing chunks in a bounded batch, then run semantic or hybrid search. In
production, install pgvector once per database with `CREATE EXTENSION vector`
or run the embedding store with `schema_mode=CREATE` using a role that can create
extensions. Choose `embedding_dimensions` deliberately and keep it stable for a
given embedding table. The example auto-selects dimensions only for OpenAI v3
embedding models; set `CAYU_EMBEDDING_DIMENSIONS` for other OpenAI embedding
models.
Changing model with the same dimensions should be treated as a new indexing run
and handled with bounded `backfill_embeddings(...)` batches. Changing dimensions
requires rebuilding the derived embedding table, because pgvector stores the
dimension in the column type.

For Microsandbox execution, use `MicrosandboxWorkspace` so file tools
read/write/list inside the same sandbox boundary as `exec_command`:

```python
from cayu import Environment, EnvironmentSpec, MicrosandboxRunner, MicrosandboxWorkspace

runner = await MicrosandboxRunner.create("session-123")
environment = Environment(
    EnvironmentSpec(name="sandbox"),
    runner=runner,
    workspace=MicrosandboxWorkspace(runner, workspace_id="sandbox-workspace"),
)
```

The omitted network policy is intentionally deny-all. Trusted applications
that require unrestricted guest networking must pass
`network=microsandbox.Network.allow_all()` explicitly; model-authored code
should instead remain denied or use an enforced virtual-egress policy.

For E2B execution, use `E2BWorkspace` so file tools read/write/list through
E2B's native filesystem API while command tools run in the same cloud sandbox:

```python
from cayu import E2BRunner, E2BWorkspace, Environment, EnvironmentSpec

runner = await E2BRunner.create(
    template="base",
    sandbox_timeout_s=300,
    close_action="kill",
)
environment = Environment(
    EnvironmentSpec(name="e2b"),
    runner=runner,
    workspace=E2BWorkspace(runner, workspace_id="e2b-workspace"),
)
```

Install the optional dependency with:

```bash
pip install "cayu[e2b]"
```

E2B's command API runs command strings through Bash. Cayu preserves process-form
`ExecCommand.process(...)` by shell-quoting argv before sending it to E2B. Use
`ExecCommand.bash(...)` only when shell parsing and expansion are intentional.
The E2B runner does not inherit the trusted host process environment; pass only
explicit command env values that are safe for model-controlled commands.
Likewise, `E2BRunner.create(envs=...)` configures sandbox-level environment
variables. Treat those values as visible to code running in the sandbox, and use
them only for non-secret boot/config values.

For AWS Lambda MicroVM execution, build the sidecar image under
`examples/aws/lambda_microvm_sidecar/`, install `cayu[aws]`, and pass the image ARN explicitly:

```python
from cayu import LambdaMicroVMRunner, RunnerWorkspace

runner = await LambdaMicroVMRunner.create(
    "arn:aws:lambda:us-west-2:123456789012:microvm-image:cayu-runner",
    region_name="us-west-2",
    ingress_network_connectors=[
        "arn:aws:lambda:us-west-2:aws:network-connector:aws-network-connector:ALL_INGRESS"
    ],
    idle_policy={
        "autoResumeEnabled": True,
        "maxIdleDurationSeconds": 900,
        "suspendedDurationSeconds": 300,
    },
)
workspace = RunnerWorkspace(runner)
```

The runner uses Boto3's standard credential chain for control-plane calls and short-lived JWE
tokens for the dedicated MicroVM HTTPS endpoint. Tokens remain in memory and are refreshed on
rejection; they are never reconnect metadata. The first-party image and token are fixed to port
8080, and readiness requires sidecar protocol version `1`. The sidecar preserves process-form
argv, uses an explicit Bash process only for shell-form commands, does not inherit its own
environment into guest commands, bounds output while draining pipes, and kills process groups
before reporting
timeouts or cancellation. The host also enforces the requested command timeout plus a bounded
cleanup grace period if the supervisor remains stuck in `running`. `close_action` accepts
`"terminate"`, `"suspend"`, or `"none"`.

See [AWS credentials for Cayu](docs/aws-credentials.md) for secure developer, coding-agent,
CI, and workload-role setup plus the required Lambda permissions.

For one MicroVM per durable session, use the environment-factory recipe in
`examples/aws/environments/lambda_microvm.py`. It stores only MicroVM identity/region/image metadata,
reattaches on resume, creates a fresh MicroVM for forks, suspends interrupted sessions, and
terminates completed or failed sessions through binding finalization.

Use `SyncBinding` when a durable source workspace should be staged into a
separate bound workspace for one session, then copied back after the run. This is
useful when the active runner has its own filesystem, such as E2B or
Microsandbox, while the app keeps the canonical workspace in local storage, S3,
or another workspace implementation:

```python
from cayu import (
    E2BRunner,
    E2BWorkspace,
    Environment,
    EnvironmentSpec,
    LocalWorkspace,
    SyncBinding,
)

source = LocalWorkspace("./project", workspace_id="project")
runner = await E2BRunner.create(close_action="kill")
target = E2BWorkspace(runner, workspace_id="session-e2b")

environment = Environment(
    EnvironmentSpec(name="project-session"),
    workspace=source,
    runner=runner,
    binding=SyncBinding(target_workspace=target, path="/home/user/workspace"),
)
```

During the run, file tools and command tools operate on the bound workspace. On
finalize, `SyncBinding` copies changed files back to the source workspace and can
also propagate deletions. It does not commit to Git, push branches, or hide
storage policy; applications decide how durable workspaces are backed and when
agent-produced changes should be reviewed or published.
The minimal snippet above uses one explicit target workspace and is suitable for
one live session. For concurrent sessions or per-session sandboxes, use
`target_workspace_factory` or an `EnvironmentFactory` so each session gets a
dedicated bound workspace.
`max_total_bytes` defaults to 64 MiB and bounds the sum of logical file bytes in
each copy-in or copy-back transfer. `max_archive_bytes` separately defaults to
128 MiB and caps the complete raw tar, including headers, padding, and path
metadata. Set smaller values for tighter memory budgets; each accepts `None` as
an explicit opt-out. `max_file_bytes=None` means `SyncBinding` does not add its
own per-file cap, but the workspace being read may still enforce its default
read limit.

Bulk transfers currently use an uncompressed tar archive buffered in memory.
Runner-backed transfers also base64-encode that archive for the runner protocol,
so peak memory is higher than `max_archive_bytes`. Configure `pattern`,
`max_files`, `max_file_bytes`, `max_total_bytes`, and `max_archive_bytes` for
the deployment's expected workspace and memory budget.

Custom `Workspace` implementations must implement `bounded_read_limit()` so a
hard aggregate ceiling composes with, rather than raises, the backend's default
read policy. Bulk export and import are independent nominal capabilities:
subclass `BoundedTarReader` only when the backend can reject both logical and
raw-archive overflow before constructing the tar, and subclass `TarWriter` when
it can import caller-validated tar bytes. `RunnerWorkspace` implements both.
Ad-hoc methods with the same names are ignored. Generic bounded copies are
staged and validated before the first copied-file write, so a byte-limit failure
during finalization does not partially sync changes back to durable storage.

Use `GitRepositoryBinding` when a workspace should start as a checked-out Git
repository:

```python
from cayu import Environment, EnvironmentSpec, GitRepositoryBinding, LocalWorkspace

workspace = LocalWorkspace("./workspaces/project-123")
environment = Environment(
    EnvironmentSpec(name="project"),
    workspace=workspace,
    binding=GitRepositoryBinding(
        repo_url="https://github.com/acme/app.git",
        ref="main",
        path="/workspace",
    ),
)
app.register_environment(environment, default=True)
```

On bind, Cayu clones into an empty workspace or fetches/checks out an existing
clean repository, fast-forwarding to the fetched remote branch when it can do so
without merging or rewriting history. It then records repo/ref/commit metadata in
the binding snapshot. On finalize, Cayu records the final commit and dirty state.
It does not commit, push, create branches, rebase, merge divergent branches, or
create pull requests; those remain explicit agent/tool or trusted app workflows.
Do not put credentials in `repo_url`; the URL is stored in durable metadata, so
HTTP(S) URLs with embedded credentials are rejected. For untrusted sandbox
runners, avoid exposing long-lived Git credentials through generic shell access;
use public repos, trusted host-side credentials, or a dedicated brokered Git tool.

Docker and Docker Sandboxes (`sbx`) do not have dedicated native workspace
adapters. Use `RunnerWorkspace` as the bound target so workspace file operations
execute through the runner:

```python
from cayu import DockerRunner, RunnerWorkspace, SyncBinding

runner = await DockerRunner.create(
    "session-123",
    image="python:3.13-alpine",
)
target = RunnerWorkspace(runner, workspace_id="session-docker")
binding = SyncBinding(target_workspace=target, path=runner.default_cwd)
```

`RunnerWorkspace` requires `python3` inside the guest because file operations run
small Python scripts through the runner. Use a Python image, or install Python
with the runner's setup commands.

The OpenAI provider uses Responses API streaming by default. Tune the ordinary HTTP
timeout and the no-provider-event stall timeout separately:

```python
from cayu import OpenAIProvider

provider = OpenAIProvider(
    timeout_s=600,
    stream_idle_timeout_s=300,
)
```

For OpenAI-compatible services that implement the older Chat Completions API
(`/v1/chat/completions`) rather than the Responses API — Google Gemini (AI Studio), Azure
OpenAI, Together, Fireworks, Mistral, and others — use `ChatCompletionsProvider`. `base_url`
follows the OpenAI-SDK convention (it includes the version path; the provider appends only
`/chat/completions`):

```python
from cayu import ChatCompletionsProvider

provider = ChatCompletionsProvider(
    name="google",
    api_key_env="GEMINI_API_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    document_encoding="image_url",  # Gemini carries PDFs via the image_url part
)
```

It streams text and tool calls, supports image and PDF/document attachments, and strips
JSON-schema keys some vendors reject (e.g. Gemini's `additionalProperties`). PDF encoding is not
portable across vendors, so it is selectable: `document_encoding="file"` (the default) uses the
OpenAI/Azure Chat Completions `file` content part, while `document_encoding="image_url"` carries
the PDF through the `image_url` part, which is what Google Gemini's compatible endpoint accepts.
Native structured output is not supported on this path; structured output still works through the
default tool strategy.

Per-vendor knobs: `auth_header`/`auth_value_prefix` for non-Bearer auth (Azure uses
`auth_header="api-key", auth_value_prefix=""`), `allow_http=True` for local servers such as
Ollama/vLLM on `http://localhost`, `endpoint_url` to override the request URL outright, and
`stream_include_usage=False` for servers that reject `stream_options`.

When an app registers multiple providers, it can route new sessions by model name without making
the caller pass a provider explicitly:

```python
from cayu import AgentSpec, AnthropicProvider, CayuApp, Message, OpenAIProvider, RunRequest

app = CayuApp()
app.register_provider(OpenAIProvider(), default=True, model_patterns=["gpt-*", "o*"])
app.register_provider(AnthropicProvider(), model_patterns=["claude-*"])
app.register_agent(AgentSpec(name="assistant", model="gpt-5.6"))

async for event in app.run(
    RunRequest(
        agent_name="assistant",
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "hello")],
    )
):
    print(event.type)
```

Explicit `RunRequest.provider_name` and `AgentSpec.provider_name` still win. Ambiguous pattern
matches fail before the session is created, and resume/fork paths keep the provider recorded on
the stored session.

To run Claude through **Amazon Bedrock**, install `cayu[aws]` and register an explicit
`BedrockProvider`. It uses Boto3's standard AWS credential chain (including `profile_name=` when
you need a named profile) and Bedrock's `ConverseStream` and `CountTokens` operations; it never
requires or silently falls back to an Anthropic API key. The caller needs
`bedrock:CountTokens` plus `bedrock:InvokeModel` for token counting and
`bedrock:InvokeModelWithResponseStream` for streaming.

See [AWS credentials for Cayu](docs/aws-credentials.md) for local profiles, coding-agent and CI
roles, credential safety rules, and IAM policy templates.

```python
from cayu import AgentSpec, BedrockProvider, CayuApp

app = CayuApp()
app.register_provider(
    BedrockProvider(region_name="us-west-2", profile_name="production"),
    default=True,
)
app.register_agent(
    AgentSpec(
        name="assistant",
        model="us.anthropic.claude-sonnet-4-6",
        provider_name="bedrock",
    )
)
```

Pass the exact Bedrock model ID or inference-profile ARN; Cayu does not rewrite regional model
names. Text, reasoning (including signature/redacted-state round trips), tools/tool results,
image/PDF attachments, tool-strategy structured output, usage, cache counters, finish reasons,
and typed AWS failures use the same runtime contracts as the other built-in providers.
Provider-specific Converse options go under
`provider_options={"bedrock": {...}}`, except `modelId`, `messages`, `system`, and `toolConfig`,
which the adapter owns. Register pricing rows under provider name `"bedrock"`.

`CountTokens` availability is model-specific. Some Claude models offered only through
cross-Region inference require Amazon's separate Bedrock Mantle token-counting endpoint, which
this adapter does not currently call; inference still works, but official preflight token
counting can fail for those models.

To run Anthropic Claude models hosted on **Google Cloud Vertex AI** (enterprises mandated to
GCP), use `VertexProvider` (install the optional `cayu[vertex]` extra). It sends the Anthropic
Messages body to the regional `:rawPredict` endpoint with an OAuth bearer token, so tool calls
and tool-strategy structured output work like the direct Anthropic provider:

```python
from cayu import VertexProvider

provider = VertexProvider(
    project_id="my-gcp-project",
    region="global",  # also accepts regional locations such as "us-east5"
)
```

Authentication uses `google-auth`: pass explicit `credentials=`, a service account via
`service_account_info=`/`service_account_file=`, or (the default) Application Default Credentials
(`gcloud auth application-default login`, a service-account key, or the GCE metadata server). The
model id rides in the URL, so use Vertex ids such as `claude-sonnet-4-6`. The selected model must
also be enabled or requested for the GCP project in Vertex Model Garden. For budget enforcement,
register pricing rows under provider name `"vertex"` (Vertex rates differ from the direct
Anthropic API).

`CayuApp` registers `LoggingEventSink` by default. It writes concise event
summaries to `logging.getLogger("cayu")` without configuring global logging
handlers, process-wide levels, or formatters. Applications control where those
logs go through ordinary Python logging configuration:

```python
import logging

from cayu import CayuApp, LoggingEventSink, SecretRedactor

logging.basicConfig(level=logging.INFO)
app = CayuApp()
```

Disable the default logging sink when an app wants to manage all event sinks
itself:

```python
app = CayuApp(enable_logging=False)
```

When your app has known resolved secrets that may appear in lower-level error
strings, configure the sink with a redactor:

```python
app = CayuApp(
    enable_logging=False,
    event_sinks=[
        LoggingEventSink(redactor=SecretRedactor(["sk-secret-value"])),
    ],
)
```

`SecretRedactor` redacts resolved secret values, not secret reference names or
keys. For example, a reference name such as `sendgrid_api_key` may remain in
metadata, while the resolved value `SG.x...` is replaced with
`[REDACTED_SECRET]` before events, transcripts, provider context, or logs are
persisted/displayed through Cayu's redaction path.

Tools can also use an environment credential proxy when the app wants a trusted
tool to request a scoped credential through a controlled boundary:

```python
from cayu import Environment, EnvironmentSpec, LocalEnvVault, PassthroughProxy

vault = LocalEnvVault({"sendgrid_api_key": "SENDGRID_API_KEY"})
environment = Environment(
    EnvironmentSpec(name="trusted-tools"),
    vault=vault,
    proxy=PassthroughProxy(vault),
)
```

`ctx.proxy.resolve(...)` values are automatically added to the redactor for that
tool result, so accidental leaks from the trusted tool result are redacted before
they reach durable events, transcripts, or the next model request. This is
defense in depth; it does not make generic shell execution safe for secrets.

Trusted tools can also call `ctx.proxy.authorize_request(...)` before using a
credential for an outbound action. Cayu emits a durable
`credential.proxy.checked` event with the destination, credential reference name,
action, metadata, and allow/deny result. This is an audit/enforcement hook for
proxy-aware tools; it does not intercept arbitrary sandbox network traffic.
See `examples/credential_proxy_tool.py` for a runnable trusted-tool example.

## Usage And Cache Metrics

`model.completed` events keep the provider's raw `usage` payload and add a
provider-neutral `usage_metrics` payload when token usage is available. Cayu
normalizes OpenAI cached input tokens and Anthropic cache read/write input tokens
into the same shape, without hiding the original provider data:

```python
summary = await app.get_session_usage("session_123")

print(summary.usage.input_tokens)
print(summary.usage.output_tokens)
print(summary.usage.cache.read_tokens)
print(summary.usage.cache.write_tokens)
```

The optional server exposes the same summary at
`GET /api/sessions/{session_id}/usage`. These metrics are observability data;
budget and stop policies should build on them instead of parsing raw provider
responses.

Cayu can also observe pre-call input token counts when a provider implements
`count_input_tokens(...)`. This is disabled by default and does not enforce
context budgets or mutate model-facing context:

```python
from cayu import CayuApp, ContextCountingConfig

app = CayuApp(context_counting=ContextCountingConfig(mode="observe"))
```

In observe mode, the runtime emits `context.counted` before `model.started`.
When the provider later reports usage, it emits `context.count.reconciled` with
the pre-call count, actual input tokens, and delta. If a provider does not
support counting, the count is reported as unavailable; if counting fails, Cayu
emits `context.count.failed` and still runs the model call.

The built-in OpenAI Responses and Anthropic Messages providers use their
official input-token count endpoints. OpenAI-compatible Chat Completions
providers report unavailable by default because those vendors do not share one
portable count endpoint.

Remote provider counters are an explicit observability tool, not the default
overflow guard. They send the request payload to the provider before the model
call, so applications should treat them as extra provider API calls. Anthropic
documents Messages token counting as free with separate RPM limits. OpenAI
documents Responses input-token counting, but does not document whether that
counting request is free or billed. Cayu records that provider-specific status
in the count result metadata and does not fold count requests into generation
usage or cost totals. Use remote counters for debugging, calibration, live
verification, or explicit near-limit checks where exactness is worth the extra
provider request. Production context management should rely first on local
budget policy estimates, compaction, rolling windows, or truncation, then treat
provider context-limit errors as a recovery path.

For work-item views that span forks or task-linked sessions, use the causal
budget summaries:

```python
usage = await app.get_causal_budget_usage("job_123")
cost = await app.get_causal_budget_cost("job_123", pricing)

print(usage.session_ids)
print(usage.usage.total_tokens)
print(cost.total_cost)
```

The optional server exposes the same grouped views at
`GET /api/causal-budgets/{causal_budget_id}/usage` and
`POST /api/causal-budgets/{causal_budget_id}/cost`. These summaries use the
same normalized usage and caller-supplied pricing as per-session summaries, but
include every session whose stored `causal_budget_id` matches and include
per-session breakdowns for debugging forks.

For a one-call work-item view, use
`POST /api/causal-budgets/{causal_budget_id}/summary`. It accepts the same
pricing body as the cost endpoint and returns included sessions, per-session
outcomes, event counts, grouped usage, and grouped estimated cost.

The raw provider `usage` payload remains available on each durable
`model.completed` event for dashboards, audits, and provider-specific
diagnostics. `usage_metrics` is Cayu's stable summary shape; raw `usage` is the
exact provider payload. If a provider reports usage fields Cayu does not
understand yet, the event still keeps raw `usage` even when `usage_metrics` is
absent.

`model.completed` events also include provider-neutral `completion` metadata and
runtime `step_classification` telemetry. `completion.finish_reason` normalizes
provider stop reasons into values such as `stop`, `tool_calls`, `length`,
`content_filter`, `error`, or `unknown`, while keeping raw provider values beside
it. `step_classification` tells apps whether the assembled assistant step should
be viewed as `continue`, `final`, `length`, `filtered`, `failed`, `think_only`,
or `invalid`. These fields are intended for dashboards, stop policies,
structured-output policies, and future subagent orchestration.

For programmable completion gates, add a before-stop loop policy. It runs only
when the model produced no tool calls and Cayu is about to complete the session:

```python
from cayu import BeforeStopDecision, CayuApp, LoopPolicy, Message


class EmptyAnswerRepairPolicy(LoopPolicy):
    async def before_stop(self, context):
        if context.classification.type == "invalid":
            return BeforeStopDecision.continue_with(
                Message.text("user", "Produce a visible final answer."),
                reason="empty final answer",
            )
        return BeforeStopDecision.complete()


app = CayuApp(loop_policies=[EmptyAnswerRepairPolicy()])
```

Policies can also interrupt or fail the session. Cayu records durable
`custom.loop.before_stop.*` events for configured policies. The runtime does
not ship a built-in goal judge or task gate; those should be app code built on
this seam. Runs with `StructuredOutputSpec` use the structured-output retry and
completion path instead of generic before-stop policies.

Add structured output to a run or resume with `StructuredOutputSpec`. By
default, Cayu injects a runtime-owned final-output tool, validates the submitted
value against your JSON Schema, and emits durable structured-output events:

```python
from cayu import Message, RunRequest, StructuredOutputSpec

request = RunRequest(
    agent_name="assistant",
    messages=[Message.text("user", "Return the invoice status.")],
    structured_output=StructuredOutputSpec(
        name="invoice_status",
        json_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["paid", "unpaid", "unknown"]},
                "confidence": {"type": "number"},
            },
            "required": ["status", "confidence"],
            "additionalProperties": False,
        },
        max_retries=2,
        strategy="tool",
    ),
)
```

When `strategy="tool"` is used, the model sees an internal
`__cayu_submit_structured_output` tool and provider-facing guidance telling it to
call that tool when the final answer is ready. The tool takes one argument,
`output`, whose value must match your schema. Cayu writes a tool result to close
that provider tool round before completing or retrying, so transcript history
remains valid.

`max_retries` defaults to `2`, which allows the initial attempt plus two repair
attempts. If tool-submitted output is invalid, Cayu emits `structured_output.failed`,
returns an error tool result, emits `structured_output.retry` when retries and
model steps remain, and lets the model repair on the next step. If the model
ignores the final-output tool and returns plain final text, Cayu treats that as
a structured-output failure and retries with a durable repair user message when
possible. On success, `structured_output.validated` includes the parsed JSON
output. If retries are exhausted, or no model step remains for repair, the
session fails.

The final-output tool is runtime-owned, not an app-registered tool: it does not
run user code, does not go through tool approval, and does not count against
user tool-call limits. If the model calls it in the same round as other tools,
Cayu rejects the whole round with tool-result errors instead of executing side
effects.

OpenAI can also use provider-native structured output:

```python
StructuredOutputSpec(
    name="invoice_status",
    json_schema={...},
    strategy="native",
)
```

For OpenAI, Cayu maps this to the Responses API `text.format` JSON-schema mode
and still validates the final JSON in the runtime before emitting
`structured_output.validated`. Providers that do not advertise native structured
output raise `NativeStructuredOutputUnsupported` before any session is created
or transitioned — model-pattern provider routing means a model name alone can
select the provider, so catch it to fall back to the portable default:

```python
from cayu import NativeStructuredOutputUnsupported, StructuredOutputStrategy

try:
    async for event in app.run(request):
        ...
except NativeStructuredOutputUnsupported:
    # Nothing ran: no session was created. Retry with the portable strategy
    # (or re-route to a provider with native support).
    request = request.model_copy(
        update={
            "structured_output": request.structured_output.model_copy(
                update={"strategy": StructuredOutputStrategy.TOOL}
            )
        }
    )
    async for event in app.run(request):
        ...
```

The portable `tool` strategy remains the default. Note that OpenAI's strict
JSON-schema mode has requirements beyond ordinary JSON Schema. Its stable
structural core — the root schema is an object, every object sets
`additionalProperties: false` and lists all properties in `required`, and
every `$ref` resolves inside the document, carries no sibling keys beyond
`$defs`/`definitions`, and points at a schema that satisfies the same rules —
is checked before any session is created: violations raise
`NativeStructuredOutputSchemaInvalid` (a `ValueError`) naming the offending
JSON path. OpenAI-side restrictions outside that core are still rejected by
OpenAI's API at request time.

## Thinking And Reasoning

Model reasoning ("thinking") is a first-class, provider-neutral concept. Configure
it with `ThinkingConfig` on an `AgentSpec` (the default for every run) and/or on a
`RunRequest`/`ResumeRequest` (a per-run override that wins over the agent default):

```python
from cayu import AgentSpec, Message, RunRequest, ThinkingConfig

app.register_agent(
    AgentSpec(name="assistant", model="claude-opus-4-8", thinking=ThinkingConfig(effort="high"))
)

# Override the agent default for one run:
request = RunRequest(
    agent_name="assistant",
    messages=[Message.text("user", "Plan the migration.")],
    thinking=ThinkingConfig(effort="low"),
)
```

The mapping to each provider is **field-driven**, so no per-model table is needed
(the request shape differs by model generation):

- `effort` (`"low" | "medium" | "high"`) → Anthropic adaptive thinking
  (`thinking={"type": "adaptive"}` + `output_config={"effort": ...}`) and OpenAI
  `reasoning={"effort": ...}`. This is the path the current Claude and OpenAI
  reasoning models use.
- `max_tokens` (≥ 1024, no `effort`) → Anthropic legacy budgeted thinking
  (`thinking={"type": "enabled", "budget_tokens": ...}`). Only older Claude models
  accept a token budget; OpenAI has no budget knob and ignores it.
- `enabled=False` is best-effort and provider-dependent: Anthropic disables
  (`thinking={"type": "disabled"}`); OpenAI reasoning models cannot be disabled, so it is
  a no-op; the generic Chat Completions adapter also no-ops (disabling isn't portable —
  pass a raw `reasoning_effort` via `provider_options` to target a backend like Gemini
  that accepts `"none"`).

Pick the field appropriate to your model; a mismatch surfaces as a clear provider
`400` rather than being silently corrected. You can still pass raw provider keys via
`AgentSpec.provider_options`; a typed `ThinkingConfig` wins over conflicting raw
thinking/reasoning keys but leaves unrelated keys untouched.

For OpenAI-compatible Chat Completions backends, `effort` maps to `reasoning_effort` and
reasoning is surfaced where the provider emits `reasoning_content` deltas (e.g. DeepSeek,
OpenRouter); Gemini's compatible endpoint accepts the request param but returns reasoning
inlined in the answer, so no separate `ThinkingPart` appears there.

Reasoning content streams as `model.thinking.delta` events and is persisted in the
assistant transcript as a `ThinkingPart`. For Anthropic, the part keeps the opaque
`signature`/`redacted_thinking` data needed to echo the block back verbatim during a
tool-use loop. For OpenAI, the readable reasoning summary is surfaced as a
display-only `ThinkingPart` while the encrypted reasoning still round-trips through
the existing provider-state item. `ThinkingConfig(include_in_transcript=False)` keeps
newly-produced readable reasoning (the OpenAI/Chat Completions display-only summary) out
of the persisted transcript; it does not suppress the live `model.thinking.delta` events,
and an Anthropic signed block is retained verbatim (its signature is needed to continue a
tool-use loop, so that block stays in the transcript). Thinking tokens are billed inside
`output_tokens` and surfaced for visibility as `usage_metrics.reasoning_output_tokens`.

For dashboards, CLIs, and audit views, the optional server exposes paginated
durable events at `GET /api/sessions/{session_id}/events`. It supports
`after_sequence`, `before_sequence`, `order_by`, `limit`, `event_id`, `event_type`,
`exclude_event_type`, `tool_name`, `agent_name`, `environment_name`, and
`workflow_name` query parameters. Inclusion and exclusion filters are applied
before pagination. `event_id` performs an exact lookup within the selected
session; event IDs are session-scoped. Ascending pages use `after_sequence` for
forward polling. Descending pages start at the newest matching event and use
`before_sequence` to load older history. Responses include each event's durable
`sequence` plus `has_more`, `next_sequence`, and `scan_through_sequence`. Pass
`next_sequence`
back using the cursor for the selected order to paginate matching history. A
forward tail reader should instead pass `scan_through_sequence` as its next
`after_sequence`; that watermark can advance across filtered-out events without
skipping a matching event that the response did not return.

For frequent lifecycle polling, use
`GET /api/sessions/{session_id}/state`. It returns only the session id, status,
update/activity timestamps, and interruption-cascade state. The store projects
those fields directly: it does not load session metadata, event history,
transcript messages, usage events, or the full runtime checkpoint. Use the
richer `/summary` endpoint for on-demand metrics rather than as a high-frequency
status heartbeat.

`GET /api/sessions/{session_id}` returns the session's identity, labels, and
full user-authored metadata as one `ApiSession` object. It does not embed event,
transcript, summary, or interruption-cascade data. Clients that previously read
`response.session`, `response.events`, or `response.transcript` should read the
session fields directly and use the bounded `/state`, `/events`, `/transcript`,
and `/summary` endpoints for their respective read models.

The provider-neutral transcript is exposed separately at
`GET /api/sessions/{session_id}/transcript`. It supports `offset`, `limit`, and
`role` filters and returns each message with its zero-based transcript `index`.
Use events to inspect what happened; use transcript to inspect the conversation
state Cayu will use for resume, compaction, and provider requests.

The optional FastAPI server is fail-closed outside local development. Production
apps must pass `auth=...` to `create_server(...)`; `dev=True` is only for trusted
local development and installs an allow-all dependency. Cayu ships `BasicAuth`
for a minimal deployable guard, and applications can pass their own dependency
returning an `AuthContext` for JWT, OIDC, session-cookie, or gateway-backed auth.
FastAPI's generated `/openapi.json`, `/docs`, and `/redoc` routes are enabled by
default only in `dev=True`; protected deployments disable them unless
`expose_docs=True` is set intentionally.

`create_server(...)` mounts the bundled dashboard at `/cayu` and the control
plane at `/api` by default. Set `dashboard_path=None` to expose only the API
routes, pass a custom dashboard path such as `dashboard_path="/inspector"`, or
pass `api_path="/cayu/api"` when dashboard and API should live under one product
path. The dashboard shell, dashboard assets, and control-plane API routes use
the same auth dependency outside `dev=True`.

Mutation endpoints return bounded Server-Sent Events. Runtime data frames carry
stable `session_id:event_id` identifiers; a reconnect that already has such a
marker can send it as `Last-Event-ID` to replay durable events without executing
the mutation again. A caller that chooses `/run`'s optional `session_id` can use
the explicit `session_id:` start marker when the connection fails before its
first event. A mutation does not return HTTP `200` until its first event is
durable, so every accepted response has a replay boundary even when the browser
receives no SSE data. Unknown event markers return `409` rather than being
treated as an implicit full-history request. Before changing an existing
session, resumable clients capture its latest unfiltered durable event as their
replay baseline. Do not retry a mutation POST without either that baseline or
the explicit start marker. The `/api/contract` response advertises the marker
formats and live frame limits. Replay reads durable history in bounded pages
without imposing a connection-wide history ceiling, and a terminal status does
not close replay until the matching durable terminal event has been observed.
Once the detached runtime driver starts, cancellation of the HTTP request drops
only that observer; it does not cancel the accepted operation.
When a slow observer, oversized frame, or idle replay ends the live view, Cayu
emits a classified `stream.error` envelope with
`kind`, `code`, and `retryable` fields. Observer errors do not cancel detached
runtime work or remove durable events; clients should continue through the
bounded session state, event, and transcript endpoints. Terminal stream-error
text is processed by the application's configured secret redactor and bounded
before it is exposed to the authenticated control plane.

Apps that want the bundled dashboard to estimate cost can inject a price book
through `dashboard_config={"priceBook": prices.model_dump(mode="json")}`.
For Cayu's reviewed default snapshot, use
`default_price_book().model_dump(mode="json")` so the configuration is explicitly
JSON-compatible. The dashboard sends that price book to summary endpoints when rendering usage
views. The app still chooses the prices explicitly; Cayu never fetches prices
or changes them at runtime.

Local generated agents should keep their runtime portable: put agent
registration in a `CayuApp` factory and use `create_server(app, dev=True)` only
for the trusted local entrypoint. Production entrypoints should pass real auth.

Composed FastAPI apps can mount CAYU under one product path:

```python
import os

from cayu.server import BasicAuth, mount_cayu

auth = BasicAuth(username="admin", password=os.environ["CAYU_SERVER_PASSWORD"])
mount_cayu(server, cayu_app, path="/cayu", auth=auth)
```

This serves the dashboard at `/cayu` and the control plane at `/cayu/api`.
Existing Next.js apps should usually expose the same shape through a same-origin
route/proxy (`/cayu` and `/cayu/api/...`) so app middleware can guard the product
path while the CAYU server still verifies the upstream auth verdict. For
Cloudflare dashboard plus Fly agent deployments, prefer serving the dashboard at
Cloudflare and proxying `/cayu/api/...` to the Fly agent rather than making the
browser talk cross-origin to Fly; that keeps SSE, credentials, and CORS simpler.

For one-call session health views, the optional server exposes
`GET /api/sessions/{session_id}/summary`. It returns session identity/status,
event totals and counts, the latest event, transcript message count, and the
same normalized usage summary as `/usage`. It also includes a derived
`outcome` object with the current status reason, terminal event, latest retry
event for the latest session invocation, and compact details such as `limit`,
`actual`, `maximum`, `error_type`, or `interruption_type` when those fields
exist in durable events. Estimated cost remains a separate
`POST /api/sessions/{session_id}/cost` call because pricing is supplied by the
application.

When `CayuApp` is constructed with a `knowledge_store`, the optional server also
exposes pending knowledge review for the dashboard: `GET /api/knowledge/pending`
lists pending entries in the configured review scope,
`GET /api/knowledge/pending/{entry_id}` loads one scoped pending entry with its
text and bounded chunks for inspection, and `POST /api/knowledge/{entry_id}/approve`
or `/reject` moves one pending entry to `active` or `archived`.

For a work item that may fork into several sessions, use
`GET /api/causal-budgets/{causal_budget_id}/usage` and
`POST /api/causal-budgets/{causal_budget_id}/cost` to inspect the combined
usage/cost and the session ids included in that causal budget. Use
`POST /api/causal-budgets/{causal_budget_id}/summary` when an app needs the
combined usage/cost plus included session status and outcome data in one call.

Programmatic apps can combine the server and app APIs without the dashboard:

```python
usage = await app.get_session_usage("session_123")
cost = await app.get_session_cost("session_123", pricing)

print(usage.usage.cache.read_tokens)
print(cost.total_cost)
```

Use `/summary` to explain why the session is currently completed, failed, or
interrupted; use `get_session_usage(...)` for normalized token/cache counters;
use `get_session_cost(...)` for caller-priced estimates; use `/events` when you
need the full durable trace. Custom storage or observability integrations can
call `SessionStore.summarize_outcome(...)`, which is the store-level primitive
behind `/summary`.

Prompt cache configuration is provider-specific. Some providers apply caching
automatically when a prompt is long and repeated, while others expose explicit
cache controls, TTLs, or routing hints. Cayu normalizes cache observability, but
does not pretend there is one universal cache-control API. Provider-specific
cache knobs should flow through the provider options for that provider.

```python
from cayu import AgentSpec

app.register_agent(
    AgentSpec(
        name="assistant",
        model="gpt-5.6",
        provider_options={
            "openai": {
                "prompt_cache_key": "tenant-a-agent",
                "prompt_cache_retention": "24h",
            },
            "anthropic": {
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        },
    )
)
```

Cayu blocks provider options that would replace runtime-owned fields such as
messages, tools, `store`, or `previous_response_id`, but it preserves
provider-specific cache controls as request options. OpenAI prompt caching is
mostly automatic and can use `prompt_cache_key` / `prompt_cache_retention`.
Anthropic automatic prompt caching uses top-level `cache_control`. Explicit
Anthropic block-level cache breakpoints are not a Cayu-level abstraction yet;
they require block-level provider metadata and should be added deliberately if
the transcript contract grows that capability.

Provider options configured on `AgentSpec` are copied into each normal model
request for that agent. Ordinary provider-backed compaction uses
`ModelCompactor(options=...)` for its separate request. The cache-aware
`PromptCacheCompactor` instead copies the first runtime request options and
applies its own `options` recursively, because that call must extend the cached
request shape.

Estimate session cost from the same durable usage events with Cayu's bundled,
dated snapshot:

```python
from cayu import default_price_book

prices = default_price_book()
cost = await app.get_session_cost("session_123", prices)

print(cost.total_cost)
print(cost.unpriced_model_steps)
```

The loader is explicit and offline. It reads the snapshot shipped in the installed
Cayu release; it never contacts providers or updates itself while an application is
running. Every price schedule carries its official source URL, verification date, and
validity window. The default price book covers a deliberately small set under provider
keys `openai`, `anthropic`, `google`, `vertex`, and `azure`.
Unknown providers and models remain unpriced rather than being treated as free.

Applications can replace the bundled prices with their own account, region, or
contract prices:

```python
from decimal import Decimal

from cayu import ModelPrice, PriceBook

pricing = PriceBook(
    prices=(
        ModelPrice.fixed(
            provider_name="openai",
            model="gpt-5.5",
            input_per_million=Decimal("2.00"),
            output_per_million=Decimal("8.00"),
            cache_read_input_per_million=Decimal("0.50"),
        ),
    )
)

cost = await app.get_session_cost("session_123", pricing)

print(cost.total_cost)
print(cost.unpriced_model_steps)
```

One-off `ModelPrice` entries default to raw prefix matching, so callers must choose narrow
stable keys or set `match="exact"` for one literal ID. The bundled catalog uses exact
canonical model IDs, exact aliases, and explicit narrow `match_prefixes` for known provider
snapshot forms. For example, the exact `gpt-5.6` alias resolves to `gpt-5.6-sol` without pricing
`gpt-5.6-mini` or `gpt-5.60`. Anthropic date forms use a narrow `-20` convention, Vertex
additionally supports `@20` snapshot IDs, and Google numeric revisions use `-00`. This prefix default is a deliberate
pre-v0.1.0 correction; the matching
algorithm still prefers an exact match and then the longest prefix. Cost estimation walks each durable
`model.completed` event and treats the provider-returned model as authoritative. It uses
the requested model only when the provider reported no resolved model. A pricing entry can
match an exact model name or a provider model-name prefix. Missing pricing is reported as
unpriced line items instead of
being silently treated as free. If cache read/write prices are omitted, Cayu
falls back to the normal input-token price for those counters; provide explicit
cache prices when your provider or account charges them differently.

Provider registration names are part of pricing identity. Gemini-compatible providers must be
registered as `google` to use the bundled Google rows; older applications using `gemini` should
rename the provider or supply custom prices under that name. Model aliases are exact model keys
and do not alias provider names. A custom `PriceBook` is a complete replacement; Cayu does not
implicitly overlay it on the bundled prices.

The optional server exposes the same estimate at
`POST /api/sessions/{session_id}/cost`. The request body supplies the selected
price book:

```json
{
  "pricing": {
    "prices": [
      {
        "provider_name": "openai",
        "model": "gpt-5.5",
        "match": "prefix",
        "schedules": [
          {
            "pricing": {
              "standard": [
                {
                  "input_per_million": "2.00",
                  "output_per_million": "8.00",
                  "cache_read_input_per_million": "0.50"
                }
              ]
            },
            "provenance": {
              "source": "application",
              "url": "application://price-book",
              "as_of": "2026-07-14"
            }
          }
        ]
      }
    ]
  }
}
```

For grouped work-item cost, send the same pricing body to
`POST /api/causal-budgets/{causal_budget_id}/cost`. The response includes
`causal_budget_id`, `session_ids`, `session_count`, and the same estimated cost
fields as session cost summaries, plus `session_costs` for per-session
breakdown.

See [Model catalog and price book](docs/model-catalog.md) for coverage, provenance,
effective dates, budget use, and the reviewed refresh/release process.

For grouped work-item status plus cost, send the same body to
`POST /api/causal-budgets/{causal_budget_id}/summary`.

Run `examples/usage_cost_summary.py` for a deterministic local session report
that emits retry events and prints normalized usage, cache counters, and
estimated cost without calling a real provider API.

The live OpenAI tools example also prints normalized usage/cache counters and an
estimated cost after the run:

```bash
OPENAI_API_KEY=... \
CAYU_OPENAI_INPUT_PER_MILLION=2.00 \
CAYU_OPENAI_OUTPUT_PER_MILLION=8.00 \
CAYU_OPENAI_CACHE_READ_INPUT_PER_MILLION=0.50 \
PYTHONPATH=src python examples/openai_local_tools.py
```

Those environment variables are example pricing inputs only. Use prices from
your own provider account and deployment.

Set hard run limits with `RunLimits` and per-request estimated-cost limits with
`BudgetLimit` on `RunRequest`, `ResumeRequest`, `DispatchRequest`, or
tool-approval continuation requests:

```python
from decimal import Decimal

from cayu import BudgetLimit, Message, RunLimits, RunRequest, default_price_book

prices = default_price_book()

request = RunRequest(
    agent_name="assistant",
    messages=[Message.text("user", "Analyze these invoices.")],
    limits=RunLimits(
        max_total_tokens=50_000,
        max_tool_calls=25,
        max_elapsed_seconds=300,
        scope="session",
    ),
    budget_limits=(
        BudgetLimit(
            scope="session",
            max_estimated_cost=Decimal("0.50"),
            pricing=prices,
        ),
    ),
)
```

Token and tool-call limits are evaluated from durable session events, so they
apply across resume and dispatch paths by default. `scope="session"` is the
default and treats token/tool-call limits as lifetime session budgets.
`scope="run"` evaluates token/tool-call limits against only the current
`run(...)`, `resume(...)`, dispatch, or approval-continuation invocation. Elapsed
time is always evaluated for the active runtime invocation and resets on each
call. Estimated-cost budget limits use the same scope names: `scope="session"`
enforces a lifetime estimated-cost budget, while `scope="run"` compares only
estimated cost added during the current invocation.

Budget limits are estimates derived from normalized usage metrics and the
price book supplied by your app. `ModelCatalog` is metadata-only and is not accepted by cost or
budget APIs. Estimates are not provider invoices. By
default, request-scoped interrupt budgets fail closed when a newly observed
model step has no matching pricing entry, because Cayu cannot prove that the
budget is still safe. Request-scoped notify budgets emit `budget.limit_reached`
for the same unverifiable usage and continue. Set `allow_unpriced=True` only
when your app intentionally allows missing prices for that run.

Request `budget_limits` can also use `scope="agent"` or `scope="causal"` when a
caller needs dynamic spend control for one API call or work item without
changing the app's global policy. The `key` must match the current agent name
for `agent` limits or the current `causal_budget_id` for `causal` limits.
`scope="app"` is accepted for deliberate per-request global checks, but app-wide
limits usually belong in `BudgetPolicy`.

`BudgetLimit.action` defaults to `"interrupt"`. With `action="notify"`, Cayu
emits a durable `budget.limit_reached` event when the threshold is reached, but
does not emit `session.limit_reached`, does not interrupt the session, and does
not close pending tool rounds. Use notify budgets for alerts and dashboards; use
the default interrupt action for enforcement.

When an interrupt limit is reached, Cayu emits `session.limit_reached`, marks
the session `interrupted`, emits `session.interrupted` with
`interruption_type="limit_reached"`, and leaves the session resumable. Resuming
with the same exhausted session-scoped token or tool-call/cost budget will
interrupt again immediately; pass a higher budget, omit that limit, or use
`scope="run"` if "continue" should mean "give this invocation a fresh
token/tool/cost budget."
In a limit event, `actual` is the value evaluated for the selected scope, while
`usage_summary` remains the cumulative session summary. Cost-limit events also
include the cumulative `cost_summary`; decimal cost values are serialized as
strings for JSON stability. If the model requested tools in the same step, Cayu
records skipped tool results before interrupting so the provider-neutral
transcript remains valid for resume.

For app-level spend control across sessions, configure a `BudgetPolicy` on
`CayuApp`. Budget windows default to all-time accounting, and can also use
rolling duration windows or local calendar reset windows for app-wide,
agent-scoped, and causal estimated-cost limits:

```python
from decimal import Decimal

from cayu import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    BudgetWindow,
    CayuApp,
    SQLiteBudgetLedger,
)

app = CayuApp(
    budget_policy=BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("25.00"),
                pricing=pricing,
            ),
            BudgetLimit(
                scope="agent",
                key="assistant",
                max_estimated_cost=Decimal("5.00"),
                window=BudgetWindow.rolling(seconds=3600),
                pricing=pricing,
            ),
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("80.00"),
                window=BudgetWindow.calendar(period="day", timezone="America/New_York"),
                pricing=pricing,
                action="notify",
            ),
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("100.00"),
                window=BudgetWindow.calendar(period="day", timezone="America/New_York"),
                pricing=pricing,
            ),
            BudgetLimit(
                scope="causal",
                key="job_123",
                max_estimated_cost=Decimal("2.50"),
                pricing=pricing,
            ),
        )
    )
)
```

App budgets use the same caller-supplied price book as request
`BudgetLimit` entries. Rolling windows are UTC timestamp duration windows over
durable model events, for example "the last hour." Calendar windows evaluate the
current local `day`, `week`, or `month` for an explicit IANA timezone; days reset
at local midnight, weeks start on Monday, and months start on the first day of
the month. Rolling and calendar windows can be used together as separate
`BudgetLimit` entries when an app needs both spend-velocity protection and
daily/monthly accounting.
Before each model step and after each completed model step, Cayu evaluates the
matching budget limits, verifies that the current requested provider/model has
pricing unless `allow_unpriced=True`, and emits `budget.checked`. If an interrupt budget
is reached, Cayu stops with `budget.limit_reached` plus the normal
`session.limit_reached` / `session.interrupted` events. If a notify budget is
reached, Cayu emits `budget.limit_reached` with `action="notify"` and continues.
App-policy notify events are emitted once per matching threshold/window; later
`budget.checked` events continue to show the above-limit state. `scope="app"`
applies to all sessions. `scope="agent"` applies when `key` matches the agent name.
`scope="causal"` applies when `key` matches `RunRequest.causal_budget_id`.
If omitted, a root session's `causal_budget_id` defaults to `task_id` when the
run is linked to a task, otherwise to its session id. Forked sessions inherit
the source session's causal budget id, so a parent run and its children can
share one work-item budget. The session remains resumable, but resuming under
the same exhausted matching budget will stop again until the app changes the
policy, raises the limit, fixes missing pricing, or intentionally allows
unpriced usage.

`CayuApp` uses `SessionBudgetStore` by default, so budget accounting reads from
the same event store already configured for sessions, including timestamp
filters for rolling and calendar windows. With `SQLiteSessionStore`, budget
accounting survives process restarts and multiple workers that share the same
database. Enforcement is cooperative: Cayu checks before model calls and again
after model completions.

For strict concurrent hard caps, add a conservative per-step reservation and a
shared ledger. Cayu reserves the configured worst-case step cost before the
provider call, reconciles it to actual normalized usage after
`model.completed`, and refuses the step before calling the provider if the
reservation would exceed the limit:

```python
app = CayuApp(
    budget_policy=BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("25.00"),
                pricing=pricing,
                reservation=BudgetReservation(
                    max_input_tokens=80_000,
                    max_output_tokens=8_000,
                    max_cache_read_input_tokens=80_000,
                ),
            ),
        )
    ),
    budget_ledger=SQLiteBudgetLedger("budget.sqlite"),
)
```

Reservation amounts are application-provided upper bounds, not provider
guarantees. Set them high enough for the model step you are willing to fund.
For context-tiered catalogs, Cayu selects the reservation's pricing tier from the
sum of `max_input_tokens`, `max_cache_read_input_tokens`, and
`max_cache_write_input_tokens`, then charges those categories at their respective
rates. Output tokens are charged by the selected tier but do not select an input-context
tier.
Reservation limits require matching pricing and cannot use `allow_unpriced=True`.
Reservation limits also require `action="interrupt"` because reservations are
hard-cap accounting, not observe-only alerts.
With rolling or calendar budget windows, unresolved active reservations continue
to consume capacity until they are reconciled or released; reconciled spend ages
out by the reconciliation/model-completion timestamp.
Built-in ledgers renew active reservations every one-third of their configured
TTL while a provider step remains live, so a slow or silent call does not age out
of the hard cap. The default TTL is one hour. Cayu validates the lease before
starting provider work, so a reservation that expires during a paused event
stream cannot start an unprotected call. If renewal is lost, Cayu fails the step
closed and charges known actual usage or, when the outcome is uncertain, the full
reserved amount. `reservation_ttl_seconds=None` disables automatic expiry
and therefore requires manual cleanup after a crashed worker. Processes sharing
a durable ledger must use the same TTL and synchronized clocks.
`SQLiteBudgetLedger` is the built-in shared ledger for multi-worker apps.
`InMemoryBudgetLedger` is the default and is suitable for tests, examples, and
single-process apps only. `InMemoryBudgetStore` is also available for custom
single-process apps that intentionally want separate in-memory budget
accounting, but causal budgets require the session-aware `SessionBudgetStore`
because they depend on persisted session identity.

Configure provider-step retries with `RetryPolicy` on `CayuApp` or on one
request. Retries are disabled by default. A retry only wraps the model provider
request before any assistant transcript message is appended and before any tool
executes:

```python
from cayu import CayuApp, Message, RetryPolicy, RunRequest

app = CayuApp(retry_policy=RetryPolicy(max_attempts=3))

request = RunRequest(
    agent_name="assistant",
    messages=[Message.text("user", "Research this domain.")],
    retry_policy=RetryPolicy(
        max_attempts=2,
        initial_delay_s=0.5,
        max_delay_s=3.0,
        retry_on_status_codes=(429, 500, 502, 503, 504, 529),
    ),
)
```

Retryable provider failures emit `model.error`, then `model.retry`, then a new
`model.started` attempt. When retries are enabled, provider-derived model events
include `step`, `attempt`, and `max_attempts` so live logs and dashboards can
separate failed-attempt output from successful output. Cayu does not retry tool
execution or any model step after tool side effects have started. Built-in
classification treats provider overload, server errors, timeouts, connection
failures, and rate-limit errors as retryable. Permanent quota/billing failures
are not retried even when a provider reports them with HTTP 429.

## Example

### Running the bundled examples

Run the deterministic echo-tool runtime example:

```bash
PYTHONPATH=src python examples/echo_tool_runtime.py
```

Run a local environment example with a filesystem workspace and local command runner:

```bash
PYTHONPATH=src python examples/local_environment_runtime.py
```

Run a deterministic stdio MCP example:

```bash
PYTHONPATH=src python examples/stdio_mcp_runtime.py
```

A remote (Streamable HTTP) MCP server uses a `url` instead of a `command`.
Credentials flow through `secret_headers` (never inlined); pass an HTTP client
with a secret resolver so those headers can be resolved:

```python
from cayu import HttpMcpClient, LocalEnvVault, McpServerSpec, SecretRef, connect_mcp_toolset

vault = LocalEnvVault({"github_token": "GITHUB_TOKEN"})
toolset = await connect_mcp_toolset(
    McpServerSpec(
        name="github",
        url="https://api.githubcopilot.com/mcp/",
        secret_headers={"Authorization": SecretRef(name="github_token")},
    ),
    client=HttpMcpClient(secret_resolver=vault),
)
# toolset.tools -> pass to register_agent(...); await toolset.close() when done.
```

This is how an agent gets GitHub (or any HTTP MCP server) tools with no custom
`Tool` code. See the [PR-reviewer recipe](docs/recipes/pr-reviewer.md) for an
end-to-end example.

MCP tool/resource discovery is bounded by default to 100 pages and 10,000
items. Both `HttpMcpClient` and `StdioMcpClient` accept positive integer
`max_list_pages=` and `max_list_items=` overrides for servers with different
manifest sizes.

`connect_mcp_toolset(...)` fingerprints the discovered MCP tool contract as
`toolset.manifest_hash`. The same value is exposed on each `McpToolAdapter` as
`mcp_manifest_hash` and is included in structured MCP tool results for audit and
debugging. A changed MCP input schema, tool description, annotations, server
instructions, or generated Cayu tool name changes the hash; the hash is not an
authorization decision. When an agent run exposes MCP tool adapters, Cayu also
emits a durable `mcp.manifest.checked` event before the model step. The event
records whether the server manifest is `first_seen`, `unchanged`, or `changed`
against prior durable events for the same server/environment and includes a
compact added/removed/changed tool diff. Comparison uses a stable
`manifest_identity` for the exposed toolset, so distinct MCP toolsets with the
same server name are audited separately.

Apps that want to enforce MCP tool drift can configure `McpManifestPolicy`:

```python
app = CayuApp(
    mcp_manifest_policy=McpManifestPolicy(
        on_first_seen="allow",
        on_unchanged="allow",
        on_changed="block",
        on_tools_added="block",
        on_tools_removed="alert",
    )
)
```

`allow` continues normally, `alert` records the policy decision on
`mcp.manifest.checked`, and `block` emits `mcp.manifest.blocked` then fails the
session before the changed tools are sent to the provider. Without a configured
policy, manifest checks remain audit-only.

Run the live Anthropic example with local tools:

```bash
export ANTHROPIC_API_KEY=...
PYTHONPATH=src python examples/anthropic_local_tools.py
```

Run the live OpenAI example with local tools:

```bash
export OPENAI_API_KEY=...
PYTHONPATH=src python examples/openai_local_tools.py
```

Run the workflow helpers example without credentials, or switch it to live OpenAI:

```bash
PYTHONPATH=src python examples/workflow_helpers.py
CAYU_WORKFLOW_PROVIDER=openai OPENAI_API_KEY=... PYTHONPATH=src python examples/workflow_helpers.py
```

Run the live OpenAI-compatible Chat Completions example with local tools (shown with Google
Gemini; point it at Azure/Together/Fireworks/Mistral by changing the provider's `base_url` and
`api_key_env`):

```bash
export GEMINI_API_KEY=...
PYTHONPATH=src python examples/chat_completions_local_tools.py
```

Run the live Vertex AI (Claude on Google Cloud) example with local tools (needs
`pip install cayu[vertex]` and `gcloud auth application-default login`):

```bash
export GOOGLE_CLOUD_PROJECT=my-gcp-project
PYTHONPATH=src python examples/vertex_local_tools.py
```

Run the live artifact/file example with image or PDF attachments:

```bash
uv sync --extra dev --extra files

export OPENAI_API_KEY=...
CAYU_PROVIDER=openai CAYU_ARTIFACT_KIND=image PYTHONPATH=src python examples/artifact_file_live.py
CAYU_PROVIDER=openai CAYU_ARTIFACT_KIND=pdf PYTHONPATH=src python examples/artifact_file_live.py

export ANTHROPIC_API_KEY=...
CAYU_PROVIDER=anthropic CAYU_ARTIFACT_KIND=image PYTHONPATH=src python examples/artifact_file_live.py
CAYU_PROVIDER=anthropic CAYU_ARTIFACT_KIND=pdf PYTHONPATH=src python examples/artifact_file_live.py
```

Run the deterministic artifact/workspace bridge example:

```bash
PYTHONPATH=src python examples/artifact_workspace_bridge.py
```

Run gated E2B examples with a real E2B account:

```bash
E2B_API_KEY=... PYTHONPATH=src python examples/e2b_runner_live.py
E2B_API_KEY=... PYTHONPATH=src python examples/e2b_workspace_live.py
```

Use durable local session/event/transcript storage:

```python
from pathlib import Path

from cayu import CayuApp, EventQuery, SQLiteSessionStore

store = SQLiteSessionStore(Path(".cayu") / "sessions.sqlite")
app = CayuApp(session_store=store)

async def inspect_session(session_id: str):
    events = await store.query_events(EventQuery(session_id=session_id))
    transcript = await store.load_transcript(session_id)
    return events, transcript
```

Process durable events with an app-side watcher:

```python
from pathlib import Path

from cayu import (
    CayuApp,
    EventQuery,
    EventType,
    EventWatcher,
    EventWatcherContext,
    SQLiteEventWatcherStore,
    SQLiteSessionStore,
)

db = Path(".cayu") / "runtime.sqlite"
app = CayuApp(
    session_store=SQLiteSessionStore(db),
    event_watcher_store=SQLiteEventWatcherStore(db),
)

async def send_budget_alert(context: EventWatcherContext) -> None:
    event = context.record.event
    # Use (context.watcher_name, event.id) as the external idempotency key.
    await send_email(
        subject="Cayu budget threshold reached",
        body=f"{event.session_id}: {event.payload}",
    )

budget_alerts = EventWatcher(
    name="budget-alert-email",
    query=EventQuery(event_type=EventType.BUDGET_LIMIT_REACHED),
    handler=send_budget_alert,
)

await app.run_event_watchers([budget_alerts])
```

### Event watchers

Event watchers are not model-facing tools. They are trusted app code that pulls
from the durable event log, records cursor/attempt state, retries failures, and
dead-letters events after the configured attempt ceiling.
Use `InMemoryEventWatcherStore` for tests/single-process examples,
`SQLiteEventWatcherStore` for durable local apps, and
`PostgresEventWatcherStore` for hosted multi-worker deployments.

Use durable local task storage for optional background work tracking:

```python
from pathlib import Path

from cayu import SQLiteTaskStore, TaskCreate, TaskQuery

tasks = SQLiteTaskStore(Path(".cayu") / "runtime.sqlite")
task = await tasks.create_task(
    TaskCreate(
        type="process_invoice",
        input={"invoice_id": "inv_123"},
        assigned_agent_name="invoice_agent",
    )
)
```

Link a run to an existing task when runtime status should follow the session:

```python
from cayu import CayuApp, Message, RunRequest

app = CayuApp(session_store=store, task_store=tasks)
request = RunRequest(
    agent_name="invoice_agent",
    task_id=task.id,
    messages=[Message.text("user", "Process this invoice.")],
)

async for event in app.run(request):
    print(event.type)
```

For queue-style workers, claim an unattached pending task before starting the
agent run. The task is `claimed` while the worker owns the lease, then Cayu
attaches it to the new session and marks it `running` when the run starts. Pass
both the task id and worker id to the run request:

```python
task = await tasks.claim_task(
    "worker-a",
    TaskQuery(type="process_invoice", assigned_agent_name="invoice_agent"),
    lease_seconds=300,
)

if task is not None:
    await tasks.heartbeat(task.id, "worker-a", extend_seconds=300)
    async for event in app.run(
        RunRequest(
            agent_name=task.assigned_agent_name or "invoice_agent",
            task_id=task.id,
            task_worker_id="worker-a",
            messages=[Message.text("user", "Process the claimed invoice task.")],
        )
    ):
        print(event.type)
```

Use direct `RunRequest.task_id` when app code already knows the exact task to
run. Use `claim_task(...)` when multiple workers compete for pending work.
Claiming only applies to unattached pending tasks; once a task is attached to a
session, session recovery and terminal runtime events own the outcome.

### Resuming a session

Resume an existing completed, failed, or interrupted session by appending new messages to its durable transcript:

```python
from cayu import Message, ResumeRequest

resume_request = ResumeRequest(
    session_id="sess_123",
    messages=[Message.text("user", "Continue from the previous result.")],
)

async for event in app.resume(resume_request):
    print(event.type)
```

New sessions use the registered agent's default model. Resume uses the session's active model by default, and can durably update that active model for future resumes while keeping the stored provider fixed:

```python
resume_request = ResumeRequest(
    session_id="sess_123",
    messages=[Message.text("user", "Continue with the larger model.")],
    model="gpt-5.6",
)
```

### Interrupting a session

Interrupt a pending or running session through the runtime. Interruption is
durable and resumable after finalization: Cayu first marks the session
`interrupting`, signals active runtime work for that session in the current
`CayuApp` process, repairs any in-progress tool round if needed, then marks the
session `interrupted` and emits `session.interrupted`. You can later continue
the same session with `ResumeRequest` after the `session.interrupted` event.
If the current process does not own the active run, or the active run is still
stopping and repairing its transcript, `interrupt_session(...)` reports that
interruption is still finalizing and leaves the session `interrupting`. The
caller should retry, poll the session/event store, or subscribe to events until
`session.interrupted` is durable.
Every `session.interrupted` event includes a normalized `interruption_type`:
`operator_requested` for explicit `InterruptSessionRequest` calls,
`tool_approval_required` for approval pauses, and `runtime_interrupted` for
runtime/status-driven interruption repairs.

On worker startup, recover sessions that were left incomplete by a deploy,
process crash, or machine restart:

```python
from cayu import IncompleteSessionsRecoveryRequest, SessionStatus

results = await app.recover_incomplete_sessions(
    IncompleteSessionsRecoveryRequest(
        statuses={SessionStatus.INTERRUPTING},
        reason="worker restart",
    )
)
for result in results:
    print(result.session_id, result.status, result.actions)
```

Recovery does not call the model and does not execute tools. It repairs any
pending ordinary tool round from durable terminal tool events when possible,
preserves pending tool approvals for `resolve_tool_approval(...)`, finalizes
stale `interrupting` sessions, and marks abandoned `pending`/`running` sessions
as `interrupted` so they can be resumed deliberately later. Batch recovery
requires explicit `statuses`; supported values are `interrupting`, `running`,
and `pending`. Include `running` or `pending` only when your app knows those
in-flight sessions are abandoned, such as a single-worker restart.

Provider streams and runner/tool calls that are currently awaited are stopped
through normal asyncio cancellation. `E2BRunner` and `MicrosandboxRunner`
separate user/runtime interruption from command timeout:

- `cancellation_cleanup` defaults to `"command"`. It asks the
  provider command handle to kill only the current command and keeps the sandbox
  reusable when that succeeds. This preserves interactive coding workspaces
  after an operator interrupt.
- `timeout_cleanup` defaults to `"command"`. It also asks the provider
  command handle to kill only the current command, so timeout handling preserves
  workspace state by default. Apps that prefer a stronger cleanup boundary can
  set it to `"sandbox"`.

Cleanup waits are bounded by `cancel_timeout_s`; cleanup failures are surfaced
as `cayu.runner_cleanup.v1` diagnostics on interrupted tool results or timed-out
`ExecResult.artifacts`. If E2B has not yet returned a command handle when
interruption or timeout arrives, Cayu first tries to stop the start attempt and
resolve the handle within the cleanup window. With `"command"` cleanup, Cayu
reports deferred cleanup and continues waiting in the background for a bounded
adapter-owned window. If that delayed command start never resolves, Cayu
preserves the sandbox but closes that runner's exec path so later commands do
not overlap with an unknown command state. With `"sandbox"` cleanup, Cayu kills
the sandbox immediately because the sandbox is the configured cleanup boundary.
Apps that need a stronger cleanup boundary can set either cleanup field to
`"sandbox"`, or use `"none"` when they own cleanup outside Cayu. Interrupting a
session does not automatically cancel a linked task; task state remains
application/workflow owned.

```python
from cayu import InterruptSessionRequest

async for event in app.interrupt_session(
    InterruptSessionRequest(
        session_id="sess_123",
        reason="operator requested stop",
        metadata={"actor": "operator"},
    )
):
    print(event.type)
```

### Forking a session

Fork a completed, failed, or interrupted session to create a new branch without mutating the source session:

```python
from cayu import ForkSessionRequest

async for event in app.fork_session(
    ForkSessionRequest(
        source_session_id="sess_123",
        session_id="sess_branch_a",
        metadata={"purpose": "try alternate plan"},
    )
):
    print(event.type)
```

Fork copies the provider-neutral transcript and, by default, the checkpoint state. A partial transcript fork can copy messages up to a 1-based transcript cursor, but must set `copy_checkpoint=False` because checkpoint state may refer to transcript content that was not copied:

```python
ForkSessionRequest(
    source_session_id="sess_123",
    session_id="sess_branch_from_first_message",
    transcript_cursor=1,
    copy_checkpoint=False,
)
```

Dispatch work for an existing session. Fork creates a branch; dispatch submits session work through a pluggable execution backend. The default `InlineDispatcher` runs immediately in the current process and returns a handle:

```python
from cayu import DispatchRequest, Message

handle = await app.dispatch(
    DispatchRequest(
        session_id="sess_branch_a",
        messages=[Message.text("user", "Run this follow-up objective.")],
        task_id=task.id,
    )
)
print(handle.status)
```

For local streaming execution, use `dispatch_inline(...)`:

```python
async for event in app.dispatch_inline(
    DispatchRequest(
        session_id="sess_branch_a",
        messages=[Message.text("user", "Run this follow-up objective.")],
    )
):
    print(event.type)
```

For model-facing delegation, register a `SubagentTool`. A subagent call creates
a normal child session with `parent_session_id` and the same `causal_budget_id`.
A subagent runs against the parent's `environment_name`; to give it a
differently-shaped environment (say, a checkout for a QA child but none for the
orchestrator), branch a single `EnvironmentFactory` on `request.agent_name` — see
[`examples/environments/per_agent_environment.py`](examples/environments/per_agent_environment.py).
Foreground subagents run to completion and return the child result as a tool
result to the parent. Background subagents start the child session, return the
`child_session_id` immediately, and keep running in the active runtime process.
Register `SubagentResultTool` when the parent model should later fetch one
child result or wait for all background children it started.
`SubagentSpec.result_max_chars` bounds foreground child text copied back into the
parent transcript. The initial context mode is `task_only`: the child receives
the delegated task, not a full copy of the parent transcript. Child events are
ordinary durable session events: observe them through event sinks or session
queries by `parent_session_id`.

```python
from cayu import (
    AgentSpec,
    CayuApp,
    SubagentExecutionMode,
    SubagentResultTool,
    SubagentSpec,
    SubagentTool,
)

app = CayuApp()

subagents = SubagentTool(
    app,
    agents={
        "reviewer": SubagentSpec(
            agent_name="security_reviewer",
            description="Review implementation risks.",
            result_max_chars=8000,
        ),
        "background_reviewer": SubagentSpec(
            agent_name="security_reviewer",
            description="Review implementation risks without blocking the parent.",
            mode=SubagentExecutionMode.BACKGROUND,
        )
    },
)

app.register_agent(
    AgentSpec(name="builder", model="gpt-5.6"),
    tools=[subagents, SubagentResultTool(app.session_store)],
)
app.register_agent(
    AgentSpec(
        name="security_reviewer",
        model="gpt-5.6",
        system_prompt="Review delegated work and return concrete risks only.",
    )
)
```

Add runtime hooks for lifecycle automation. Hooks run after terminal session state is already durable, so they are useful for follow-up work such as extracting knowledge from a completed builder session:

```python
from cayu import AgentSpec, DispatchRequest, ForkSessionRequest, Message, RuntimeHook, TaskCreate


class KnowledgeHook(RuntimeHook):
    async def after_session_completed(self, context):
        if context.session.metadata.get("purpose") == "knowledge_extraction":
            return

        child_id = f"{context.session.id}_knowledge"
        await context.fork_session(
            ForkSessionRequest(
                source_session_id=context.session.id,
                session_id=child_id,
                metadata={"purpose": "knowledge_extraction"},
            )
        )
        task = await context.create_task(
            TaskCreate(type="knowledge_extraction", session_id=child_id)
        )
        await context.dispatch(
            DispatchRequest(
                session_id=child_id,
                task_id=task.id,
                messages=[Message.text("user", "Extract implementation notes.")],
            )
        )


app.register_agent(
    AgentSpec(name="builder", model="claude-sonnet-4-6"),
    runtime_hooks=[KnowledgeHook()],
)
```

Hooks can also observe completed, failed, and blocked tool calls:

```python
from cayu import RuntimeHook


class ToolAuditHook(RuntimeHook):
    async def after_tool_call(self, context):
        await context.emit_custom_event(
            "custom.tool.audit",
            payload={
                "tool_name": context.tool_name,
                "tool_call_id": context.tool_call_id,
                "is_error": context.result.is_error,
            },
        )
```

`after_tool_call` runs after Cayu has persisted the tool result event. It is for auditing, memory extraction, follow-up tasks, and observability; it does not rewrite the tool result that is appended to the model transcript.

Customize model-facing context without rewriting durable transcript history:

```python
from cayu import (
    AgentSpec,
    RecentTurnsContextPolicy,
)

app.register_agent(
    AgentSpec(name="assistant", model="claude-sonnet-4-6"),
    context_policy=RecentTurnsContextPolicy(max_user_turns=10),
)
```

Use `UsageTriggeredContextPolicy` to react to actual provider usage from the
previous completed model call in the same session. This is post-call state, not
a remote pre-call counter: the next request can switch to a smaller projection
after the prior call reported high `usage_metrics.input_tokens` or total tokens.
When the previous `model.completed` event includes a transcript cursor, Cayu also
exposes `context_usage.input_pressure`, which estimates current context pressure
as previous actual input tokens plus a local estimate for messages appended after
that cursor. For `UsageTriggeredContextPolicy`, the estimated trigger is computed
against the base policy's model-facing request shape before switching: context
messages after the base policy has run, known tool schemas, structured-output
wiring, provider-visible options, tool-call arguments, tool-result text,
thinking/provider-state parts, and conservative file-attachment size estimates. Set
`trigger_estimated_context_tokens` with `reserved_output_tokens` when you want a
soft proactive threshold based on `estimated_input + output_reserve`; it is not
billing data or an exact final provider-request context-window guarantee. Cayu
uses provider pressure profiles for local calibration of image attachments,
document/PDF attachments, and tool-schema payloads; the runtime consumes those
profile hints without branching on provider names. Cayu emits
`context.pressure.estimated` before the model call and
`context.pressure.reconciled` after `model.completed` when context counting
observe mode is enabled, so applications can compare the local input estimate
against provider-reported `actual_input_tokens`. If exact pre-call verification
is worth an extra provider request, set
`verify_estimate_with_provider_count=True`; Cayu then calls
`provider.count_input_tokens(...)` only when the local estimate is near the
estimated threshold or when `provider_count_min_delta_tokens` marks the new
estimated delta as large. The provider count, when available, decides the
estimated trigger; otherwise Cayu falls back to the local estimate.
Estimated triggers evaluate `base_policy` before deciding whether to switch, so
`base_policy` must be side-effect-free; runtime-managed base policies such as
checkpoint compaction are rejected for this mode. The trigger is sticky by
default: once a threshold is crossed, Cayu stores a session
checkpoint marker and keeps using
`triggered_policy` for later requests in that session. Set `sticky=False` only
when you explicitly want last-call-only routing.

```python
from cayu import (
    AgentSpec,
    RecentTurnsContextPolicy,
    UsageTriggeredContextPolicy,
)

app.register_agent(
    AgentSpec(name="assistant", model="claude-sonnet-4-6"),
    context_policy=UsageTriggeredContextPolicy(
        base_policy=RecentTurnsContextPolicy(max_user_turns=20),
        triggered_policy=RecentTurnsContextPolicy(max_user_turns=6),
        min_input_tokens=120_000,
    ),
)

app.register_agent(
    AgentSpec(name="assistant", model="claude-sonnet-4-6"),
    context_policy=UsageTriggeredContextPolicy(
        base_policy=RecentTurnsContextPolicy(max_user_turns=20),
        triggered_policy=RecentTurnsContextPolicy(max_user_turns=6),
        trigger_estimated_context_tokens=150_000,
        reserved_output_tokens=20_000,
        verify_estimate_with_provider_count=True,
        provider_count_threshold_ratio=0.9,
        provider_count_min_delta_tokens=10_000,
    ),
)
```

Register `context_overflow_policy` when a session should recover once after a
provider rejects a request as too large for the model context. The policy is
opt-in and only handles classified `ModelContextOverflowError` responses from
the provider. Cayu emits `context.overflow.detected`, rebuilds the provider
request with the overflow policy, emits `context.overflow.recovering`, and
runs the rebuilt request through the normal model-step retry policy. Cayu only
performs one overflow rebuild for a model step; if the rebuilt request also
overflows, Cayu emits `context.overflow.failed` and fails the session.

```python
from cayu import AgentSpec, CheckpointCompactionContextPolicy, RecentTurnsContextPolicy

app.register_agent(
    AgentSpec(name="assistant", model="claude-sonnet-4-6"),
    context_policy=RecentTurnsContextPolicy(max_user_turns=20),
    context_overflow_policy=CheckpointCompactionContextPolicy(max_user_turns=8),
)
```

Use `strip_old_file_attachments(...)` inside custom context policies when you build your
own transcript projection and want the same bounded native-file behavior.

Automatically inject relevant durable knowledge before each model call:

```python
from cayu import AgentSpec, KnowledgeInjectionPolicy, RecentTurnsContextPolicy

app.register_agent(
    AgentSpec(name="assistant", model="gpt-5.6"),
    context_policy=KnowledgeInjectionPolicy(
        RecentTurnsContextPolicy(max_user_turns=10),
        namespace="project:cayu",
        labels={"project": "cayu"},
        max_hits=3,
        max_bytes=4000,
    ),
)
```

`KnowledgeInjectionPolicy` searches the active environment's `knowledge_store`
with the latest user message, then appends a synthetic assistant call to
`cayu_knowledge_retrieval` and a matching tool result to the model-facing
context. Retrieved text is marked as `<untrusted_knowledge>` reference data, not
user or system instruction, and the synthetic round never enters the durable
transcript. Searches fail closed by default unless `fail_open=True`. Keep the
explicit `ListKnowledgeTool`, `SearchKnowledgeTool`, and `ReadKnowledgeTool`
available when the agent should actively explore or expand knowledge on demand.

Scope tool authority per agent:

```python
from cayu import AgentSpec, ExecCommandTool, ListFilesTool, ReadFileTool, StaticToolPolicy

app.register_agent(
    AgentSpec(name="reviewer", model="gpt-5.6"),
    tools=[ReadFileTool(), ListFilesTool(), ExecCommandTool()],
    tool_policy=StaticToolPolicy(allow=["read_file", "list_files"]),
)
```

Use parameter-constrained policies when a tool is allowed only for specific
argument shapes:

```python
from cayu import (
    AgentSpec,
    DenyPatternRule,
    ParameterConstrainedToolPolicy,
    RequiredAllowlistRule,
    RequiredFieldRule,
    ToolPolicyDecision,
)

policy = ParameterConstrainedToolPolicy(
    {
        "send_email": [
            RequiredFieldRule("to"),
            RequiredAllowlistRule(
                "template",
                values=["invoice_reminder", "receipt"],
            ),
            DenyPatternRule("to", patterns=[r"@example\.invalid$"]),
        ],
    },
    decision=ToolPolicyDecision.REQUIRE_APPROVAL,
)

app.register_agent(
    AgentSpec(name="billing_assistant", model="gpt-5.6"),
    tools=[send_email_tool],
    tool_policy=policy,
)
```

Here `send_email_tool` is an application-owned tool whose spec name is
`send_email`. The policy runs before the tool implementation. Violations either
block the call or request durable approval, depending on the configured
decision.

`RequiredAllowlistRule` is the safe default when an allowlisted field must be
present: it rejects missing, empty, non-string, and disallowed values in one
rule. `AllowlistRule` intentionally ignores a missing field and is appropriate
only when the field is genuinely optional.

Use taint-aware policies when untrusted source content should not flow into
sensitive outbound tools without a checkpoint:

```python
from cayu import AgentSpec, TaintAwareToolPolicy, ToolPolicyDecision

policy = TaintAwareToolPolicy(
    taint_sources={
        "read_email": ["external_email"],
        "read_pdf": ["artifact_pdf"],
    },
    protected_tools={
        "send_email": ["external_email", "artifact_pdf"],
        "make_payment": ["*"],
    },
    decision=ToolPolicyDecision.REQUIRE_APPROVAL,
)

app.register_agent(
    AgentSpec(name="billing_assistant", model="gpt-5.6"),
    tools=[read_email_tool, read_pdf_tool, send_email_tool, make_payment_tool],
    tool_policy=policy,
)
```

This policy does not scan text for prompt-injection phrases. It treats output
from configured source tools as untrusted by origin. Cayu derives prior taint
from durable tool events and also handles a single model round such as
`read_email` followed by `send_email` before either tool runs.

An `ExecCommandTool` can carry a deny-by-default `ProcessCommandPolicy` to
restrict *which* model-controlled commands may reach the runner:

```python
from cayu import ExecCommandTool, ProcessCommandPolicy

policy = ProcessCommandPolicy(
    # Executable strings match argv[0] exactly. "git" does not allow
    # "./git", "/usr/bin/git", or a lookalike basename.
    allowed_executables={"git"},
    # These are canonical POSIX roots in the runner's namespace.
    allowed_cwds={"/workspace/repository"},
    # Model-supplied environment entries are denied unless named here.
    allowed_env_names={"LANG"},
    # A named entry can instead be restricted to one exact value.
    allowed_env_values={"GIT_TERMINAL_PROMPT": "0"},
    max_env_value_bytes=256,
    # stdin is denied by default. This example keeps that default.
    allow_stdin=False,
    max_timeout_s=30,
)

tool = ExecCommandTool(policy=policy)
```

`request.cwd` preserves the original requested value for audit and policy
explanations. `request.canonical_cwd` is resolved by the active runner before
the policy runs and is the value passed to execution after an allow decision,
including when the request omitted `cwd`. Use the canonical value for containment
checks; built-in runners reject escaping relative traversal and absolute paths
outside their configured root first. Attaching a command policy therefore
requires a custom runner whose resolver accepts its own canonical output.

The policy authorizes no executable, cwd root, environment name, stdin, or shell
capability by default. Process executable identities use exact string matching;
configure an absolute path if the deployment needs absolute invocation. Allowed
roots use segment-aware containment. Model environment values and stdin are
UTF-8 byte-bounded and never copied into denial reasons. `max_timeout_s` may be
stricter than the tool's global 600-second input maximum. Shell source is a
separate capability: setting `allowed_executables` never enables it. Hosts that
deliberately need shell execution must set `shell_decision` explicitly, and must
treat that as permission to execute arbitrary shell code rather than as argv
authorization.

Allowing a programmable executable such as Python, Node, a shell, a package
manager, a build system, or a debugger lets that program perform behavior that
an argv allowlist cannot constrain. This policy is an authorization layer, not
filesystem, process, or network isolation; keep untrusted coding agents inside
an appropriate container or microVM. Executable-specific restrictions can
compose by implementing `CommandPolicy` and delegating the common controls to a
`ProcessCommandPolicy` before evaluating that executable's argv protocol. Cayu
does not provide a generic command-rule language. Cayu's built-in
[`GitCommandPolicy`](docs/git-command-policy.md) is the first such specialization.

Without `policy=...`, the compatibility contract is intentionally unchanged:
model-controlled command, cwd, environment, and stdin arguments pass to the
runner subject to the tool's input bounds and the runner's own isolation.

A `CommandPolicy` denial surfaces as a `tool.call.failed` event (see the note in
Contract Rules on blocked-vs-failed). To gate a whole tool behind human approval,
pass `AlwaysRequireApprovalToolPolicy(tools=["post_pr_comment"])` as the agent's
`tool_policy` instead of hand-rolling a deny-everything rule.

Custom policies can also require caller approval before a tool round runs. The runtime checkpoints the pending approval, emits `tool.call.approval_requested`, marks the session `interrupted`, and waits for `resolve_tool_approval(...)`:

```python
from cayu import PendingToolApproval, ToolApprovalDecision, ToolApprovalRequest

# `event` is the tool.call.approval_requested event from the run's event stream.
# The approval is nested under event.payload["approval"]; from_event reads it
# (event.payload["approval_id"] would be None).
pending = PendingToolApproval.from_event(event)

async for resolved in app.resolve_tool_approval(
    ToolApprovalRequest(
        session_id=event.session_id,
        approval_id=pending.approval_id,
        decision=ToolApprovalDecision.APPROVE,
    )
):
    print(resolved.type)
```

While a tool approval is pending, `app.resume(...)` rejects normal message resume. Approving or denying the pending approval writes the required tool results and clears the pending checkpoint atomically, then the model loop continues with valid provider-neutral history. If approval close is retried, Cayu reuses durable terminal tool events instead of running completed tools again. If a side-effecting tool started but Cayu cannot prove whether it completed, record the known external outcome and continue without re-running the tool:

```python
from cayu import ToolApprovalRecoveryOutcome, ToolApprovalRecoveryRequest

async for event in app.recover_tool_approval(
    ToolApprovalRecoveryRequest(
        session_id=event.session_id,
        approval_id=pending.approval_id,
        tool_call_id=pending.tool_call_id,
        outcome=ToolApprovalRecoveryOutcome.COMPLETED,
        message="Confirmed in email provider logs: the email was sent.",
        structured={"sent": True},
        reason="Confirmed in email provider logs.",
    )
):
    print(event.type)
```

Use checkpoint-backed compaction for long-running sessions. Without an explicit
compactor, Cayu uses a deterministic transcript digest fallback. For semantic
summaries, provide a model-backed compactor:

```python
from cayu import (
    AgentSpec,
    AnthropicProvider,
    CheckpointCompactionContextPolicy,
    ModelCompactor,
)

summary_provider = AnthropicProvider()

app.register_agent(
    AgentSpec(name="assistant", model="claude-sonnet-4-6"),
    context_policy=CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(
            provider=summary_provider,
            model="claude-sonnet-4-6",
            system_prompt="Return only a compact continuation summary.",
            options={"anthropic": {"max_tokens": 2000}},
        ),
        max_user_turns=10,
        compact_after_messages=40,
    ),
)
```

Advanced users can replace the compaction prompt body with
`ModelCompactor(prompt_builder=...)`. The built-in checkpoint policy stores the
summary and transcript cursor in the session checkpoint, then injects the
summary as model-facing user context. It does not append the summary to the
durable transcript.

For providers with prompt caching, `PromptCacheCompactor` preserves the first
compaction's exact runtime request prefix—including tools, request-level
thinking options, and resolved prompt attachments—then switches subsequent
checkpoints to bounded `previous summary + new delta` compaction:

```python
from cayu import PromptCacheCompactor

context_policy = CheckpointCompactionContextPolicy(
    compactor=PromptCacheCompactor(
        provider=summary_provider,
        options={"anthropic": {"max_tokens": 1000}},
    ),
    max_user_turns=10,
    compact_after_messages=40,
)
```

Compactor options override copied agent/provider options recursively. Both the
cache-aware first call and later bounded calls follow the same structured
provider-error/retry contract and are included in durable session usage. Exact
reuse additionally requires durable proof that the previous request used the
same provider and requested model. A provider/model identity change or
tool-based structured-output request takes the bounded path, preventing an
incompatible provider request from receiving the copied full prefix, tools, or
resolved attachment bytes.
