# Cayu

Cayu is a runtime for building production AI agents in Python.

It provides the durable execution layer—sessions, model calls, tool execution,
approvals, context management, budgets, recovery, events, and evals—while
applications retain control of their UI, authentication, domain logic, and
workflows.

Cayu is designed for agents that do consequential or long-running work. It is
not a prompt-chain DSL, a visual workflow builder, or an application frontend;
you compose its runtime primitives directly.

## Why Cayu

Agent prototypes are easy to start. Production failures happen at the
boundaries:

- a process dies after a side effect but before state is recorded;
- a model requests a valid tool with the wrong authority;
- a run needs human input or approval halfway through;
- context grows until a provider rejects the next request;
- retries, forks, or subagents lose cost and causal attribution;
- operators cannot reconstruct what happened from prompt text alone; or
- evals test final prose while missing the runtime trajectory.

Cayu treats these as runtime contracts. Important actions become structured
events; tool authority and recovery are explicit; configured durable stores let
transcripts and checkpoints survive process boundaries; and the same public
seams support local development, tests, control-plane inspection, and hosted
deployments.

## What Cayu provides

| Need | Cayu primitive |
| --- | --- |
| Long-running work | Durable sessions, transcripts, events, resume, fork, interruption |
| Safe effects | Typed tools, effect declarations, policies, approvals, idempotency keys |
| Human interaction | User-input checkpoints, approval resolution, manual recovery |
| Context pressure | Token counting, projection, compaction, overflow recovery |
| Cost control | Usage events, run limits, budgets, pricing, causal-budget summaries |
| Execution boundaries | Environments, workspaces, runners, artifacts, vaults, egress |
| Reviewed knowledge | Durable entries, approval state, keyword/vector retrieval, recall tools |
| Provider flexibility | OpenAI, Anthropic, Bedrock, Vertex, OpenAI-compatible APIs |
| Durable automation | Tasks, dispatchers, event watchers, subagents, runtime hooks |
| Behavioral proof | Runtime tests, trajectory assertions, replay, eval reports |
| Operations | FastAPI control plane and a packaged inspection dashboard |

## Quickstart

### Start a project

The generated project is the recommended path for both humans and coding
agents. Cayu requires Python 3.11 or newer.

```bash
pip install cayu pytest
cayu new myagent
cd myagent

cayu inspect --json
cayu check --json
pytest
cayu eval run evals.assistant:build_eval
```

The scaffold is credential-free and includes:

- a process-scoped `build_app()` factory with bounded local-store initialization;
- one safe example tool;
- a hermetic runtime test and trajectory eval; and
- a project-local `AGENTS.md` with the exact build and verification contract.

Open the generated project, replace the tracer-bullet behavior with your domain
behavior, and keep the public test/eval seam intact.

### Run an agent

This compact example shows the core API. Real projects should put the same
registrations in the generated `build_app()` factory instead of constructing a
module-global app.

```python
import asyncio

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    OpenAIProvider,
    RunRequest,
    run_to_completion,
)


async def main() -> None:
    app = CayuApp()
    app.register_provider(OpenAIProvider(), default=True)  # reads OPENAI_API_KEY
    app.register_agent(AgentSpec(name="assistant", model="gpt-5.6"))

    outcome = await run_to_completion(
        app,
        RunRequest(
            agent_name="assistant",
            messages=[Message.text("user", "Explain durable agent sessions.")],
        ),
    )

    if outcome.ok:
        print(outcome.final_text)
    else:
        print(f"{outcome.status}: {outcome.error}")


asyncio.run(main())
```

`CayuApp()` uses in-memory stores by default, which is appropriate for this
one-shot example and for tests. The generated project configures SQLite so its
sessions survive process restarts. Multi-process production deployments should
select a conforming shared store such as PostgreSQL.

`CayuApp.run(...)` is the lower-level event-stream API. Runtime failures are
terminal `session.failed` events, not exceptions raised from iteration.
`run_to_completion(...)` consumes that same stream and returns a typed outcome
when an application only needs the result. It retains the complete event stream
in `RunOutcome.events`; use it for bounded runs. Consume `CayuApp.run(...)`
incrementally for long-lived or high-volume runs.

For a runnable example with no API key, use
[`examples/echo_tool_runtime.py`](https://github.com/cayu-dev/cayu/blob/main/examples/echo_tool_runtime.py). To add
workspace tools and command execution, see
[`examples/local_environment_runtime.py`](https://github.com/cayu-dev/cayu/blob/main/examples/local_environment_runtime.py).

### Build with a coding agent

The generated `AGENTS.md` is the project-local source of truth. Ask the coding
agent to read it first, then use Cayu's package-shipped guides and structured
inspection:

```bash
cayu guide anatomy
cayu guide authoring
cayu inspect --json
cayu check --fail-on warning --json
```

The supported authoring loop is:

```text
understand -> inspect -> plan -> change -> test -> eval -> exercise -> report evidence
```

Use `cayu generate slice ... --dry-run --json` to plan a small vertical slice.
Generated slices are intentionally unfinished tracer bullets; replace their
placeholder prompt, tool behavior, runtime test, and eval before treating them
as application functionality.

## Mental model

Cayu separates the agent's identity from the resources and durable state used
for one execution:

```text
AgentSpec
  identity, model, instructions, defaults, runtime policies

Environment
  workspace, runner, artifacts, vault, proxy, knowledge, MCP

Session
  durable identity, transcript, events, status, checkpoints

ToolContext
  the active environment services and call identity for one tool execution
```

- **Agent** describes who is acting and how model work is configured.
- **Environment** describes what that agent can touch.
- **Session** records one durable execution and its lineage.
- **Tool** is an explicitly registered capability, not arbitrary model code.
- **Task** is an optional durable unit of background or orchestrated work.
- **Workflow** is deterministic application orchestration around agent steps.

An environment is optional for a conversational agent. It becomes important
when tools need files, commands, artifacts, secrets, network policy, or a
sandbox. Static environments are useful for trusted local work;
`EnvironmentFactory` creates or reattaches session-specific environments in
production.

## Use the smallest runtime shape

Do not add every Cayu primitive to every application.

| Desired behavior | Start with |
| --- | --- |
| One model-driven interaction | `CayuApp`, `AgentSpec`, provider, `RunRequest` |
| Deterministic model-callable action | `Tool`, `ToolSpec`, explicit `ToolEffect` |
| Authority over an effect | `ToolPolicy`; approval only where a human gate is required |
| Mutable files or commands | Explicit `Environment`, `Workspace`, and `Runner` |
| Durable uploaded or generated files | `ArtifactStore` |
| Long-lived conversation or recovery | Durable `SessionStore` and checkpoint APIs |
| Background durable work | `TaskStore` plus an explicitly started worker |
| Delegated model work | Subagent tools with bounded child-session policy |
| Behavioral regression proof | `EvalSuite` and trajectory assertions |

A conversation agent does not automatically need a workflow, task queue,
environment, memory store, server, or multi-agent topology. A coding agent does
not automatically need unrestricted shell access. Prefer narrow domain tools
and add authority only when the behavior requires it.

## Application UI and control plane

Your application should own:

- end-user prompts and domain forms;
- product authentication and authorization;
- business-specific workflow and state;
- user-facing streaming, notifications, and presentation; and
- decisions about when a run, task, approval, or interruption is allowed.

Cayu owns runtime execution and the operational state recorded by the
application's configured stores. Its optional dashboard is a control plane for
developers and operators: inspect sessions,
events, transcripts, tasks, usage, artifacts, pending actions, and recovery
state. It is not intended to replace the product experience your users need.

Start work through the API that matches the trigger:

- `run` for an immediate new session;
- `resume` for a deliberate continuation;
- `dispatch` for placement through a dispatcher;
- a task worker for durable queued work;
- a subagent for model-selected bounded delegation; or
- an event watcher for durable reactions to already-persisted events.

See [Triggering runs](https://github.com/cayu-dev/cayu/blob/main/docs/triggering-runs.md) for the decision guide and
lifecycle responsibilities.

## Providers and environments

The base package includes the provider contracts and built-in OpenAI, Anthropic,
and OpenAI-compatible HTTP adapters. Optional extras add integrations without
forcing their dependencies into every deployment:

| Extra | Adds |
| --- | --- |
| `cayu[server]` | FastAPI control plane and packaged dashboard |
| `cayu[postgres]` | PostgreSQL session, task, knowledge, and related stores |
| `cayu[aws]` | Amazon Bedrock and Lambda MicroVM support |
| `cayu[vertex]` | Anthropic models through Google Cloud Vertex AI |
| `cayu[e2b]` | E2B runner and workspace |
| `cayu[microsandbox]` | Local microVM-backed untrusted-code runner |
| `cayu[egress]` | Virtual egress and credential-broker primitives |
| `cayu[files]` | Image and PDF inspection |
| `cayu[console]` | Interactive application console |

Providers normalize text, thinking, tool calls, usage, completion reasons, and
typed failures behind one runtime contract. Cayu does not infer a provider from
an arbitrary model name: applications register providers explicitly and may
add deterministic model-pattern routing.

The same agent can run in a local workspace, trusted Docker container, E2B,
Microsandbox, Lambda MicroVM, or an application-owned runner without changing
its identity or transcript contract.

## Production boundaries

Cayu makes safety boundaries explicit, but configuration still matters:

- `LocalRunner` is trusted local execution, not a sandbox.
- `DockerRunner` is useful for development and CI; ordinary Docker isolation
  is not presented as a secure untrusted-code boundary.
- Environment registration does not imply selection: mark a default explicitly
  or name the environment on the request. Provider defaults and model-pattern
  routing should likewise be configured deliberately and kept unambiguous.
- Tool effects do not authorize themselves. Use policies, approvals, scoped
  credentials, and destination controls where consequences require them.
- SQLite is appropriate for local and single-writer deployments. Use PostgreSQL
  or another conforming shared store for sustained multi-process concurrency.
- The FastAPI control plane requires authentication outside explicit
  `dev=True`; protect it as an operator surface, not an end-user endpoint.
  Generated API documentation is a separate exposure decision.
- Usage is derived from recorded events and survives restarts when those events
  use a durable store; cost remains an estimate against the price book your
  application selects.
- Recovery never invents the outcome of an ambiguous external side effect.
  Reconcile it through the typed recovery APIs.

Read [Runtime contracts](https://github.com/cayu-dev/cayu/blob/main/docs/runtime-contracts.md) before changing persistence,
replay, approval, interruption, budget, provider, runner, or recovery behavior.

## Documentation

Start with the document that matches the job:

| Goal | Guide |
| --- | --- |
| Build an application, by hand or with an AI coding agent | [Authoring guide](https://github.com/cayu-dev/cayu/blob/main/src/cayu/guides/authoring.md) |
| Classify tool mutation and replay behavior | [Tool effects](https://github.com/cayu-dev/cayu/blob/main/src/cayu/guides/tool-effects.md) |
| Understand factories, process roles, and lifecycle | [Application anatomy](https://github.com/cayu-dev/cayu/blob/main/src/cayu/guides/application-anatomy.md) |
| Choose how work starts | [Triggering runs](https://github.com/cayu-dev/cayu/blob/main/docs/triggering-runs.md) |
| Create per-session workspaces and runners | [Environment factories](https://github.com/cayu-dev/cayu/blob/main/docs/environment-factories.md) |
| Implement a runner for your platform | [Build a runner](https://github.com/cayu-dev/cayu/blob/main/docs/build-a-runner.md) |
| Configure network and credential boundaries | [Virtual egress](https://github.com/cayu-dev/cayu/blob/main/docs/virtual-egress.md) |
| Design assertions and trajectory evals | [Evals](https://github.com/cayu-dev/cayu/blob/main/docs/evals.md) |
| Estimate and govern cost | [Cost optimization](https://github.com/cayu-dev/cayu/blob/main/docs/cost-optimization.md) |
| Use the application console | [Console](https://github.com/cayu-dev/cayu/blob/main/docs/console.md) |
| Inspect supported model metadata | [Model catalog](https://github.com/cayu-dev/cayu/blob/main/docs/model-catalog.md) |
| Look up exact runtime behavior | [Runtime contracts](https://github.com/cayu-dev/cayu/blob/main/docs/runtime-contracts.md) |
| Track prerelease behavior and migrations | [Release notes](https://github.com/cayu-dev/cayu/blob/main/docs/release-notes.md) |

Maintainer-facing architecture is documented in
[Architecture](https://github.com/cayu-dev/cayu/blob/main/docs/architecture.md),
[Project layout](https://github.com/cayu-dev/cayu/blob/main/docs/project-layout.md),
and the [Glossary](https://github.com/cayu-dev/cayu/blob/main/docs/glossary.md).

## Examples

- [Echo tool runtime](https://github.com/cayu-dev/cayu/blob/main/examples/echo_tool_runtime.py) — credential-free model/tool loop.
- [Local environment runtime](https://github.com/cayu-dev/cayu/blob/main/examples/local_environment_runtime.py) — files and commands.
- [Server example](https://github.com/cayu-dev/cayu/blob/main/examples/server_example.py) — protected API and control plane.
- [Cloud PR reviewer](https://github.com/cayu-dev/cayu/blob/main/docs/recipes/pr-reviewer.md) — durable task, isolated workspace,
  QA, and an explicit external effect.
- [Business approvals](https://github.com/cayu-dev/cayu/blob/main/docs/recipes/business-approvals.md) — domain approval routing
  over the binary runtime primitive.
- [Advanced runtime examples](https://github.com/cayu-dev/cayu/blob/main/examples/ADVANCED_RUNTIME_EXAMPLES.md) — forks,
  compaction, taint isolation, speculative approval, and measured evidence.

Advanced examples are executable runtime specifications, not claims that one
strategy fits every workload. Their evidence boundaries and measured results
are described in
[Advanced runtime strategies](https://github.com/cayu-dev/cayu/blob/main/docs/advanced-runtime-examples.md).

## Contributing and security

Framework contributors should read
[CONTRIBUTING.md](https://github.com/cayu-dev/cayu/blob/main/CONTRIBUTING.md) for
placement policy, setup, validation commands, and pull-request requirements.
New third-party integrations normally live in their own packages against
Cayu's public extension contracts.

Report suspected vulnerabilities privately as described in
[SECURITY.md](https://github.com/cayu-dev/cayu/blob/main/SECURITY.md). Do not open a public issue or pull request for a
suspected security vulnerability.

For questions and project discussion, join
[Discord](https://discord.gg/jWa3kKJ7R8). Use
[GitHub issues](https://github.com/cayu-dev/cayu/issues) for actionable bugs and
concrete feature proposals.

## License

Cayu is licensed under the
[Apache License 2.0](https://github.com/cayu-dev/cayu/blob/main/LICENSE).
