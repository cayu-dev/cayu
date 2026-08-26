# Cayu

Cayu is a production agent runtime for building and operating AI agents in
Python.

A harness turns a model into an agent by supplying its context, tools,
permissions, and execution logic. Cayu gives applications control of the full
agent execution lifecycle: how context is assembled, models and tools are
invoked, where agent code runs, how state is persisted, authority is governed,
failures are recovered, and behavior is observed and evaluated.

Cayu provides durable agent-runtime primitives including sessions, task
dispatch, leased workers, resumable workflow steps, approvals, and recovery.
Applications can use them directly without a separate workflow engine.

Applications retain control of their UI, authentication, domain logic, and
business workflows.

Cayu is designed for agents that do consequential or long-running work. You
compose its runtime primitives directly in your application.

## Why we built Cayu

Cayu was extracted from the production runtime behind an agent-operated
software factory that built and deployed thousands of business applications.
Specialized agents worked together as the AI SRE, AI product manager, AI coder,
and FDE assistant behind that delivery process.

We began by building agents with SDKs and frameworks including the Claude Agent
SDK, Mastra, and LangGraph. They helped us implement the model-and-tool loop
quickly. Production quality required deeper control of the loop itself: context
assembly, model and tool invocation, output validation, and failure handling.

Production quality also depended on everything around that loop: scheduling,
durable state, credentials, execution environments, human intervention,
recovery, cost attribution, and evaluation. Cayu gives applications control of
that entire agent execution lifecycle.

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
| Long-term recall | Bounded knowledge/transcript sources, deterministic fusion, exact locators and coverage |
| Provider flexibility | OpenAI API, experimental OpenAI subscription login, Anthropic, Bedrock, Vertex, OpenAI-compatible APIs |
| Agent operations | Tasks, dispatchers, event watchers, subagents, runtime hooks |
| Behavioral proof | Runtime tests, production-session promotion, durable evals, comparison, and CI reports |
| Operations | FastAPI control plane and a packaged dashboard for sessions, workflows, usage, and evals |

## Quickstart

### Start a project

The generated project is the recommended path for both humans and coding
agents. Cayu requires Python 3.11 or newer.

You can give a coding agent one request: “Run `pip install cayu` and create a
code review agent.”

```bash
pip install cayu pytest
cayu new myagent
cd myagent

cayu inspect --json
cayu check --json
pytest
cayu eval run

# After configuring the provider in app.py:
python run.py --message "Review this change."
```

The scaffold is credential-free and includes:

- a process-scoped `build_app()` factory with bounded local-store initialization;
- one model-only agent with no required tools;
- a hermetic runtime test and output eval; and
- a project-local `AGENTS.md` with the exact build and verification contract.

Open the generated project, describe the requested job in the existing agent,
and keep its public test/eval seam intact.

For an explicit, editable repository-coding starter with bounded file and Git
tools, durable knowledge, background review delegation, and human input, opt in
to the maintained composition:

```bash
cayu new mycoder --composition coding
cd mycoder
cayu check --json
pytest -q tests/test_coding_composition.py
python run.py --agent mycoder --message "Implement the requested change."
```

The generated repository starts from a clean Git commit and requires `git`, `rg`,
and the POSIX descriptor-relative filesystem primitives used by secure
`LocalWorkspace` path operations. Unsupported hosts fail during generation or
application construction. Its local workspace and runner are trusted-host
development adapters, not a hostile-code sandbox. The default scaffold remains
the minimal one-agent project, and the coding composition cannot be combined
with `--template service`.

### Deploy with Cayu Cloud

Cloud commands ship in the same `cayu` package; no additional CLI package is
required. Cayu Cloud is currently invite-only. Login-backed commands are pinned to
the production service at `https://cloud.cayu.dev`; users never select an API URL.

```bash
cayu cloud --help
cayu cloud login
cayu cloud whoami
cayu cloud init
cayu cloud deploy
```

Login uses WorkOS device authorization. It opens the browser when possible and always
prints a verification URL and one-time user code for SSH, containers, Cursor, Codex,
and Claude Code. The resulting Organization-scoped session is stored privately and
refreshed automatically; no WorkOS secret is embedded in Cayu. Use `--no-browser` when
a human will open the displayed URL on another device. Login selects the WorkOS session
over any previously persisted private context; a later `cayu cloud context use PATH`
deliberately switches back to that internal automation context. Private contexts and
explicit API credentials remain endpoint-bound operator mechanisms and do not change
the default customer endpoint.

`cayu cloud init` generates the small deployment descriptor from standard
`pyproject.toml` metadata and a configured Cayu server, worker, or console
script. Review the generated process topology before deploying it. Existing
descriptors are never replaced unless `--force` is explicit.

The default command packages the current local working directory and uploads it
directly to the selected Cayu Cloud Organization as an immutable source bundle.
It includes an applied patch and does not require GitHub, a clean worktree, a
commit, or a push. Git-ignored files and common local credential/cache paths are
omitted.

Deploy creates or updates the 8-63 character application slug declared in
`cayu-cloud.toml`. Slugs use lowercase letters, numbers, and interior hyphens.
`--application SLUG` overrides that create-or-update slug; check it carefully because
a valid typo creates a separate application.

Deploy output and local evidence redact runtime environment values. Cayu verifies the
evidence destination before Cloud mutation; if only the final evidence write fails after
a successful rollout, deploy still exits successfully with `evidence_id: null` and an
`evidence.status: unavailable` diagnostic.

To deploy an exact remote GitHub revision instead, pass its canonical URL and
40-character commit. For a private source, make authentication available to the
process that runs Cayu:

```bash
gh auth status --hostname github.com
# For noninteractive CI or an isolated Codex/Claude Code home:
GH_TOKEN="$GITHUB_TOKEN" cayu cloud deploy \
  https://github.com/example/agent --revision COMMIT_SHA
```

GitHub CLI credentials stored in a desktop keychain may not be reachable after
changing `HOME`, even when `GH_CONFIG_DIR` points at an authenticated GitHub CLI
configuration. In that case, pass a short-lived `GH_TOKEN` explicitly. Cayu
does not persist the token in its Cloud context or include GitHub CLI stderr in
its JSON errors. Public GitHub repositories remain deployable without GitHub
authentication.

For CI and other noninteractive automation, the existing
`CAYU_CLOUD_API_KEY`, `CAYU_CLOUD_API_KEY_FILE`, and private Cloud-context
options remain available. Explicit automation credentials take precedence over
the saved interactive login.

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
one-shot example and for tests. The generated project configures all local Cayu
stores in `data/cayu.db` so sessions survive process restarts. Multi-process
production deployments should select a conforming shared store such as
PostgreSQL.

`CayuApp.run(...)` is the lower-level event-stream API. Runtime failures arrive
as terminal `session.failed` events instead of exceptions raised from
iteration.
`run_to_completion(...)` consumes that same stream and returns a typed outcome
when an application only needs the result. It retains the complete event stream
in `RunOutcome.events`; use it for bounded runs. Consume `CayuApp.run(...)`
incrementally for long-lived or high-volume runs.

For a credential-free domain-tool tracer bullet, run `cayu guide
references#domain-tool`, then use `cayu generate tool`. To add workspace tools
and command execution, see
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

The package-shipped `cayu guide authoring#cayu-map` routes each optional
capability to the smallest version-matched local reference. Its online
[source mirror](https://github.com/cayu-dev/cayu/blob/main/src/cayu/guides/authoring.md#cayu-map)
is secondary. The
[examples index](https://github.com/cayu-dev/cayu/blob/main/examples/README.md)
provides runnable references without making them required project structure.

The supported authoring loop is:

```text
understand -> inspect -> plan -> change -> test -> eval -> exercise -> report evidence
```

Start by editing the existing model-only agent, test, and eval. Add a generated
tool-backed slice only when the requested job needs a capability outside the
model; generated slices remain unfinished until their placeholder behavior,
test, and eval have been replaced.

## Mental model

Cayu separates the agent's identity from the resources and durable state used
for one execution:

```text
AgentSpec
  identity, model, system prompt, defaults, runtime policies

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
- **Tool** is an explicitly registered, application-owned capability that the
  model may request. A native Python `Tool` runs inside the trusted Cayu
  application process; `ToolPolicy` gates its invocation but does not sandbox
  its implementation.
- **Task** is an optional durable unit of background or orchestrated work.
- **Workflow** is deterministic application orchestration around agent steps.

An environment is optional for a conversational agent. It becomes important
when tools need files, commands, artifacts, secrets, network policy, or a
sandbox. Static environments are useful for trusted local work;
`EnvironmentFactory` creates or reattaches session-specific environments in
production.

Choose the execution surface according to where code should run and which
boundary should contain it:

| Surface | Execution location | Boundary |
| --- | --- | --- |
| Native Python `Tool` | Cayu application process | Trusted application code; policy controls invocation, not host-process access |
| Process-isolated host `Tool` | Disposable POSIX child process session | Hard wall-clock liveness and owned process-group cleanup for an explicitly reconstructable trusted adapter; not a security sandbox |
| Runner-backed operation (`ctx.runner`) | Selected runner for that operation; the enclosing `Tool.run()` stays in the Cayu application process | Isolation, environment, network, and filesystem guarantees for the operation come from the admitted runner and environment |
| MCP tool | Configured MCP process or server | Separate integration boundary whose process, transport, credentials, and isolation remain deployment choices |
| Virtual egress | Selected runner plus a trusted broker outside it | A conforming adapter can keep the real credential out of the workload; code isolation still depends on the runner |

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

Start a conversation agent with the model and state it needs. Add workflows,
task queues, environments, memory stores, servers, or multi-agent topology when
the behavior requires them. Give coding agents narrow domain tools before
granting broader shell access.

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
state. Your application remains responsible for the product experience.

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
OpenAI-compatible HTTP, and experimental OpenAI-subscription adapters. Optional
extras add integrations without forcing their dependencies into every
deployment:

| Extra | Adds |
| --- | --- |
| `cayu[server]` | FastAPI control plane and packaged dashboard |
| `cayu[server-settings]` | Server extra plus typed environment and `.env` loading |
| `cayu[postgres]` | PostgreSQL session, task, knowledge, and related stores |
| `cayu[aws]` | Amazon Bedrock and Lambda MicroVM support |
| `cayu[vertex]` | Anthropic models through Google Cloud Vertex AI |
| `cayu[e2b]` | E2B runner and workspace |
| `cayu[microsandbox]` | Local microVM-backed untrusted-code runner |
| `cayu[egress]` | Virtual egress and credential-broker primitives |
| `cayu[files]` | Image and PDF inspection |
| `cayu[console]` | Interactive application console |
| `cayu[otel]` | OpenTelemetry tracing and metrics support |
| `cayu[all]` | Every runtime integration extra above |

`cayu[all]` intentionally excludes `cayu[browser]`, which is dashboard and
browser verification tooling rather than a runtime integration.

Providers normalize text, thinking, tool calls, usage, completion reasons, and
typed failures behind one runtime contract. Applications register providers
explicitly and may add deterministic model-pattern routing; an arbitrary model
name never selects a provider.

Cayu focuses on OpenAI, Anthropic, OpenRouter, Google, Bedrock, and Vertex.
OpenRouter is a first-class `cayu new --provider openrouter` and
`CAYU_PROVIDER=openrouter` choice backed by the generic Chat Completions adapter;
it requires `OPENROUTER_API_KEY` and an explicit `CAYU_MODEL` slug. Compatible
Chat Completions services such as Fireworks, Baseten Model APIs, and OpenCode Go
use that generic adapter directly. Run `cayu guide providers#openrouter` or use
the [package guide](https://github.com/cayu-dev/cayu/blob/main/src/cayu/guides/providers.md#openrouter)
for exact setup.

For local development without separate OpenAI API billing, users can sign in
with their own ChatGPT subscription:

```bash
cayu auth openai login
# For SSH or a remote machine:
cayu auth openai login --headless
```

```python
from cayu import OpenAISubscriptionProvider

app.register_provider(OpenAISubscriptionProvider(), default=True)
```

This experimental integration uses the Codex backend. It does not use the
documented OpenAI Platform API. Cayu identifies itself with `originator: cayu`
and preserves upstream rejections. OpenAI has not documented this raw backend
as a general third-party provider API, so support may change or stop.

> **Intended-use boundary:** Use this path only for a subscription holder's own
> local development and evaluation. For production, customer-facing or
> multi-user services, use the OpenAI Platform API or another officially
> supported provider. Do not share or resell credentials or bypass plan limits.

See [OpenAI subscription authentication](https://github.com/cayu-dev/cayu/blob/main/docs/openai-subscription.md) for the
support boundary, credential storage, and fallback options.

The same agent can run in a local workspace, trusted Docker container, E2B,
Microsandbox, Lambda MicroVM, or an application-owned runner without changing
its identity or transcript contract.

## Open and replaceable control plane

Cayu ships three equally supported control-plane choices: use the bundled compiled dashboard,
eject the exact version-matched React/TypeScript source for application-owned customization, or
provide a completely custom UI over the versioned control-plane API.

```bash
cayu dashboard eject ./control-plane
cd control-plane
npm ci
npm run dev
npm run build
```

Extraction uses package data only—no repository clone, GitHub access, or network request. Serve
the resulting `dist/` with `DashboardConfig(directory=...)`,
`mount_cayu(..., dashboard_dir=...)`, or `mount_dashboard(..., dashboard_dir=...)`. Cayu never
rewrites the extracted application-owned source during upgrades. See [Open and replaceable
control plane](https://github.com/cayu-dev/cayu/blob/main/docs/control-plane.md) for the complete workflow, compatibility gate, and
redistribution obligations.

## Production boundaries

Cayu makes safety boundaries explicit, but configuration still matters:

- Native Python `Tool` implementations are trusted host-process code and can
  access authority available to the Cayu application. `ToolPolicy` controls
  whether the model may call a tool and with which arguments; it is not an OS
  isolation boundary. Run model-authored or otherwise untrusted code through an
  admitted runner or separately governed external tool boundary. Native tools
  that need credentials should use explicit `SecretRef` values through
  `ctx.proxy` or `ctx.vault` and return only safe results; ambient host
  environment values are not automatically mediated as workload credentials.
- `LocalRunner` executes directly on a trusted local machine and provides no
  sandbox isolation.
- `DockerRunner` is useful for development and CI; ordinary Docker isolation
  is not presented as a secure untrusted-code boundary.
- Environment registration does not imply selection: mark a default explicitly
  or name the environment on the request. Provider defaults and model-pattern
  routing should likewise be configured deliberately and kept unambiguous.
- Tool effects do not authorize themselves. Use policies, approvals, scoped
  credentials, and destination controls where consequences require them.
- SQLite is appropriate for local and single-writer deployments. Use PostgreSQL
  or another conforming shared store for sustained multi-process concurrency.
- The FastAPI control plane requires an explicit `ServerConfig` access policy.
  Use `AuthenticatedAccess` for deployed operator surfaces; `OpenAccess` and
  `ServerConfig.local_development()` are deliberate local-only choices.
  Deployment names are descriptive metadata and never relax security policy.
  See [server configuration](https://github.com/cayu-dev/cayu/blob/main/docs/server-configuration.md).
  `AuthContext.tenant` records authenticated operator provenance but does not
  filter or isolate Cayu data. See [Server authentication and tenant
  isolation](https://github.com/cayu-dev/cayu/blob/main/docs/recipes/server-auth-tenancy.md).
  Generated API documentation is a separate exposure decision.
- Public or multi-user product routes need a separate customer authorization
  boundary; operator authentication does not make the raw Cayu surface
  tenant-scoped. Start new services with
  `cayu new NAME --template service`, then require both
  `cayu check --deploy --fail-on warning --json` and the generated
  `tests/test_public_service_security.py`. Cayu reports arbitrary host-owned
  ASGI routes outside that maintained factory as unverified.
- When embedding with `mount_cayu(..., path="/your/path")` or the lower-level
  `mount_dashboard(...)`, use `/your/path/` as the canonical dashboard URL.
  Cayu redirects an exact GET or HEAD of the slashless non-root mount after a
  successful dashboard mount. That public 307 may be returned without
  credentials; dashboard HTML, assets, deep links, and other protected content
  at the canonical target still require configured authentication.
  `mount_cayu(...)` places its control-plane API under `/your/path/api`;
  `mount_dashboard(...)` configures `apiBaseUrl` independently and defaults it
  to `/api`.
- Usage is derived from recorded events and survives restarts when those events
  use a durable store; cost remains an estimate against the price book your
  application selects.
- Recovery never invents the outcome of an ambiguous external side effect.
  Reconcile it through the typed recovery APIs.

Read [Runtime contracts](https://github.com/cayu-dev/cayu/blob/main/docs/runtime-contracts.md) before changing persistence,
replay, approval, interruption, budget, provider, runner, or recovery behavior.

## Documentation

Start with the [documentation index](https://github.com/cayu-dev/cayu/blob/main/docs/README.md)
for maintained guides, operational references, design records, research evidence,
and explicitly archived material.

Start with the document that matches the job:

| Goal | Guide |
| --- | --- |
| Choose Cayu concepts and build an application, by hand or with an AI coding agent | `cayu guide authoring#cayu-map` |
| Configure primary or compatible model services | `cayu guide providers` ([source](https://github.com/cayu-dev/cayu/blob/main/src/cayu/guides/providers.md)) |
| Classify and verify tool mutation and replay behavior | `cayu guide tool-effects` |
| Build a durable propose, authorize, act, verify, and recover lifecycle | `cayu guide durable-operations` ([source](https://github.com/cayu-dev/cayu/blob/main/src/cayu/guides/durable-operations.md)) |
| Understand factories, process roles, and lifecycle | `cayu guide anatomy` ([source](https://github.com/cayu-dev/cayu/blob/main/src/cayu/guides/application-anatomy.md)) |
| Choose how work starts | [Triggering runs](https://github.com/cayu-dev/cayu/blob/main/docs/triggering-runs.md) |
| Create per-session workspaces and runners | [Environment factories](https://github.com/cayu-dev/cayu/blob/main/docs/environment-factories.md) |
| Implement a runner for your platform | [Build a runner](https://github.com/cayu-dev/cayu/blob/main/docs/build-a-runner.md) |
| Contain a non-cooperative trusted host dependency behind a hard deadline | [Process-isolated host tools](https://github.com/cayu-tech/cayu/blob/main/docs/process-isolated-tools.md) |
| Configure network and credential boundaries | [Virtual egress](https://github.com/cayu-dev/cayu/blob/main/docs/virtual-egress.md) |
| Let an agent search and read bounded public web evidence | [Web fetch and hosted search](https://github.com/cayu-dev/cayu/blob/main/docs/web-fetch.md) |
| Run GitHub CLI without giving the runner a real token | [GitHub CLI through virtual egress](https://github.com/cayu-dev/cayu/blob/main/docs/recipes/github-cli-virtual-egress.md) |
| Design assertions and trajectory evals | [Evals](https://github.com/cayu-dev/cayu/blob/main/docs/evals.md) |
| Understand knowledge authorization, retrieval fusion, and memory baselines | [Memory foundation contracts](https://github.com/cayu-dev/cayu/blob/main/docs/memory-foundation.md) |
| Reproduce bounded stateful agent evaluations | [Portable agent snapshots](https://github.com/cayu-dev/cayu/blob/main/docs/runtime-contracts.md#portable-agent-snapshots) |
| Estimate and govern cost | [Cost optimization](https://github.com/cayu-dev/cayu/blob/main/docs/cost-optimization.md) |
| Use the application console | [Console](https://github.com/cayu-dev/cayu/blob/main/docs/console.md) |
| Start a configured server process | [Project server](https://github.com/cayu-dev/cayu/blob/main/docs/project-server.md) |
| Use, customize, or replace the operator control plane | [Open and replaceable control plane](https://github.com/cayu-dev/cayu/blob/main/docs/control-plane.md) |
| Start a named worker process | [Project workers](https://github.com/cayu-dev/cayu/blob/main/docs/project-workers.md) |
| Configure CLI session-store discovery | [Session-store targets](https://github.com/cayu-dev/cayu/blob/main/docs/session-store-targets.md) |
| Inspect durable sessions safely | [Session inspection](https://github.com/cayu-dev/cayu/blob/main/docs/session-inspection.md) |
| Configure a control-plane server deployment | [Server configuration](https://github.com/cayu-dev/cayu/blob/main/docs/server-configuration.md) |
| Embed Cayu behind tenant-aware product APIs | [Server authentication and tenant isolation](https://github.com/cayu-dev/cayu/blob/main/docs/recipes/server-auth-tenancy.md) |
| Inspect supported model metadata | [Model catalog](https://github.com/cayu-dev/cayu/blob/main/docs/model-catalog.md) |
| Look up exact runtime behavior | [Runtime contracts](https://github.com/cayu-dev/cayu/blob/main/docs/runtime-contracts.md) |
| Track prerelease behavior and migrations | [Release notes](https://github.com/cayu-dev/cayu/blob/main/docs/release-notes.md) |

Maintainer-facing architecture is documented in
[Architecture](https://github.com/cayu-dev/cayu/blob/main/docs/architecture.md),
[Project layout](https://github.com/cayu-dev/cayu/blob/main/docs/project-layout.md),
and the [Glossary](https://github.com/cayu-dev/cayu/blob/main/docs/glossary.md).

## Examples

- [Examples index](https://github.com/cayu-dev/cayu/blob/main/examples/README.md):
  find the smallest reference for a capability.
- `cayu guide references#domain-tool`: credential-free domain-tool authoring
  and generator path.
- [Local environment runtime](https://github.com/cayu-dev/cayu/blob/main/examples/local_environment_runtime.py):
  files and commands.
- [Server example](https://github.com/cayu-dev/cayu/blob/main/examples/server_example.py):
  protected API and control plane.
- [Cloud PR reviewer](https://github.com/cayu-dev/cayu/blob/main/docs/recipes/pr-reviewer.md):
  durable task, isolated workspace,
  QA, and an explicit external effect.
- [Business approvals](https://github.com/cayu-dev/cayu/blob/main/docs/recipes/business-approvals.md):
  domain approval routing
  over the binary runtime primitive.
- [GitHub CLI through virtual egress](https://github.com/cayu-dev/cayu/blob/main/docs/recipes/github-cli-virtual-egress.md):
  an
  unmodified CLI with a virtual token, exact REST policy, and explicit mutation boundary.
- [Advanced runtime examples](https://github.com/cayu-dev/cayu/blob/main/examples/ADVANCED_RUNTIME_EXAMPLES.md):
  forks,
  compaction, taint isolation, speculative approval, and measured evidence.

Advanced examples are executable runtime specifications. Each example states
its evidence boundary instead of presenting one strategy as suitable for every
workload. Their measured results are described in
[Advanced runtime strategies](https://github.com/cayu-dev/cayu/blob/main/docs/advanced-runtime-examples.md).

## Contributing and security

Cayu contributors should read
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
