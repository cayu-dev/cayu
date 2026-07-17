# Building applications with Cayu

This guide is the canonical concept map and implementation path for Cayu
applications. Generated projects repeat only their local commands and
registration rules in `AGENTS.md`.

For local development, the supported loop is:

`edit the requested behavior -> inspect -> check -> test -> eval`

## 1. Start with one model-only agent

In a fresh generated project, edit the existing agent, test, and eval in place.
Do not retain the starter and add a second agent. Give the agent a focused job,
domain input, and observable output. A system prompt is optional until the job
requires one, and a model-only agent needs no tools.

Use safe local defaults for reversible development choices. Ask questions when
the requested behavior itself is ambiguous. Questions about recipients,
credentials, spending authority, destructive effects, ambiguous retries,
durable recovery, and infrastructure begin when the user requests those
capabilities or asks to deploy.

## Cayu Map

Use only the concepts your agent needs. Start with the first row and add another
only when the requested behavior requires it.

| When you need it | Cayu concepts | Start here |
| --- | --- | --- |
| One model-driven agent | `CayuApp`, `AgentSpec`, `ModelProvider`, `RunRequest` | [`cayu new`](https://github.com/cayu-dev/cayu/blob/main/src/cayu/cli/scaffold.py), [application anatomy](https://github.com/cayu-dev/cayu/blob/main/src/cayu/guides/application-anatomy.md) |
| A provider-neutral run result | `run_to_completion`, `RunOutcome`, events | [application anatomy](https://github.com/cayu-dev/cayu/blob/main/src/cayu/guides/application-anatomy.md) |
| Model-specific routing or capabilities | provider registration, model catalog, thinking, structured output | [model catalog](https://github.com/cayu-dev/cayu/blob/main/docs/model-catalog.md), [structured output](https://github.com/cayu-dev/cayu/blob/main/examples/structured_output_live.py) |
| A capability outside the model | `Tool`, `ToolSpec`, `ToolContext` | [echo tool](https://github.com/cayu-dev/cayu/blob/main/examples/echo_tool_runtime.py) |
| Replay or mutation semantics | `ToolEffect`, idempotency keys | [tool effects](https://github.com/cayu-dev/cayu/blob/main/src/cayu/guides/tool-effects.md) |
| Authority or a human decision | `ToolPolicy`, approvals, user-input checkpoints | [business approvals](https://github.com/cayu-dev/cayu/blob/main/docs/recipes/business-approvals.md) |
| Files or commands during a run | `Environment`, `Workspace`, `Runner` | [local environment](https://github.com/cayu-dev/cayu/blob/main/examples/local_environment_runtime.py), [environment factories](https://github.com/cayu-dev/cayu/blob/main/docs/environment-factories.md) |
| Durable uploads or generated files | `ArtifactStore`, artifact/workspace bridges | [artifact example](https://github.com/cayu-dev/cayu/blob/main/examples/artifact_workspace_bridge.py) |
| Secrets or restricted network access | vaults, virtual credentials, egress policies | [virtual egress](https://github.com/cayu-dev/cayu/blob/main/docs/virtual-egress.md) |
| Tools exposed over MCP | MCP adapters and manifest policy | [stdio MCP](https://github.com/cayu-dev/cayu/blob/main/examples/stdio_mcp_runtime.py) |
| Conversation history that survives restarts | `SessionStore`, transcripts, checkpoints, resume | [runtime contracts](https://github.com/cayu-dev/cayu/blob/main/docs/runtime-contracts.md) |
| Context approaching a model limit | token counting, context policies, compaction, overflow recovery | [context counting](https://github.com/cayu-dev/cayu/blob/main/examples/context_counting_live.py) |
| Reviewed or retrievable knowledge | knowledge stores, review state, recall tools | [local knowledge](https://github.com/cayu-dev/cayu/blob/main/examples/knowledge_remember_local.py) |
| Durable background work | `TaskStore`, dispatcher, worker, event watcher | [triggering runs](https://github.com/cayu-dev/cayu/blob/main/docs/triggering-runs.md), [task worker](https://github.com/cayu-dev/cayu/blob/main/examples/task_worker_loop.py) |
| Deterministic orchestration | workflow helpers and runtime hooks | [workflow helpers](https://github.com/cayu-dev/cayu/blob/main/examples/workflow_helpers.py) |
| Delegated model work | subagent tools and child-session policy | [subagent example](https://github.com/cayu-dev/cayu/blob/main/examples/subagent_live.py) |
| Behavioral regression proof | `EvalSuite`, runtime assertions, replay | [evals](https://github.com/cayu-dev/cayu/blob/main/docs/evals.md) |
| Usage limits or cost control | usage events, run limits, budgets, pricing | [cost optimization](https://github.com/cayu-dev/cayu/blob/main/docs/cost-optimization.md), [usage summary](https://github.com/cayu-dev/cayu/blob/main/examples/usage_cost_summary.py) |
| Developer and operator inspection | `cayu inspect`, `cayu check`, console, dashboard, tracing | [console](https://github.com/cayu-dev/cayu/blob/main/docs/console.md), [OpenTelemetry](https://github.com/cayu-dev/cayu/blob/main/examples/otel_tracing.py) |
| An HTTP control plane | `cayu[server]`, authenticated FastAPI application | [server example](https://github.com/cayu-dev/cayu/blob/main/examples/server_example.py) |
| Advanced authority, isolation, caching, or speculation | composed runtime strategies with explicit evidence boundaries | [advanced runtime strategies](https://github.com/cayu-dev/cayu/blob/main/docs/advanced-runtime-examples.md) |

This map is a menu, not a checklist. A conversational, classification,
generation, or research agent does not automatically need a tool, workflow,
task queue, environment, approval step, knowledge store, server, or multi-agent
topology. Use the [examples index](https://github.com/cayu-dev/cayu/blob/main/examples/README.md)
to find the smallest runnable reference for an optional capability.

A tool-backed slice is optional. Add one only when the agent needs a real
capability outside the model. Prefer a narrow domain tool; when command
execution is necessary, use an explicit runner and enforcing command/tool
policy.

## 2. Use the project factory

A Cayu project declares a synchronous factory in `pyproject.toml`:

```toml
[tool.cayu]
factory = "app:build_app"
eval_target = "evals.agent:build_eval"
```

Calling the factory constructs a fresh, process-scoped `CayuApp`. The app is
not a global registry or cross-process singleton. Durable stores coordinate
state between scripts, consoles, servers, workers, and tests. Importing project
modules must not construct the app, connect to services, migrate storage,
start workers/recovery/schedulers, or invoke a model or tool.

The factory may expose optional dependency-injection arguments for tests as
long as a normal zero-argument call remains valid. Tests should inject
`ScriptedModelProvider` and in-memory stores through those public seams.
The separate `eval_target` returns an eval plan and lets `cayu eval run` use the
project's default suite without treating the application factory as an eval.

## 3. Inspect before changing

Run from the project root or any nested directory:

```bash
cayu inspect --json
cayu check --json
```

`inspect` builds the configured factory once and returns the versioned,
redacted application manifest. It describes configuration and static
resolution; it does not invoke a provider, tool, environment factory, worker,
watcher, session, or recovery path. Capability fields distinguish declared and
resolved configuration from process availability and live verification.
Read-only describes Cayu's inspection phase, not arbitrary code inside the
project-owned factory: `inspect` must call that factory, so keep factory boot
effects limited to constructing the application graph.

After factory construction, `check` evaluates only that manifest. It performs
no live probes and does not mutate the application, stores, or source tree.
Exit status `0` means no finding at the selected threshold, `1` means findings,
and `2` means discovery, import, factory, or invocation failure. Each finding
gives a stable code, machine path, observed parameters, correction, and
verification command.

## 4. Add a tool-backed slice only when needed

Do not use the generator for the first model-only agent. When the application
actually needs an additional tool-backed slice, plan before applying:

```bash
cayu generate slice reviewer --tool assess_submission --effect none --dry-run --json
cayu generate slice reviewer --tool assess_submission --effect none
```

The planner does not import the app or write files. It reports exact proposed
contents and verification commands. Apply mode creates independent files and
changes only the delimited machine-owned import/registration regions in
`app.py`. Missing anchors, conflicting files, or different user content fail
without partial writes. Repeating a successful invocation is a no-op.

Generated code is a tracer bullet, not finished domain behavior. A generated
slice carries an explicit `AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET`
marker, so `cayu check --fail-on warning --json` rejects it as a completed
submission while its structural inspection, runtime test, and eval remain
runnable.

Replace the domain system prompt, tool schema and body, runtime test inputs and
assertions, and trajectory eval behavior and assertions. Only then remove the
`authoring_state` argument and unused `AgentAuthoringState` import from the
generated agent module. Cayu deliberately trusts that explicit state instead
of parsing arbitrary Python, prompts, or test source; clearing it is an author
claim, not a framework proof of domain correctness.

Generated slices define each tool name once in the tool module and reuse that
constant in the `ToolSpec`, agent instructions, `workflow_tool_names`, runtime
test, and eval. Preserve that single source when renaming a tool. For
hand-authored agents, declare every exact tool name that machine-owned workflow
instructions expect through `AgentSpec.workflow_tool_names`; do not maintain an
independent list in prose or tests.

## 5. Treat effects as a security contract

Every tool declares one effect:

- `none`: no externally meaningful durable mutation;
- `idempotent`: may mutate durable state, but a stable downstream idempotency
  key or equivalent contract collapses replay;
- `external`: non-idempotent or outcome-ambiguous mutation that generic retry
  must not assume is safe to repeat.

Use `cayu guide tool-effects` for the canonical decision table. Transport,
billing, observability, and a name such as "read" do not determine the effect.
Declaration is replay metadata, not authorization. External tools require an
enforcing `ToolPolicy`; use an approval policy when a person must decide.
Comments, prompts, UI confirmation, allowlists that are not consulted by the
runtime, and tests that bypass policy are not enforcement.

Define what happens when an effect starts but completion cannot be proven.
Never blindly retry an ambiguous external effect. Persist enough identity and
checkpoint state for an operator or recovery path to reconcile it.

### Keep model-controlled command selectors as data

Model-controlled command selectors are untrusted argv input. A value described
as a path, target, test node ID, or filter can still become a new option when
application code appends it to an otherwise fixed command. For example,
`--help` can exit zero without running a check, while an output option can write
outside the workspace. A zero exit status alone is therefore not proof that the
intended check ran.

An executable allowlist does not authorize its argument protocol. For every
allowed command, define and validate the exact selector grammar owned by the
application. A file-selector recipe should:

1. reject empty values, NULs, leading-option forms, absolute paths, traversal,
   platform separators outside the supported grammar, and unsupported syntax;
2. parse compound forms such as a test path plus node IDs before validating
   each component;
3. resolve the path against the authorized workspace and prove containment;
4. construct argv as a sequence with no shell interpolation; and
5. insert `--` only when the target executable's documented, tested semantics
   treat it as an end-of-options delimiter at that exact position.

For a tool that supports only Python test files and simple pytest node IDs, an
application-owned validator can look like this:

<!-- cayu-guide-include:pytest-selector -->

The fixed prefix and validated selectors can then be passed directly to the
runner without a shell. This example deliberately makes no claim that adding
`--` is valid for every pytest invocation; confirm the installed executable's
contract before doing so. Filters passed as values to an application-owned
option need their own closed grammar rather than reuse of the path validator.

Classify process outcomes using that executable's contract. Keep rejection,
unavailable or non-executable command, timeout, check failure, and zero tests
executed distinct from verified success. Preserve a structured cannot-run reason
such as not found, permission denied, invalid executable format, or another OS
launch error. Report whether the check used full discovery or an intentionally
selected subset, including the exact validated selectors; a passing selected
check verifies only that subset.

Inventory writes made by representative success and adversarial runs, including
caches and reports, and compare those observed effects with the tool's declared
`ToolEffect`. Keep the process outcome and effect comparison orthogonal: a
failed check that also writes remains a failed check with a separate effect
mismatch. These controls do not replace container or microVM isolation when the
command executes untrusted repository code.

## 6. Put state in the right place

- Transcript/session state belongs in a `SessionStore`.
- Durable work ownership and results belong in a `TaskStore`.
- Curated/retrievable knowledge belongs in a knowledge store; project skills
  and instructions remain human-readable files.
- Mutable working files belong in a `Workspace`.
- Stable uploads and outputs belong in an `ArtifactStore`.
- Commands run through a `Runner`; a local runner is not a sandbox.
- Secrets come from an explicit vault/provider configuration and must not enter
  manifests, diagnostics, events, prompts, generated plans, or repository files.

Use an environment only when tools need these execution capabilities. Bind and
finalize workspaces explicitly. When recovery or reconnect matters, verify the
same identity and naming contract used by the original run.

Environment selection is opt-in. Pass `default=True` to
`register_environment(...)` or `register_environment_factory(...)` only when
unnamed `RunRequest`s should use that environment. Otherwise leave it
non-default and set `RunRequest.environment_name` explicitly. Registering the
first environment never makes it the default implicitly.

## 7. Prove behavior through public seams

The default credential-free proof is:

```bash
cayu inspect --json
cayu check --json
pytest
cayu eval run
```

Tests should exercise `CayuApp.run` or `run_to_completion` with a
`ScriptedModelProvider`, not private methods or a fake replacement runtime.
Trajectory evals should assert domain behavior plus important runtime events:
tool calls, approval interruption, artifacts, child sessions, usage, or final
state as appropriate.

Run `cayu check --json` to compare each agent's declared
`workflow_tool_names` with the tools registered for that same agent in the
public manifest. Unknown, stale, renamed, removed, or agent-mismatched names are
authoring errors. This check reads the explicit workflow contract; it does not
guess tool references from arbitrary natural-language prompt text.

`ScriptedModelProvider` can prove runtime handling of predetermined calls, but
it cannot prove prompt comprehension, model tool choice, or live-provider
behavior. Keep manifest-backed prompt/tool alignment evidence separate from a
scripted trajectory. Optional live-provider evidence may exercise comprehension
and tool choice, but remains credential-gated and is not required for hermetic
CI.

Report evidence in four separate layers:

1. static inspection and structured checks;
2. hermetic runtime tests and evals;
3. real process-boundary checks using the built wheel;
4. optional credential/infrastructure-gated live checks.

State exactly which commands ran and which were skipped. Successful imports,
construction, mocks, scripted providers, or a local runner do not prove live
provider access, sandbox isolation, network egress, or deployment readiness.

Before deploying a tool declared `NONE`, use the bounded workspace check in
`cayu guide tool-effects` when workspace mutation is part of its risk boundary.
The check is explicit and behavioral; `cayu check` never executes tools. Treat
an unchanged workspace as scoped evidence only, and use domain-specific tests
for databases, external services, idempotency, and effects outside that first
supported observer.

## 8. Shape-specific reminders

- **Conversation:** preserve transcript/recovery semantics; omit tasks and
  environments unless behavior needs them.
- **Research/documents:** make source inputs and artifact outputs explicit;
  eval citations and document decisions, not only final prose.
- **Coding/repositories:** use a repository binding, isolated workspace/runner,
  narrow command policy, and human confirmation before commit/push/PR actions.
- **Operations:** model idempotency, ambiguity, approvals, budgets, and restart
  recovery before adding autonomous effects.
- **Durable workflows:** keep deterministic orchestration outside model prompts;
  use model steps only where judgment is required.
- **Multi-agent:** justify each role, bound delegation, persist lineage, and eval
  both child behavior and parent synthesis.

Finish by rerunning inspection, checks, focused tests, the relevant eval, and
any explicitly available process/live checks. Report limitations rather than
substituting weaker evidence.
