# Building applications with Cayu

This is the canonical, vendor-neutral authoring path for humans and coding
agents working from an installed Cayu package. Generated projects repeat only
the project-specific commands and registration rules in `AGENTS.md`.

The supported implementation loop is:

`understand -> clarify -> inspect -> check -> plan -> change -> test -> eval -> exercise -> report evidence`

## 1. Start from behavior, not framework objects

Before editing, identify:

- the users and the job they are trying to complete;
- triggers, domain inputs, and observable outputs;
- what the model decides and what deterministic code decides;
- autonomy, human oversight, and approval points;
- durable state, replay, interruption, and recovery expectations;
- external effects and the authority required for each effect;
- workspaces, execution environments, artifacts, credentials, and egress;
- representative successful, failing, interrupted, and recovery trajectories.

Ask one bounded clarification batch when an answer changes the security
boundary, durable data model, effect semantics, or acceptance behavior. Safe
defaults are appropriate for naming, local file layout, and reversible
presentation choices. Never silently choose recipients, destinations,
credentials, spending authority, destructive behavior, or ambiguous retry
semantics.

## 2. Select the smallest Cayu shape

Use only the concepts the application needs:

| Desired behavior | Cayu concept |
| --- | --- |
| One model-driven interaction | `CayuApp`, `AgentSpec`, `ModelProvider`, `RunRequest` |
| Deterministic action callable by a model | `Tool`, `ToolSpec`, explicit `ToolEffect` |
| Authority over effects | `ToolPolicy`; approval policies for human gates |
| Mutable files or command execution | `Environment` with an explicit `Workspace` and `Runner` |
| Durable uploaded/generated files | `ArtifactStore` |
| Long-lived conversation/recovery | durable `SessionStore` and checkpoint APIs |
| Background durable work | `TaskStore`, dispatcher, and an explicitly started worker |
| Deterministic orchestration | workflow helpers |
| Delegated model work | subagent tools and explicit child-session policy |
| Behavioral regression proof | runtime-native `EvalSuite` and trajectory assertions |

A conversational or research agent does not automatically need a workflow,
task queue, environment, approval step, memory store, server, or multi-agent
topology. A coding agent does not automatically need a shell: prefer narrow
file and domain tools; when command execution is necessary, use an explicit
runner and an enforcing command/tool policy.

## 3. Use the project factory

A Cayu project declares a synchronous factory in `pyproject.toml`:

```toml
[tool.cayu]
factory = "app:build_app"
```

Calling the factory constructs a fresh, process-scoped `CayuApp`. The app is
not a global registry or cross-process singleton. Durable stores coordinate
state between scripts, consoles, servers, workers, and tests. Importing project
modules must not construct the app, connect to services, migrate storage,
start workers/recovery/schedulers, or invoke a model or tool.

The factory may expose optional dependency-injection arguments for tests as
long as a normal zero-argument call remains valid. Tests should inject
`ScriptedModelProvider` and in-memory stores through those public seams.

## 4. Inspect before changing

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

## 5. Add one vertical slice

For a generated project, plan before applying:

```bash
cayu generate slice reviewer --tool assess_submission --effect none --dry-run --json
cayu generate slice reviewer --tool assess_submission --effect none
```

The planner does not import the app or write files. It reports exact proposed
contents and verification commands. Apply mode creates independent files and
changes only the delimited machine-owned import/registration regions in
`app.py`. Missing anchors, conflicting files, or different user content fail
without partial writes. Repeating a successful invocation is a no-op.

Generated code is a tracer bullet, not finished domain behavior. Replace its
placeholder tool body and eval prompt with the real domain contract while
preserving its closed input schema, explicit effect, visible registration,
runtime test, and trajectory assertion.

## 6. Treat effects as a security contract

Every tool declares one effect:

- `none`: no externally observable mutation;
- `idempotent`: an effect whose replay is safe under the domain's idempotency contract;
- `external`: an effect that may be observable or ambiguous after failure.

Declaration is metadata, not authorization. External tools require an
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

## 7. Put state in the right place

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

## 8. Prove behavior through public seams

The default credential-free proof is:

```bash
cayu inspect --json
cayu check --json
pytest
cayu eval run evals.assistant:build_eval
```

Tests should exercise `CayuApp.run` or `run_to_completion` with a
`ScriptedModelProvider`, not private methods or a fake replacement runtime.
Trajectory evals should assert domain behavior plus important runtime events:
tool calls, approval interruption, artifacts, child sessions, usage, or final
state as appropriate.

Report evidence in four separate layers:

1. static inspection and structured checks;
2. hermetic runtime tests and evals;
3. real process-boundary checks using the built wheel;
4. optional credential/infrastructure-gated live checks.

State exactly which commands ran and which were skipped. Successful imports,
construction, mocks, scripted providers, or a local runner do not prove live
provider access, sandbox isolation, network egress, or deployment readiness.

## 9. Shape-specific reminders

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
