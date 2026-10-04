# Building applications with Cayu

For a complete installed-package support agent with restartable clarification and
external approval receipts, run `cayu guide order-support`.

This guide is the canonical concept map and implementation path for Cayu
applications. Generated projects repeat only their local commands and
registration rules in `AGENTS.md`.

For local development, the supported loop is:

`edit the requested behavior -> inspect -> check -> test -> eval`

Configure reasoning through `ThinkingConfig` on the agent, run/resume request, or
workflow `StepRunOptions`. Effort values are provider-specific; consult
`cayu guide thinking` for the exact vocabulary,
precedence, local rejection rules, and backend acceptance boundaries.

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
| Create or extend a generated application | complete scaffold convention, normalized plan, explicit registration | `cayu guide applications` |
| One model-driven agent | `CayuApp`, `AgentSpec`, `ModelProvider`, `RunRequest` | `cayu guide anatomy` |
| A provider-neutral run result | `run_to_completion`, `RunOutcome`, events | `cayu guide anatomy#verify-the-contract` |
| Model-specific routing or capabilities | provider registration, model catalog, thinking, structured output | `cayu guide providers`, `cayu guide structured-output` |
| A capability outside the model | `Tool`, `ToolSpec`, `ToolContext` | `cayu guide references#domain-tool` |
| Replay or mutation semantics | `ToolEffect`, idempotency keys | `cayu guide tool-effects` |
| Notify a person about a durable pause | pending actions, attention identity, durable sink handoff, reconciliation | `cayu guide human-attention` |
| Durable operational changes | proposal, policy-bound approval, action receipt, verification, recovery | `cayu guide durable-operations` |
| Rebuild service-backed tools after a restart | component behavior identities, injected or environment-bound knowledge | `cayu guide durable-service-tools` |
| Authority or a human decision | `ToolPolicy`, approvals, user-input checkpoints | `cayu guide references#approvals` |
| Files or commands during a run | `Environment`, `Workspace`, `Runner` | `cayu guide references#environments` |
| Application workspace references or inventories | `WorkspaceReferenceBinding`, `ToolContext.require_workspace_binding` | `cayu guide authoring#workspace-references-and-inventories` |
| Durable uploads or generated files | `ArtifactStore`, artifact/workspace bridges | `cayu guide references#artifacts` |
| Secrets or restricted network access | vaults, virtual credentials, egress policies | `cayu guide references#secrets-egress` |
| Tools exposed over MCP | MCP adapters and manifest policy | `cayu guide references#mcp` |
| Conversation history that survives restarts | `SessionStore`, transcripts, checkpoints, resume | `cayu guide references#sessions` |
| Context approaching a model limit | token counting, context policies, compaction, overflow recovery | `cayu guide references#context` |
| Reviewed or retrievable knowledge | knowledge stores, review state, recall tools | `cayu guide references#knowledge` |
| Durable background work | `TaskStore`, dispatcher, worker, event watcher | `cayu guide references#background-work` |
| A task done only when an independent check accepts it | `WorkContract`, `VerifiedTaskWorker`, completion verifier, result resolver | `cayu guide verified-work` |
| Deterministic orchestration | workflow helpers and runtime hooks | `cayu guide references#workflows-hooks` |
| Delegated model work | subagent tools and child-session policy | `cayu guide references#subagents` |
| Behavioral regression proof | `EvalSuite`, runtime assertions, replay | `cayu guide references#evals` |
| Usage limits or cost control | usage events, run limits, budgets, pricing | `cayu guide references#cost-control` |
| Developer and operator inspection | `cayu inspect`, `cayu check`, console, dashboard, tracing | `cayu guide references#observability` |
| An HTTP control plane | `cayu[server]`, authenticated FastAPI application | `cayu guide references#server` |
| A public or multi-user agent product | `cayu new NAME --preset service`, maintained tenant-safe service factory | `cayu guide references#server` |
| A maintained repository-coding starter | `cayu new NAME --preset coding`, explicit workspace, knowledge, reviewer, and input composition | `cayu guide authoring#coding-composition` |
| A product API around repository coding | application-owned authentication, durable intake, workers and independent acceptance | `cayu guide authoring#coding-product-host` |
| Advanced authority, isolation, caching, or speculation | composed runtime strategies with explicit evidence boundaries | `cayu guide references#advanced-runtime` |

This map is a menu, not a checklist. A conversational, classification,
generation, or research agent does not automatically need a tool, workflow,
task queue, environment, approval step, knowledge store, deployed application
server, or multi-agent topology. The package-shipped `cayu guide references`
topic is the offline index for optional capabilities.

The local developer/operator control plane is a standard inspection surface,
not part of the agent's domain behavior. Fresh generated projects install the
server extra through their `dev` extra; run `uv run cayu serve --dev` and open
`http://127.0.0.1:8000/cayu/`. The explicit `--dev` flag is for trusted local
use. A deployed control plane still requires configured authentication, and an
application that already owns FastAPI embeds Cayu explicitly with
`mount_cayu(..., path="/cayu")`. Never use `OpenAccess()` on a public listener.
Client-IP and forwarded-header checks are not authentication; public or deployed
mounts require `AuthenticatedAccess(...)` with application-owned authorization.

## Package ownership

Use the concept package when exploring a capability: `cayu.sessions` owns
session requests and stores, `cayu.tools` owns tool contracts and policies,
`cayu.context` owns context management, and `cayu.approvals` owns approvals and
user input. Other owners include `cayu.tasks`, `cayu.workflows`, `cayu.memory.base`,
`cayu.knowledge`, `cayu.budgets`, `cayu.snapshots`, and `cayu.delivery`.

```python
from cayu.applications import CayuApp
from cayu.agents import AgentSpec
from cayu.messages import Message
from cayu.sessions import RunRequest
from cayu.tools import Tool, ToolSpec
```

Root imports such as `from cayu import CayuApp` remain supported. Package
`_exports.py` files identify the defining modules; matching `__init__.pyi` files
provide static declarations. `cayu.runtime` coordinates execution. The prerelease
migration removes legacy module aliases; use current concept paths for new
imports and generated code. Same-build recovery remains supported; old import
paths and cross-build execution-profile continuity are not compatibility promises.

The [complete source ownership and migration map](https://github.com/cayu-dev/cayu/blob/main/docs/public-concepts.md)
includes specialized modules and retained root-level concepts. This package's
installed source and export manifests are authoritative for its version.

## Coding composition

For a repository-coding application that needs the same explicit capabilities
on day one, use `cayu new NAME --preset coding`. The opt-in project assembles
existing public tools and stores in the canonical `tools/`, `policies/`,
`environments/`, `operations/`, `knowledge/`, `prompts/`, and `agents/` modules:
bounded file and `rg` search operations, Git change review, local artifacts,
reviewed durable knowledge, a bounded background reviewer with result recovery,
and human input. Root `composition.py` remains only a compatibility import.
It requires `git` and `rg`, creates a clean initial Git commit, and ships one
credential-free smoke for the complete composition. Its trusted-host local
workspace and runner are not a sandbox. Keep the default scaffold for jobs that
do not need these capabilities, and do not combine the coding composition with
the multi-user service template.

This is a restriction on combining generated templates, not a prohibition on
hosting a coding application behind an application-owned API. For that advanced
composition, use `cayu guide authoring#coding-product-host`; it does not inherit
the service template's authentication or verification just by adding routes.

Selecting this composition chooses implementations; it does not grant authority.
Its exposure policy separately controls model-visible tools, and its ordinary
tool policy, approval policy, and runtime gates independently authorize calls.

For a public or multi-user product, start with
`cayu new NAME --preset service`. That template keeps customer authentication,
tenant-qualified application storage, public identifiers, allow-listed product
responses, and the operator-only `/cayu/` mount in one maintained service
factory. Its deployment check proves only the declarative factory posture; the
generated assembled-ASGI security suite proves the product behavior. Arbitrary
host-owned routes remain outside Cayu's verification boundary.

Online, the repository's
[examples index](https://github.com/cayu-dev/cayu/blob/main/examples/README.md)
is a secondary catalog of runnable source examples.

A tool-backed slice is optional. Add one only when the agent needs a real
capability outside the model. Prefer a narrow domain tool; when command
execution is necessary, use an explicit runner and enforcing command/tool
policy.

## Coding product host

Start with one generated coding project and retain its actual scaffold metadata.
Do not overlay a second generated service project, relabel its preset, or expose
the operator control plane as a customer API. Build the product host explicitly
using public Cayu APIs and the following existing owners:

| Boundary | Application home and implementation responsibility |
| --- | --- |
| Construction | `app.py` constructs and registers the same application graph in each process; it does not start workers, migrate stores or perform model calls. |
| Durable state | `configuration/` constructs shared session/task/artifact backends. Application-owned business reservations need their own durable transaction owner; a hash of an intake request is not an atomic reservation. |
| Spending authority | `policies/` configures matching budget scopes, explicit pricing and conservative reservations against a shared durable ledger. A per-operation causal limit is not a deployment-wide total cap; verify the actual run identity and refusal before provider dispatch with `cayu guide references#cost-control`. |
| Product intake | `operations/` authenticates the customer, obtains tenant/subject from that authentication, captures the admitted source and configuration, and reserves stable public/product/session/task identities under an idempotency key. Resolve reads by authenticated tenant and public ID; return an allow-listed product projection, not raw Runtime records. |
| Task handoff | After reserving identity, use `CayuApp.create_task(TaskCreate(...))`. Reservation and task insertion can commit separately: retry the same reservation and task identity, read back an ambiguous insertion, and reject conflicting contents rather than allocating replacement work. |
| Worker lifetime | A separate `operations/` role uses `run_task_worker` with `TaskQuery` for its phase. Reconstruct the original reservation from the store-owned claim and validate its current ownership before executing. Use `complete_managed_task` after independent acceptance; do not create another lease or retry engine. |
| Coding result | `workflows/` coordinates the maintained coding product; `domain/` checks the admitted revision, permitted changes, required checks and independent verifier against retained artifact evidence. `CodingProductArtifactRepository` provides request/publication readback. Model text, successful project tests or task completion alone are not independent acceptance. |
| Operator host | Mount the control plane with `cayu.server.mount_cayu` and explicit `AuthenticatedAccess`. Protect application operator routes separately; customer authentication neither grants operator access nor authorizes another tenant's resources. |
| Delivery | `integrations/` owns separately approved Git/GitHub handoffs. Keep patch readiness, exact reviewed change/destination approval, publication, hosted checks and final delivery distinct. No approval is implied by producing a patch. |

First test the assembled HTTP application with injected providers and stores:
authentication refusal, cross-tenant lookup, duplicate intake, conflicting retry,
independent acceptance and truthful pending/failed responses. Then exercise the
same application with its installed distribution and persistent deployment.
Keep source, Runtime/artifact state and delivery state separate; credentials
stay outside the admitted coding guest. Prepare schemas explicitly and run
homogeneous application versions with coordinated backup and restart.

Check the actual factory before building the deployment image: run `cayu check`
from the generated project with its intended configuration. Keep environment
reads, collection-building calls and other construction inside the synchronous
factory or runtime functions, not module import expressions. An importable module
can still violate the scaffold's import-side-effect contract. Preserve the
generated enforcing policies where possible; a custom policy returning "allow"
does not establish inspectable external-tool coverage. Resolve
`cayu guide diagnostics#external-tool-coverage-unknown` rather than suppressing
that error or treating HTTP startup as a passing deployment check.

For narrower coding authority, extend the generated concrete
`ParameterConstrainedToolPolicy` rules in `policies/` and wire that policy at
the existing construction seam. Preserve the existing path and selector rules;
append a `RequiredAllowlistRule("path", values=[...])` to restrict writes to
specific paths. Do not replace those rules with a custom delegating `ToolPolicy`
or subclass merely to add a restriction: inspection cannot prove that arbitrary
Python calls its underlying enforcement. An `execution_profile_identity` binds
behavior for reconstruction; it does not certify static policy coverage.
Give the changed policy its own explicit, versioned behavior identity and
advance that identity when its enforcement changes; do not reuse the original
generated policy's identity for different behavior.
The maintained `StructuredCommandToolPolicy` preserves known base coverage,
but cannot make an unknown custom base statically trusted. Run `cayu check`
again after composition changes, before deployment or worker dispatch.

### Queue reserved work

A business reservation may choose a future session ID, but that is not a native
task attachment. Leave `TaskCreate.session_id` unset when enqueueing ordinary
work: `run_task_worker` claims only unattached pending tasks. Setting that field
early makes the task ineligible even if the named session does not exist. Keep
the intended session ID in the durable business reservation; the claimed worker
resolves it there, and Runtime admission owns any native task/session attachment.
Never clear an existing attachment merely to make a task claimable.

For example, after authenticating and resolving the original durable reservation:

```python
from cayu import TaskCreate


async def enqueue_reserved_coding_task(app, reservation):
    return await app.create_task(
        TaskCreate(
            task_id=reservation.task_id,
            type="coding_product",
            assigned_agent_name="coding",
            input={"public_id": reservation.public_id},
        )
    )
```

The worker must load that same reservation and validate its task, tenant, source
and execution authority before using its intended session ID. On ambiguous task
insertion, read back and compare the original task instead of choosing new IDs.

For a detached coding product, a worker handler can finish through this existing
managed boundary. Here `verify_product_for_claim` is an application-owned function
that must resolve the original reservation, run or recover the maintained product,
and independently validate its exact artifact and cleanup evidence. It is not an
agent-text parser or a substitute for an independent verifier:

```python
from cayu import complete_managed_task


async def handle_coding_task(app, claimed, worker_id):
    verified = await verify_product_for_claim(app, claimed)
    await complete_managed_task(
        app.task_store,
        claimed,
        worker_id,
        {
            "product_run_id": verified.product_run_id,
            "result_digest": verified.result_digest,
        },
    )
```

`run_task_worker` owns the heartbeat while this handler executes.
`complete_managed_task` uses its latest acknowledged lease and returns a durable
`Task`; it does not create a caller-keyed `TaskTerminalizationReceipt`. Do not
load a lease and construct a raw `TaskTerminalizationRequest` alongside that
heartbeat merely to obtain such a receipt. Extending the lease duration does not
remove the race. Use the original claimed task with the managed helper.

If completion's reply is lost, read the original task through `TaskStore.load_task`
and validate its type, session, completed status and exact result against the
saved reservation and independently verified product artifact. That durable
task/result is normal-completion evidence; it need not have a separately keyed
terminal receipt. Do not rerun coding, manufacture a receipt, or complete a still
claimed task from a read-only screen. Cancellation reconciliation uses a different
explicit receipt, as described below and in `cayu guide references#background-work`.

Run schema creation and validation with the same pinned Python runtime as the
API and workers, preferably from their deployment image. Matching Cayu versions
alone is insufficient: durable transcript indexes also bind Python's Unicode
tokenizer identity. A host-created database can therefore reject a container
using a different Python runtime. Preserve an incompatible database and its
evidence; do not rewrite its tokenizer metadata or erase accounting to pass
startup. Choose the matching runtime or separately initialize a new disposable
database with the intended runtime before a fresh trial.

For worker loss, use registered recovery through `cayu recovery plan` and
`cayu recovery execute` with the original application and admitted workspace.
A dead worker, expired lease or cancelled await does not prove an external
effect stopped. Require authoritative effect and invocation-release evidence
before reconciling the outer task. Unknown effects remain fenced; cleanup-only
recovery must not redispatch the model. Prove native resource disposal or a
supported retained recovery owner separately from session/task terminal status.
Do not label cancellation settlement as a verified repair.

Observe and reconcile the retained native allocation and command journal before
destructive disposal. Removing a guest first can destroy the only reconnectable
workspace or terminal-command receipt needed by registered recovery. Use the
registered resource owner and its allowed disposition; an unrelated Docker
cleanup script cannot substitute for native recovery settlement.

Recovery must retain the original invocation controls, not just the same agent
and workspace. In `ToolRoundRecoveryRequest`, omitted continuation controls use
the recorded invocation. Changing `max_steps` from eight to one still changes
the execution profile; it is not an observation-only switch. Ordinary manual
tool-round recovery may continue the model after settling the round. If execution
must not continue, select a supported interruption or failure disposition whose
owner can settle the retained effect and resources; do not invent tool success,
delete its recovery evidence, or relax profile validation. See
`cayu guide tool-effects#native-durable-recovery-evidence`.

Keep application-owned recovery diagnostics provisional rather than storing
`recovery_required` as an immutable final result. The application needs an exact
transaction that consumes the validated native terminal task/result or explicit
cancellation-reconciliation receipt, settles the original business reservation,
and releases its source fence only after effect settlement and required resource cleanup are positively
verified. Unresolved work retains its fence and a supported recovery owner.
Preserve prior diagnostics and reject changed task/session/source or evidence.
When consuming an explicit receipt, make replay of the same receipt idempotent.
Native session recovery alone does not perform this application transaction;
do not replace it with a scenario-specific script
that edits business rows or assumes process death proves quiescence.

This composition remains application-owned. A successful `cayu check` or a
declared extension is structural evidence, not proof of its authentication,
acceptance, deployment or recovery behavior. See `cayu guide references#server`,
`cayu guide references#background-work`, and `cayu guide tool-effects` for the
reusable boundaries; preserve the generated convention rather than adding a
parallel orchestration framework.

## coding-host-settlement-example

The installed example `cayu.guides.coding_host` extends the generated coding
application rather than implementing another recovery engine. Its SQLite
`BusinessStore` is application state, separate from the Runtime task/session
stores. It is a single-tenant, bounded local-host recipe, not a production
multi-host database adapter or an HTTP authentication layer.

From a generated coding project, wrap its normal factory result before starting
the API or task worker. This deployment must already have its reviewed priced
budget policy configured:

```python
from pathlib import Path

from app import build_coding_product_application
from cayu.guides.coding_host import BusinessStore, extend_generated_application

business = BusinessStore(Path(".cayu/coding-business.sqlite"))
configured = build_coding_product_application()
pricing = configured.app.budget_policy.limits[0].pricing
product = extend_generated_application(
    configured, business=business, tenant="disposable-example",
    pricing=pricing, cost_basis="observed",
)
```

The extension uses the generated, project-owned preparation method to reserve the
source before execution. Recovery checks the existing reservation and never
synthesizes missing intake provenance. Retain the complete original
`CodingProductTask`, including its instruction and controls, in the existing
authenticated intake/task payload; a native request fingerprint cannot reconstruct
that input. Keep the original product/task/session IDs.
The managed handler still owns ordinary task completion: record exactly
`{"product_run_id": publication.candidate.product_run_id,
"result_digest": publication.result_reference.digest}` using
`complete_managed_task`. Then call `product.settle_completed(task,
result_digest=..., pricing=..., cost_basis="observed")`. The same method handles
completion acknowledgement loss after reading the exact completed task; it never
reruns the handler. Patch readiness is not independent application acceptance,
human approval, or successful external delivery.

For cancelled work, first execute supported registered native recovery. The
example's `product.settle_cancelled` requires an authenticated `ResolutionActor`,
the original task, a stable reconciliation ID, pricing and an explicit cost basis.
Its default worker observer uses the existing maintained Docker-host owner:
run the trusted `maintenance.coding` worker with `maintenance_worker_id` from
`cayu.guides.coding_host_owner` and `CAYU_MAINTENANCE_WORKER_OWNER=docker`.
Pass `await maintenance_worker_id("maintenance.coding")` as the native
`run_task_worker` worker ID. This observer's supported container command is
`cayu worker coding --shutdown-grace-seconds 30`; its trusted service image needs
`/usr/local/bin/docker` and authorized access to the local Docker socket. A
different supervisor needs its own reviewed generation observer, not a boolean
or an inference from PID absence.
It inspects the exact container lifetime on the local daemon; no missing-container
or lease-expiry inference substitutes for a stopped generation. The native
inspector must independently prove invocation release, and the maintained
file/check effect validator rejects unknown/custom effects. This recipe neither
disposes guests nor authorizes recovery actions that native planning refuses.
Keep the source fenced when any evidence is unavailable.
This example settles patch-ready completion and owner-lost cancellation, not
arbitrary failed tasks or unsupported custom effects. Those remain explicitly
reserved for the application's operator/recovery owner; do not reset the task or
use the internal transaction method as a substitute for positive evidence.

The complete native reconciliation request is saved before its store call.
Retries reuse its actor, evidence, timestamps and idempotency key. The application
then records the native receipt and result in the same transaction that releases
its source reservation. A lost reply at either boundary is reconciled without a
new task or model invocation. Do not expose `BusinessStore.prepare/settle` directly
as HTTP operations: they are internal application transaction primitives.

Both terminal paths use `project_result`. Select `synthetic` only in explicitly
controlled fixtures; missing usage never implies free work. Cost is a recorded
estimate under the identified price book, not complete billing. Historical reads
use `business.read(reservation)` and remain valid after later source changes.
Before a **new delivery**, use `require_current_source` under the application's
exclusive source/delivery owner: it compares a bounded byte-level observation to
the authenticated retained result, not just Git status. Keep that owner and the
delivery's own exact approval checks through publication. This observation alone
does not authorize any Git or GitHub action.

The reservation pins the complete price book and cost basis. Reconciliation
rejects a changed pricing/basis configuration rather than relabeling a synthetic
trial as observed provider spending after restart.

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
resolved configuration from process availability and live verification. Agent
entries expose `has_system_prompt` so prompt presence is inspectable, but the
manifest deliberately omits prompt content and implementation bodies. The
application fingerprint is structural: prompt absence versus presence changes
it, while editing one non-empty prompt into another does not. Prove prompt
behavior through runtime tests and evals rather than treating the fingerprint
as a content digest.
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

Do not generate a tool for the first model-only agent unless the requested job
actually needs a capability outside the model. When it does, attach the first
tool tracer bullet to the existing starter:

```bash
cayu generate tool assess_submission --agent reviewer --effect none --dry-run --json
cayu generate tool assess_submission --agent reviewer --effect none
```

For an additional agent with its own tool-backed slice, plan before applying:

```bash
cayu generate slice reviewer --tool assess_submission --effect none --dry-run --json
cayu generate slice reviewer --tool assess_submission --effect none
```

The planner does not import the app or write files. It reports exact proposed
contents and verification commands. Apply mode creates independent files and
changes only the delimited machine-owned import/registration regions in
`app.py`. Missing anchors, conflicting files, or different user content fail
without partial writes. Repeating a successful invocation is a no-op.

Generated tools declare `execution_profile_identity=ExecutionProfileBehaviorIdentity(...)`
in their class-level `ToolSpec`. Keep that declaration on every custom tool you
write, including read-only ones: sessions record the tool set's identities, so a
tool without one makes a paused or resumed session fail with
`ExecutionProfileMismatchError` (changed in `tool_implementations`) as soon as
tools change. Bump `behavior_version` when a tool's behavior changes. Cayu does
not invent a default identity, because a guessed identity would let changed
behavior resume a session recorded with the old one.

Generated code is a tracer bullet, not finished domain behavior. A generated
slice carries an explicit `AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET`
marker, while the scaffold's first-tool flow sets `_AUTHORING_STATE` to
`"unfinished_generated_tracer_bullet"`. In either case,
`cayu check --fail-on warning --json` rejects the agent as a completed
submission while its structural inspection, runtime test, and eval remain
runnable.

Replace the domain system prompt, tool schema and body, runtime test inputs and
assertions, and trajectory eval behavior and assertions. Only then clear the
marker. For `cayu generate tool`, change `_AUTHORING_STATE` to `None` in the
generated agent-config region. For `cayu generate slice`, remove the
`authoring_state` argument and unused `AgentAuthoringState` import from the
generated agent module. Cayu deliberately trusts that explicit state instead
of parsing arbitrary Python, prompts, or test source; clearing it is an author
claim, not runtime-verified proof of domain correctness.

Generated slices define each tool name once in the tool module and reuse that
constant in the `ToolSpec`, agent instructions, `workflow_tool_names`, runtime
test, and eval. Preserve that single source when renaming a tool. For
hand-authored agents, declare every exact tool name that machine-owned workflow
instructions expect through `AgentSpec.workflow_tool_names`; do not maintain an
independent list in prose or tests.

For a complete credential-free composition of durable work, per-session local
environments, guarded file/command tools, failure recovery, and
application-owned verification, see `examples/durable_file_workflow`. Its
`verified.py` runs the same job as a contract-bound task with an independent
verifier and bounded attempts; see `cayu guide verified-work`. “File worker”
describes that example's job; it is not an `AgentSpec` profile or a new runtime
concept.

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

### Report expected failures as failed results

When a tool fails in a way you expected, such as a record not found, input
rejected, or the service returning an error response, return
`ToolResult(content="what went wrong", is_error=True)`. The model sees the
failure and can retry, ask, or explain it, and Cayu records a failed call. A
failure returned as ordinary content, for example `{"error": ...}` with
`is_error` left false, looks like a success to Cayu and to loop policies.

Raising behaves differently depending on the declared effect. From a `none` or
`idempotent` tool, the exception also reaches the model as a failed result.
From an `external` tool, which is the default when no effect is declared, it
means the effect's outcome is unknown: Cayu interrupts the session for
reconciliation instead of continuing. Raise only when that is true.

### End each turn through a known tool

If every turn must end in a particular tool, such as a clarification question
or a recorded proposal, pass `RequireFinalTool([...])` in `loop_policies` on
each run and resume. When the model stops before one of those tools has
returned without error, it gets a reminder instead of the turn silently
completing with nothing to show. See `cayu guide order-support`.

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

Treat workspace IDs as opaque identities owned by the selected workspace.
Never reconstruct an ID from an environment name, filesystem path, hash, or
another environment builder's naming convention. For an explicitly registered
static environment, obtain it from
`app.get_environment(name).environment.workspace.id` after checking that the
workspace exists. Inside tools, use `ToolContext.workspace_id` and the active
`ToolContext.workspace`; factory-created or bound workspaces may not exist until
the session has materialized them. An environment name selects execution
capabilities; it is not the workspace's identity.

Bind application-owned file inventories and receipts to that actual workspace
identity and validate it before accepting their authority. Runtime cannot infer
the semantics of arbitrary application metadata: a string named `workspace_id`
inside that metadata does not establish a Runtime-validated binding. Fail with
an explicit identity-mismatch diagnostic instead of silently treating every
declared file as unavailable. Prove the contract through a native run with an
independently assigned workspace ID, including a different environment builder
or evaluation wrapper when the application supports one.

Environment selection is opt-in. Pass `default=True` to
`register_environment(...)` or `register_environment_factory(...)` only when
unnamed `RunRequest`s should use that environment. Otherwise leave it
non-default and set `RunRequest.environment_name` explicitly. Registering the
first environment never makes it the default implicitly.

### Continue after a process crash

Use `app.resume(ResumeRequest(...))` for the customer's next message on an
existing session. With native stores, continuation recovers a `running` or
`interrupting` session when its recorded executor is provably gone on this
host, or its execution lease has expired. It records the takeover as
`session.run_fenced` before continuing. Approval, user-input and provider-operation
resolution use the same admission check before their existing recovery protocol.

`SessionExecutionInProgress` means the executor is live or abandonment cannot be
established. Show that the conversation is still in progress and retry later;
do not turn this exception into an unconditional recovery call. Liveness uses
the configured lease (60 seconds by default), independently renewed during
silent model/tool calls; a process proved dead on this host ends it early.
Inactivity alone does not establish abandonment. A process that is alive but
cannot renew its lease, for example because it lost the store or blocks its
event loop longer than the lease, loses ownership and is fenced, so keep
blocking work off the event loop.

Sessions created by older writers without an execution owner need explicit
operator recovery. First stop or otherwise prove the old executor has stopped,
then call
`app.recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=...))`
and inspect its result before continuing. Keep the original executable profile
available for recovery; a changed tool/provider identity still requires explicit
profile adoption.

Recovery does not rerun a tool whose call was interrupted, and it cannot show
the model that call's original arguments. For `NONE` and `IDEMPOTENT` tools, the
model receives a failed result saying the outcome is unknown and that calling the
tool again for the same operation is safe, and the conversation continues. An
idempotent tool should therefore recognize a repeated request for an operation it
already started, by business identity rather than exact arguments, and return the
original outcome, for example by reusing the operation key it saved before the
first attempt. An uncertain `EXTERNAL` effect emits
`tool.effect.outcome_unknown` and interrupts for reconciliation. Use the service's receipt or idempotency key to
establish the outcome; never infer that the write failed because its process died.

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
behavior. A scripted test can also pass while the real model sees nothing
useful: a custom tool's result reaches the model only through its `content`,
never its `structured` data. Assert on what the model reads with
`model_facing_text(provider.requests[i])` (one line per text part, peer
content, tool call and tool result `content`) or
`model_facing_tool_result(result)`:

```python
from cayu import model_facing_text

seen = model_facing_text(provider.requests[1])
assert "tool audit_csv: 3 blank rows in data.csv" in seen
```

Keep manifest-backed prompt/tool alignment evidence separate from a
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
- **Operations:** start with `cayu guide durable-operations`; model stable action
  identity, idempotency, ambiguity, approvals, verification, and restart recovery
  before adding autonomous effects.
- **Durable workflows:** keep deterministic orchestration in application code
  and durable state; use Cayu tasks and workflow helpers where needed. Use
  model steps only where judgment is required. Use `WorkflowBase.execute(...,
  execution_deadline=ExecutionDeadline.after(seconds))` for a Runtime finish-by boundary.
  Inspect `await ctx.remaining_seconds()` to reserve application-selected finalization
  time; children and resumed executions retain the original effective expiry. See the
  [execution deadline contract](../../../docs/runtime-contracts.md#execution-deadlines).
- **Multi-agent:** justify each role, bound delegation, persist lineage, and eval
  both child behavior and parent synthesis.

Finish by rerunning inspection, checks, focused tests, the relevant eval, and
any explicitly available process/live checks. Report limitations rather than
substituting weaker evidence.

## Workspace references and inventories

Use `WorkspaceReferenceBinding` for application-owned references, inventories, or
receipts. Obtain it from the actual registered workspace with
`workspace.reference_binding()`, or from the admitted workspace inside a tool
with `ctx.workspace_reference_binding()`. Do not reconstruct an ID from an
environment name, path, or naming convention.

```python
from pydantic import BaseModel
from cayu import WorkspaceReferenceBinding

class Inventory(BaseModel):
    binding: WorkspaceReferenceBinding
    paths: tuple[str, ...]

inventory = Inventory(binding=workspace.reference_binding(), paths=("note.txt",))
# Store alongside your application data, including in durable JSON metadata.
serialized = inventory.model_dump_json()
restored = Inventory.model_validate_json(serialized)

# Inside Tool.run(ctx, args), before using paths or inventory claims:
ctx.require_workspace_binding(restored.binding)
```

`require_workspace_binding` checks the workspace actually admitted for that tool
invocation, after factory materialization and workspace binding. A sync binding
can select a target different from the registered source: create the inventory
from that target or from the active context. Source ownership does not transfer
merely because files were copied. Evaluation wrappers and alternate execution
adapters use the same API on their actual workspace objects.

A mismatch raises `WorkspaceReferenceBindingError` with
`code == "workspace_binding_mismatch"`. A context without a live Runtime-bound
workspace raises `workspace_binding_unavailable`; setting `ctx.workspace_id` or
putting an ID in metadata cannot supply it. Catch the diagnostic and reject or
rebuild the reference before consuming its claims. A deserialized `ToolContext`
cannot validate a reference; validate in a new admitted invocation.

The versioned JSON binding composes `WorkspaceIdentity` (ID and observing adapter)
with an incarnation `generation`. It is distinct from `WorkspaceBinding`, which
connects a workspace and runner, and from file or workspace revision observations.
The default generation is scoped to the workspace object's lifetime. Reopening
application JSON preserves the original binding exactly; it does **not** rebind
it. Reconstructing even the same adapter at the same path with the same ID yields
a mismatch. Rebuild the inventory from the newly selected workspace after
recovery when using the default implementation.

A durable workspace adapter may override `Workspace.reference_binding()` to
return an adapter-owned binding that survives reconnect. It must independently
persist and verify the workspace incarnation, preserve all binding fields for
that incarnation, rotate the generation on replacement/reset, and fail explicitly
when continuity cannot be established. Never recover a generation from the
inventory being checked or derive it from names, paths, or file contents. An
adapter wrapper may delegate to its underlying workspace only when it represents
the same incarnation. Runtime does not infer such equivalence. Built-in adapters
currently use the conservative object-lifetime default.

This is an ownership assertion, not authentication of application data. It does
not establish file existence, content integrity, freshness, read permission, or
verified claims; an inventory binding neither certifies its claims nor grants
access. Continue to use workspace reads, revision checks, policy enforcement, and
application verification for those separate questions. Treat serialized bindings
as application data subject to your normal integrity controls.

Run `python examples/workspace_reference_binding.py` for a synthetic example
requiring no model credentials. It accepts the original inventory and reports
`workspace_binding_mismatch` for a replacement with the same ID, path, and bytes.
