# Cayu project diagnostics

`cayu check` renders these stable findings from the public application manifest.
Run the correction, then rerun `cayu inspect --json` and `cayu check --json`.
Two sections cover runtime behavior instead of check findings:
[execution-profile-mismatch](#execution-profile-mismatch) explains the error a
resumed session raises after the agent's code changed, and
[request-cost](#request-cost) covers what a running server spends on each HTTP
route and how to check it.

## scaffold-import-removed

`SCAFFOLD_IMPORT_REMOVED` identifies an outdated `from cayu import ...` or
`from cayu.runtime import ...` statement in a declared project or an explicitly
imported local helper. The finding includes the source line and replacement
import, preserving any local alias. Update every reported import, then rerun
`cayu check --json`. Cayu does not import the application while these findings
remain, including when a tag filter would otherwise hide them.

Installed packages also report replacement imports when a removed name is
accessed directly. These known removed routes raise `ImportError`; unknown
attributes retain `AttributeError`. No removed route loads its former object.

## task-event-retry-policy-undeclared

`TASK_EVENT_RETRY_POLICY_UNDECLARED` is an advisory source finding: a worker
sets `execution_timeout_s`, while a `TaskCreate` for the same literal type has
neither `retry_policy` nor `retry_key`. A task timeout cannot stop a sweeper
from creating a new task for the event. Choose an event policy from the
application's requirements; see `cayu guide long-running#event-retry-policy`.
The check does not infer event identity, require a particular deadline, or
change runtime behavior. Dynamic calls and policies passed through `**kwargs`
are not analyzed. A creation that is never retried needs no event budget.

## app-no-agents

`APP_NO_AGENTS` means the factory returned an app with no registered specialist.
Register an `AgentSpec` with `CayuApp.register_agent()`.

## agent-generated-tracer-bullet-unfinished

`AGENT_GENERATED_TRACER_BULLET_UNFINISHED` means a generator left the agent's
explicit authoring-state marker in place. The generated prompt, placeholder
tool behavior, runtime test, and scripted trajectory are a runnable wiring
proof, not evidence that the requested domain behavior is complete.

Replace the domain system prompt, tool schema and implementation, runtime test
inputs and assertions, and trajectory eval behavior and assertions. For a
scaffold updated by `cayu generate tool`, change `_AUTHORING_STATE` to `None`
inside the generated agent-config region. For `cayu generate slice`, remove
`authoring_state=AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET` and
the unused import from the generated agent module. Verify with
`cayu inspect --json && cayu check --fail-on warning --json`.

The marker is an author assertion, not source analysis: Cayu does not scan
Python or prose for words such as `sample`, `echo`, or `tracer bullet`, and an
absent marker does not prove domain correctness.

## agent-provider-not-found

`AGENT_PROVIDER_NOT_FOUND` means an agent's explicit provider is absent, or no
model-pattern/default provider can resolve the agent. Register the named
provider, correct `provider_name`, or define an unambiguous default route.

## agent-provider-ambiguous

`AGENT_PROVIDER_AMBIGUOUS` means more than one registered provider pattern
matches the agent model. Make patterns disjoint or set the agent's
`provider_name` explicitly.

## agent-environment-not-found

`AGENT_ENVIRONMENT_NOT_FOUND` means an agent was registered with
`register_agent(..., environment_name="NAME")`, but no environment named `NAME`
is registered, so a new run of that agent fails before it starts. Register the
environment, correct the name, or pass `NO_ENVIRONMENT` to run the agent
without one.

## agent-workflow-tool-not-registered

`AGENT_WORKFLOW_TOOL_NOT_REGISTERED` means an agent's explicit
`workflow_tool_names` contract names a tool that is not registered for that
same agent. Use the exact registered name, update the machine-owned tool-name
source after a rename, or register the intended tool. Cayu checks this explicit
contract and does not parse arbitrary natural-language prompt text.

## agent-workflow-workspace-not-registered

`AGENT_WORKFLOW_WORKSPACE_NOT_REGISTERED` means a registered file tool named in
`workflow_tool_names` has no structurally available workspace. Register a
static environment with a workspace, or an environment factory that supplies
one per session.

## agent-workflow-runner-not-registered

`AGENT_WORKFLOW_RUNNER_NOT_REGISTERED` means `exec_command`, `run_check`, or
another runner-backed workflow tool is registered and named in
`workflow_tool_names`, but no runner is structurally available.
Register a static environment with a runner, or an environment factory that
supplies one per session.

## agent-workflow-command-policy-not-registered

`AGENT_WORKFLOW_COMMAND_POLICY_NOT_REGISTERED` means an agent explicitly
declares `exec_command` or `run_check` as part of its workflow but the
registered command tool has no `CommandPolicy`. `RunCheckTool` rejects that
configuration during construction; attach a deny-by-default policy such as
`ProcessCommandPolicy`. Inspection reports only the policy type, never its
allowed executables, directories, environment values, or other policy data.

## external-tool-unguarded

`EXTERNAL_TOOL_UNGUARDED` means a tool declaring `ToolEffect.EXTERNAL` is under
a policy that can allow that specific tool without an enforcing boundary. The
diagnostic reports the effective per-tool coverage rather than trusting the
policy class name. Register an enforcing policy. Use
`AlwaysRequireApprovalToolPolicy(tools=[...])` when a human must authorize
execution, and include the external tool's actual name in its scope. Naming the
tool in an explicit allowlist (`StaticToolPolicy(allow=[...])`) is a deliberate
authorization and satisfies this check. The built-in `ask_user` tool always
pauses for the user before anything happens, so it is never reported.

## external-tool-coverage-unknown

`EXTERNAL_TOOL_COVERAGE_UNKNOWN` means an external-effect tool uses a custom or
otherwise unrecognized policy whose behavior Cayu cannot verify statically.
This remains an error rather than an acknowledgment-based bypass: use a
statically describable enforcing policy. To add application validation, compose
`GuardedToolPolicy(guards=(app_guard,), then=AlwaysRequireApprovalToolPolicy(...))`.
Each guard returns `None` or a denial; invalid results and exceptions deny, so a
guard cannot bypass the downstream enforcement. Cayu derives coverage from the
exact downstream built-ins without calling guards or probing application stores.
Overriding policy subclasses and unknown downstream policies remain unknown.
Declare a behavior identity on custom guards for portable recovery; coverage is
not a self-attestation and does not prove the guard's application logic.

## tool-approval-validity-rule

`TOOL_APPROVAL_VALIDITY_RULE` is informational. It names each required-field or
required-allowlist rule configured under `REQUIRE_APPROVAL`. Missing, empty or
wrongly typed arguments are always denied, including when an earlier rule would
request approval. Approval applies to authority violations of well-formed calls.
Correct the arguments and submit a new call; an approver cannot supply them.
Inspect these rules with `cayu check --json` and `cayu inspect --json`.

## tool-input-schema-unconstrained

`TOOL_INPUT_SCHEMA_UNCONSTRAINED` means a registered tool exposes `{}` as its
input schema. That is valid JSON Schema, but it accepts every JSON value and
does not teach the model which arguments to send. Declare the expected object
properties, required fields, and `additionalProperties` behavior in
`ToolSpec.input_schema`. If a tool derives its schema dynamically, override the
public `Tool.schema` property; Cayu treats that property as authoritative when
the tool is registered.

## tool-terminal-payload-unbounded

`TOOL_TERMINAL_PAYLOAD_UNBOUNDED` means an application's own tool declares no
`ToolSpec.max_terminal_payload_bytes`. Cayu reserves the declared size of each
deferred tool round from terminal publication capacity shared by every session
in the process (`ToolExecutionConfig.tool_terminal_staged_capacity_bytes`,
512 MiB by default). Each call to a tool without a declaration reserves the
largest terminal result Cayu can store, 1,984 KiB, so fewer rounds fit at once
and, under heavy concurrent load, rounds wait for capacity and can reach the
admission deadline (`ToolExecutionConfig.tool_round_admission_timeout_seconds`,
600 seconds by default), which fails them with
`ToolRoundAdmissionTimeoutError`. Set `max_terminal_payload_bytes` to the
largest result the tool returns, at most 1,984 KiB; a smaller bound than 64 KiB
reserves 64 KiB. A result larger than the declaration fails as
`invalid_tool_output` instead of being published, so keep the tool's own output
caps inside it. Cayu also logs a warning when the tool is registered. Built-in
tools and MCP tools are not reported.

## tool-knowledge-store-unavailable

`TOOL_KNOWLEDGE_STORE_UNAVAILABLE` means an agent has `list_knowledge`,
`search_knowledge`, `read_knowledge` or `remember_knowledge`, but the
environment it runs in by default has no knowledge store, so every call fails
with `missing_knowledge_store`. A tool call gets its knowledge store and access
scope from its environment, not from `CayuApp(knowledge_store=...)`. Pass
`knowledge_store=` (and `knowledge_access_scope=` when the store has no bound
scope) to the `Environment` or the factory: `DockerCodingEnvironmentFactory`,
`BrokeredSandboxCodingEnvironmentFactory` and `CodingBackend.factory_for` take
both, and so does `register_coding_environment`, where
`inherit_app_knowledge=True` uses the app's store and exactly its scope. The check
covers concrete environments and factories that report
`configured_knowledge_store`; an agent registered with `NO_ENVIRONMENT`, or in
an app with no environment, is reported too. Cayu logs the same warning when
the agent is registered and its environment is already registered.

## tool-input-schema-runtime-unsupported

`TOOL_INPUT_SCHEMA_RUNTIME_UNSUPPORTED` means the registered schema is not a
locally resolvable Draft 2020-12 schema, or it declares a different `$schema`
dialect. Runtime keeps the prior tool-owned argument validation instead of
denying every call. Update the schema to Draft 2020-12 and inline external
references to enable runtime validation. No remote schema URLs are fetched. Run
`cayu check --json` to verify the repair.

## public-service-development-mode

`PUBLIC_SERVICE_DEVELOPMENT_MODE` means the maintained public-service factory
was assembled with its explicit local-development profile. Development access
may accept caller-selected test identities and an open Developer dashboard mount, so it is
restricted to `cayu serve --dev` on a loopback listener. Build the same service
factory with `mode="production"` before deployment.

## evals-project-identity-not-configured

`EVALS_PROJECT_IDENTITY_NOT_CONFIGURED` means automatic Evals assembly cannot
derive a stable project identity. Add a valid PyPA distribution name under
`[project]` in `pyproject.toml`; Cayu normalizes runs of `.`, `_`, and `-` to a
single lowercase `-`. This warning does not disable an explicitly supplied
`EvalsConfig`.

Verify with `cayu check --json`.

## evals-project-store-not-configured

`EVALS_PROJECT_STORE_NOT_CONFIGURED` means production project serving cannot
select durable Evals storage. Configure SQLite or PostgreSQL under
`[tool.cayu.session_store]`, or set `CAYU_DATABASE_URL`. Production does not
create a database merely to clear the warning. Explicit trusted-local
`cayu serve --dev` may use the project-local `data/cayu.db` default.

Connection strings and paths are excluded from the finding. Verify with
`cayu check --json` in the intended deployment environment.

## evals-service-factory-context-migration-required

`EVALS_SERVICE_FACTORY_CONTEXT_MIGRATION_REQUIRED` means a maintained service
factory still uses the older signature and therefore cannot carry Cayu's
framework-owned Evals project context to `create_agent_service(...)`. The
service remains compatible and starts normally; only automatic project
assembly is unavailable through that factory.

Run `cayu generate service-context --dry-run`, review the proposed edit, then
run `cayu generate service-context`. The command modifies only the recognized
previous generated form. If it reports `manual_action_required`, add an
optional keyword-only `project_context: ProjectControlPlaneContext | None =
None` parameter and pass it unchanged as `project_context=project_context` to
`create_agent_service(...)`. Verify with `cayu check --json`.

## public-service-product-access-unsafe

`PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE` means the maintained product API uses a
development or fail-closed placeholder access adapter rather than configured
production authentication. Configure `AuthenticatedProductAccess` so its
server-side dependency returns a trusted `ProductPrincipal`. Tenant identity
must not come from the product request body, query, Cayu labels or metadata,
model output, or tool input. For OIDC/JWT bearer tokens,
`OidcBearerAuth(...).product_dependency(tenant_claim=...)` takes both from the
verified token and rejects a token without the tenant claim.

## public-service-operator-access-unsafe

`PUBLIC_SERVICE_OPERATOR_ACCESS_UNSAFE` means the separately mounted `/cayu/`
Developer dashboard uses `OpenAccess` or a fail-closed placeholder because
production authentication is missing or invalid. Configure
`AuthenticatedAccess` for production. Dashboard authentication protects the raw
Developer dashboard; it does not make that surface customer-facing or tenant-scoped.

## public-service-identity-store-not-durable

`PUBLIC_SERVICE_IDENTITY_STORE_NOT_DURABLE` means the public-to-private identity
mapping declares development-only process state. Use durable application-owned
storage that atomically reserves idempotency identities and performs
tenant-qualified resource lookup before any Cayu operation. Cayu session IDs,
task IDs, labels, and metadata are not product authorization state.

## public-service-session-store-not-durable

`PUBLIC_SERVICE_SESSION_STORE_NOT_DURABLE` means the maintained service uses a
development-only, read-only, or unverified Cayu session store. Public-service
sessions must survive process restarts and accept runtime writes, so configure
a built-in durable `SessionStore` or a custom store that explicitly declares
`service_durability = RuntimeStoreDurability.DURABLE` after its durability
contract is verified.

## public-service-task-store-required

`PUBLIC_SERVICE_TASK_STORE_REQUIRED` means the maintained public service's
`CayuApp` has no task store, so it cannot bind and run the private task identity
reserved by the application-owned product mapping. Configure a durable
`TaskStore` before deployment.

## public-service-task-store-not-durable

`PUBLIC_SERVICE_TASK_STORE_NOT_DURABLE` means a task store is configured but it
is development-only or has not declared verified durability. Configure a
built-in durable `TaskStore` or a custom store that explicitly declares
`service_durability = RuntimeStoreDurability.DURABLE` after its durability
contract is verified.

These findings apply only to the inspectable service returned by Cayu's
maintained factory contract. The check intentionally reports host-owned routes
outside that contract as unverified; it does not scan arbitrary ASGI source or
claim to prove its authorization behavior. Run the generated assembled-app
suite with `pytest -q tests/test_public_service_security.py` in addition to the
deployment check.

Tool implementations must also declare `run` with `async def`. Cayu validates
that contract during agent registration so a synchronous implementation fails
before a session starts.

Inspection and checks are structural. Clearing all diagnostics does not prove a
provider credential, remote service, sandbox, network path, or deployment is
live. The manifest reports `has_system_prompt` but never prompt text; its
fingerprint records prompt presence, not prompt contents. A prompt edit between
two non-empty values therefore needs a runtime test or eval for verification.

## execution-profile-mismatch

`ExecutionProfileMismatchError` is a runtime error, not a `cayu check` finding.
A session records an execution profile on its first run: a digest and an
identity kind for each class of component (tools, policies, provider, model and
so on). Every later resume or continuation rebuilds the profile from the
current code and compares the two. When they differ, Cayu refuses before any
model call, tool call or other governed work. This happens most often while
you edit code and rerun against the same local store.

### Reading the message

The first line lists the changed component classes. The `What changed` lines
show, for each class, how the stored and the current profile identified it:

- `declared`: the components declare an `ExecutionProfileBehaviorIdentity`.
- `structural`: Cayu identifies them from their configuration, as it does for
  built-in providers, built-in policies and tool descriptors.
- `opaque`: at least one component has no declared identity, so Cayu can only
  identify it within one process and one `CayuApp` instance. A request loop
  policy without a declared identity is identified per policy object.
- `unavailable`: the identity could not be resolved.

| Stored → current | What it usually means |
| --- | --- |
| `opaque` → `opaque` | Nothing may have changed. An undeclared component never matches in a new process or `CayuApp` instance. Declare an identity and start a new session. |
| `declared` or `structural` → `opaque` | A component without a declared identity was added, or a declaration was removed. |
| `opaque` → anything else | The stored baseline can't be matched by any later process. Declaring an identity now doesn't repair it; start a new session or adopt. |
| `declared` → `declared` | A real change: a declared name or version changed, or a component was added, removed or reconfigured. |
| `structural` → `structural` | A real change in configuration, such as a tool's description or schema, a budget limit, or the model. |

Stored profiles keep one digest per class, so the message can't name the
individual tool, policy or version that changed. `error.differences` carries
the same evidence as data: `component_class`, `category`, `expected_strength`
and `candidate_strength`. `error.boundary` is `"resume"` for an ordinary resume
with nothing pending, `"continuation"` when pending work had to continue under
its recorded profile, or `None` when unknown.

To see the same comparison before resuming, run
`cayu execution-profile compare SESSION_ID` in the project. It builds the
application, compares the stored profile with the current one class by class,
prints the same explanation, and exits 3 when the session would not resume.
It builds the current profile the way Runtime does at the session's next
boundary. For pending work it restores the run settings recorded with the
approval, user input or tool round, such as `thinking` and budget limits. At a
resume boundary, resume inherits `max_steps`, `limits`, `retry_policy` and
`structured_output` from verified admitted settings, including when the first
provider request failed. The comparison uses those same settings. A finished
interaction does not inherit request `thinking` or `budget_limits`; those classes
are listed under "not compared" with the reason instead of as a change. An
ordinary-turn `tool_completion` policy must still be supplied explicitly. Request `loop_policies` are
never stored, so `invocation_policies` is not compared when the session used
them. The command runs in a new process, so classes with opaque identity
always show as changed there. That includes request `loop_policies` without a
declared identity: no later call can match them, so the command predicts the
rejection instead of listing them as not compared. If the session's agent,
provider or environment is no longer registered, for example after a rename,
the command fails with `REGISTRATION_MISSING` and the registry error the
continuation would raise; register it under the recorded name again.

### Fix the cause

Tools. Give every custom tool a declared identity in its `ToolSpec`, as
`cayu generate tool` does, and bump `behavior_version` when its behavior
changes:

```python
from cayu.runtime import ExecutionProfileBehaviorIdentity

lookup_identity = ExecutionProfileBehaviorIdentity(
    name="lookup", behavior_version="1", implementation_version="1"
)
```

Adding, removing or redescribing a tool changes the session's tool authority,
so a changed tool set needs a new session or adoption.

Policies and hooks. A custom `LoopPolicy`, `ToolPolicy`, `CommandPolicy` or
`RuntimeHook` without a declared identity is opaque. Override its
`execution_profile_identity` property. Built-ins such as `RequireFinalTool`
already declare one. Request `loop_policies` on `RunRequest` and
`ResumeRequest` are part of the session's profile, so pass the same policies on
every run and resume of the session, including in each simulated process of a
test. A request loop policy used together with profile adoption must also
return a stable `adoption_replay_identity`:

```python
from cayu.runtime import ExecutionProfileBehaviorIdentity, LoopPolicy

class CompletionCheck(LoopPolicy):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="completion-check", behavior_version="1", implementation_version="1"
        )

    @property
    def adoption_replay_identity(self) -> str:
        return "completion-check:1:1"
```

Provider. Built-in providers and a `ScriptedModelProvider` built from
positional event batches keep the same identity in every process.
`ScriptedModelProvider(response_factory=...)`, provider subclasses and custom
`ModelProvider` classes are opaque unless they override
`execution_profile_identity`. In a test that simulates several processes with
several `CayuApp` instances on one store, build the provider the same way in
each one:

```python
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent

def scripted_provider() -> ScriptedModelProvider:
    return ScriptedModelProvider(
        [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]]
    )
```

Provider options, thinking and failover settings are part of the profile too
(`provider_request_policy`).

Model. `provider_target` means the agent now resolves to a different provider
or model than the session uses. Register the same provider name and model, or
switch explicitly with `ResumeRequest(..., target=ModelTarget(provider_name=...,
model=...))`, which Cayu adopts at a clean resume boundary.

Cayu itself. `runtime` means the Cayu version or build changed. When it is the
only change and no `ExecutionProfilePolicy` is configured, resume adopts it
automatically at a clean boundary.

Other components. Custom environments, runners, environment factories and
context policies declare `execution_profile_identity` the same way. Structural
settings such as budget limits, structured output, `max_steps`, run limits and
egress must match the stored session.

### Next steps

Start a new session when the change is real or the baseline is opaque:
`app.run(RunRequest(agent_name=..., session_id="<new id>", messages=[...]))`.

To keep the session instead, adopt the current profile at a clean resume
boundary, meaning the previous interaction finished and nothing is pending.
Adoption needs two things: an `ExecutionProfilePolicy` on the `CayuApp` that
returns `adopt` for explicit requests, and a `ResumeRequest` that carries
`profile_adoption`. Without a configured policy, every adoption request is
rejected with `ExecutionProfileAdoptionRejected`. This policy adopts any
explicit request, which suits local development; in production, check
`request.intent.requested_by` and `request.changed_component_classes` before
adopting:

```python
from cayu import CayuApp, Message, ResumeRequest
from cayu.approvals import ResolutionActor, ResolutionActorSource
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
)

class AdoptExplicitRequests(ExecutionProfilePolicy):
    @property
    def identity(self) -> str:
        return "my-app:adopt-explicit-requests:v1"

    async def decide(self, request: ExecutionProfilePolicyRequest) -> ExecutionProfilePolicyResult:
        if request.intent is None:
            return ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.REJECT,
                reason="The profile changed and no adoption was requested.",
            )
        return ExecutionProfilePolicyResult(
            action=ExecutionProfilePolicyAction.ADOPT,
            reason=request.intent.reason,
            authority_decision=(
                ExecutionProfileAuthorityDecision.AUTHORIZED
                if request.authority_review_required
                else ExecutionProfileAuthorityDecision.NOT_REQUIRED
            ),
        )

async def adopt_current_profile(app: CayuApp, session_id: str, key: str) -> None:
    async for event in app.resume(ResumeRequest(
        session_id=session_id,
        messages=[Message.text("user", "Continue with the updated agent.")],
        profile_adoption=ExecutionProfileAdoptionIntent(
            idempotency_key=key,
            reason="Local development: adopt the edited agent code.",
            requested_by=ResolutionActor(subject="developer", source=ResolutionActorSource.REQUEST),
        ),
    )):
        print(event.type)
```

Pass the policy as `CayuApp(..., execution_profile_policy=AdoptExplicitRequests())`
and call `adopt_current_profile(app, session_id, "adopt-1")`. Use a new
idempotency key for each adoption; reusing a key with a different request
fails. Adoption records a durable `session.execution_profile.decided` event
with your reason and actor. If the adopted profile is still opaque, the next
process or `CayuApp` instance mismatches again, so declare identities first.

When `error.boundary` is `"continuation"`, the session has pending work, such
as an approval, user input, a tool round or an unfinished invocation, and that
work must continue under the profile it started with. Restore the previous code
or identities and finish the pending work, then adopt on a later resume, or
start a new session. Cayu never adopts a changed profile on its own, apart from
the runtime-only and explicit `ModelTarget` cases above.

A pending tool approval is the exception: it can adopt the current profile as
it is settled, so an upgrade doesn't strand it. When only Cayu changed and no
`ExecutionProfilePolicy` is configured, approving or denying adopts the new
runtime on its own, as a resume does. For any other change, pass
`profile_adoption` on the `ToolApprovalRequest` (or in the body of
`POST /api/inbox/{id}/settle` or `POST /api/tool-approvals/resolve`), and the
policy decides as it does for a resume. Before the approval is claimed, Cayu
checks each paused call again against the current tool and `ToolPolicy`, with
the paused round's own run context: a call that was allowed must still be
allowed, and the approved call must be allowed or need exactly the approval it
got. Anything else raises `ToolApprovalProfileAdoptionRefused`; the approval
stays pending and no tool runs, so deny it and let the agent propose the call
again. The inbox reports a settle that needs adoption as a 409 with code
`profile_changed`.

One change can't be adopted at all: adding, removing or redescribing one of the
agent's tools. The paused round froze the tools the model saw, and approving
continues that round, so approving with `profile_adoption` raises
`ToolApprovalProfileAdoptionRefused` before any policy runs (without it, the
ordinary mismatch above is raised). Approve such calls with the application
version that proposed them before you deploy the new tool set.

A denial always settles. When the round can't continue under the current
profile, whether the change can't be adopted or no policy adopts it, denying
ends the session instead of continuing it. Nothing runs: Cayu records
`tool.call.approval_denied` with the person and reason, removes the round from
resumable work, and fails the session with `error_type: "SessionAbandoned"` and
`reason: "tool_approval_denied_after_profile_change"`. The session can't be
resumed; fork it to continue its conversation. Repeating the same denial replays
those events. This needs a session with no environment binding and no task of
its own (a `propose_tool_call` proposal, for example); for any other session the
mismatch is still raised.

```python
from cayu.approvals import ToolApprovalRequest

async def approve_across_upgrade(app: CayuApp, approval: ToolApprovalRequest, key: str) -> None:
    adopting = approval.model_copy(update={"profile_adoption": ExecutionProfileAdoptionIntent(
        idempotency_key=key,
        reason="Deployed the new bug-filing tool.",
        requested_by=ResolutionActor(subject="priya", source=ResolutionActorSource.REQUEST),
    )})
    async for event in app.resolve_tool_approval(adopting):
        print(event.type)
```

## request-cost

A running Cayu server records the cost of every request: method, route
template, status, wall time, CPU time, and response size, keyed by a hashed
client. It is on by default under `cayu serve` and for local development
(`ServerConfig.local_development()`, or `mount_cayu` with `OpenAccess`). With
`mount_cayu`, pass `observe_host_requests=True` to also time the host app's
own routes, such as the `/api/...` routes your UI polls. Records stay in memory
only and never include raw paths, IDs, query values, cookies, or credentials.

A slow request is cheap to miss locally: on a laptop with a handful of records,
a 300 ms handler polled every 2 seconds looks fine, but in production it can
use a third of a 0.5 vCPU web process for one idle tab and grow with every job
stored. The server logs these warnings, once per condition:

- `cayu.server.slow_request`: a GET route's median time went over 250 ms.
- `cayu.server.growing_cost`: a GET route keeps getting slower as it is called,
  which usually means it reads all stored history on each call.
- `cayu.server.hot_poll`: one client called the same GET more than 12 times a
  minute for over 2 minutes and mostly got the same response back. See
  `cayu guide app-ui` for polling alternatives.

### Leave it idle, measure, fix

Check the cost of an open page before calling UI work done:

1. Start the app (`cayu serve --dev`, or your own server with `mount_cayu`).
2. Open the UI in a browser and leave it idle for 60 seconds.
3. Measure the idle cost against a budget, for example 1% of a 0.5 vCPU web
   process:

   ```bash
   cayu diagnostics requests --since 60s --budget-idle-cpu 0.01 --vcpu 0.5
   ```

   Add `--server-url http://127.0.0.1:8000/cayu` when Cayu is mounted at
   `/cayu` by `mount_cayu`. The command prints requests per minute, p50/p95
   wall time, CPU seconds per minute, and the share of `--vcpu` for each route,
   highest CPU first. It exits 1 when the budget is exceeded and 2 when it
   cannot read the summary. Use `--json` for machine-readable output.
4. Fix the top route and repeat until the command exits 0. Typical fixes: poll
   less often or stop polling a hidden tab, return 304 Not Modified when nothing
   changed, push changes over a stream, and make GET handlers read only recent
   or changed records instead of rebuilding everything per request.

The same summary is available to dashboard users at `GET /api/diagnostics/requests`
(protected like the rest of the Developer dashboard) and can be added to a support
bundle with `cayu doctor --bundle cayu-support.zip --requests-from SERVER_URL`.

CPU time is process CPU shared out across the requests running at the same
time, so work running in the background during a request, such as an agent
run, is charged to it. Measure idle cost while no agent work is running.
Timing adds about 15 microseconds of CPU per request (measured on an Apple M1
Pro with Python 3.14), under 0.1% of one CPU at 50 requests per second.

## EVAL_EXECUTION_IDENTITY_PROCESS_LOCAL

A registered tool or model provider has no stable execution identity. Its eval
baselines can compare within the same agent instance, but cannot prove unchanged
behavior after a restart or deploy. The warning names each affected component.

For tools, set `ToolSpec(execution_profile_identity=ExecutionProfileBehaviorIdentity(...))`.
For providers, return that declaration from the provider's
`execution_profile_identity` property. Import `ExecutionProfileBehaviorIdentity`
from `cayu.runtime`; supply a stable `name`, `behavior_version`, and
`implementation_version`. Keep versions stable for identical implementations;
update them when their respective contracts change. Then run and approve a new
baseline. Existing undeclared baselines are not retroactively repaired, and
other compatibility checks still apply.

Verify with `cayu check --json`. The warning does not block the default check;
`--fail-on warning` treats it as a failure. Source import-safety checks still run
before the agent is loaded. Names come from booted registration evidence;
loading a serialized manifest alone cannot establish this diagnostic.
