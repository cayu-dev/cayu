# Cayu project diagnostics

`cayu check` renders these stable findings from the public application manifest.
Run the correction, then rerun `cayu inspect --json` and `cayu check --json`.
The last section, [request-cost](#request-cost), covers what a running server
spends on each HTTP route and how to check it.

## app-no-agents

`APP_NO_AGENTS` means the factory returned an app with no registered agent.
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
execution, and include the external tool's actual name in its scope.

## external-tool-coverage-unknown

`EXTERNAL_TOOL_COVERAGE_UNKNOWN` means an external-effect tool uses a custom or
otherwise unrecognized policy whose behavior Cayu cannot verify statically.
This remains an error rather than an acknowledgment-based bypass: use a
statically describable enforcing policy until Cayu provides a trusted custom
coverage contract.

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
may accept caller-selected test identities and an open operator mount, so it is
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
model output, or tool input.

## public-service-operator-access-unsafe

`PUBLIC_SERVICE_OPERATOR_ACCESS_UNSAFE` means the separately mounted `/cayu/`
operator control plane uses `OpenAccess` or a fail-closed placeholder because
production authentication is missing or invalid. Configure
`AuthenticatedAccess` for production. Operator authentication protects the raw
control plane; it does not make that surface customer-facing or tenant-scoped.

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

The same summary is available to operators at `GET /api/diagnostics/requests`
(protected like the rest of the control plane) and can be added to a support
bundle with `cayu doctor --bundle cayu-support.zip --requests-from SERVER_URL`.

CPU time is process CPU shared out across the requests running at the same
time, so work running in the background during a request, such as an agent
run, is charged to it. Measure idle cost while no agent work is running.
Timing adds about 15 microseconds of CPU per request (measured on an Apple M1
Pro with Python 3.14), under 0.1% of one CPU at 50 requests per second.
