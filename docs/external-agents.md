# External vendor harnesses

Use Cayu agents for applications whose model loop, tools and recovery Cayu should
own. External adapters serve the narrower case of deliberately running a vendor's
whole harness, such as comparing Claude Code and Codex reviews with a Cayu agent.
They are separate from `ModelProvider`, which represents one model step.

Install the optional SDK you use:

```bash
pip install 'cayu[server,claude-agent-sdk,codex]'
```

`cayu.external_agents` exports a vendor-neutral `ExternalAgentAdapter` protocol,
`ExternalAgentEvent`, `ExternalAgentRecorder`, `run_external_agent`,
`ClaudeAgentAdapter`, `CodexAgentAdapter`, the Docker launcher, and the native tool
bridge. SDK imports happen only when an adapter starts. The HTTP gateway lives in
`cayu.external_agents.server` and requires the server extra.

## Sessions, tasks and deadlines

`run_external_agent(app, request, adapter, prompt=..., output_schema=...)` returns
an `ExternalAgentResult` with the session ID, task ID and protected output. The app
must have a task store. The request must have a finite `ExecutionDeadline`; an
ambient earlier deadline wins. Every invocation allocates a fresh session and a
fresh vendor thread. An omitted task ID creates an ordinary task automatically.
For durable queue work, supply the exact task ID and worker lease from
`run_task_worker`:

```python
from cayu import Message, RunRequest
from cayu.deadlines import ExecutionDeadline
from cayu.external_agents import run_external_agent

async def handle(app, task, worker_id):
    prompt = task.input["prompt"]
    result = await run_external_agent(
        app,
        RunRequest(
            agent_name="vendor-review",
            task_id=task.id,
            task_worker_id=worker_id,
            task_lease_expires_at=task.lease_expires_at,
            messages=[Message.text("user", prompt)],
            execution_deadline=ExecutionDeadline.after(600),
        ),
        adapter,
        prompt=prompt,
        output_schema={"type": "object"},
    )
```

External runs enforce the task subject's session access grant before creating the
session and recheck it during execution. A task grant alone cannot authorize a
session, and revocation stops the harness.

The adapter owns and joins its SDK transport and guest process tree before the
run settles. A failed sandbox removal retains the live task and execution presence
while cleanup retries. For an unleased run, the recovery helper can release that
wait only after its supervisor callback proves quiescence. Deadline expiry fails the task; operator interruption stops the
harness and cancels that task while its worker continues accepting other jobs.
Managed worker shutdown remains owned by the worker. Create the
queue task with `TaskCreate(restartable=True)` to requeue it after graceful worker
shutdown: the worker joins adapter and sandbox cleanup, retires the old session,
and the next claim invokes this handler from the beginning. A process crash needs
supervisor evidence that the old process and sandbox are stopped before ordinary
task recovery; lease expiry alone is not that evidence. Queue restart policy is
separate from vendor session resumption; the adapter never adopts an old CLI session. Ordinary retry-series and verified-work execution are not supported.

For an implicitly created, unleased task abandoned by a crashed process,
`recover_external_agent_run(app, session_id, quiesce=..., evidence_id=...)`
records interruption and unknown in-flight usage, then settles that old task.
This includes a crash between session creation and task attachment: recovery
verifies the task and session invocation provenance before settling the task.
The asynchronous `quiesce(session)` callback must stop and join the old service,
sandbox and proxy work, or validate a positive supervisor receipt; it must fail
if this cannot be established. The helper refuses worker-owned tasks. After
recovery, schedule a fresh run. Unknown crash usage is recorded when discovered,
so rolling budgets include it; retries reuse the durable receipt timestamp.
Generic native-session recovery refuses external sessions because it cannot
recover vendor state or infer that a detached container is stopped.

Supported request controls are agent name, messages, metadata, labels, deadline,
budget limits and exact ordinary-task ownership. Native-loop settings such as
`max_steps`, structured-output tools, thinking and environment selection are
rejected; configure the adapter's model, effort, launcher and `output_schema`
instead. External runs cannot resume, fork or capture execution snapshots, and
are not replayable native trajectories.

SDK text, tool observations and final results appear as `custom.external_agent.*`
events. Usage appears as normal `model.completed` events, so session inspection,
usage summaries, the budget store and the dashboard use the usual projections.
Repeated observation IDs are idempotent; conflicting payloads fail. Observations
are limited to 8 MiB each, 64 MiB per run and 100,000 distinct IDs. Secret redaction
handles text split across SDK chunks. An interrupted call records unknown usage
rather than implying that it was free.
SDK tool-call and result observations retain bounded names, IDs, type and status.
Arguments, commands, file changes and result content are published only by the
native permission/tool bridge under the tool's publication contract, including
MCP manifest admission and private-argument controls. Permission-only built-in
tools refuse approval argument edits: the SDK consumes permission for the
original invocation and cannot apply the edited arguments.
Guest stderr is discarded instead of entering service logs. The pinned Codex SDK's
transport rejects frames over 8 MiB and runs exceeding 64 MiB or 100,000 incoming
frames, including notifications received before initialization or outside a turn.
These cumulative limits also bound SDK queues that are never consumed.

Budgets are **after-step checks**. The harness makes its own model calls, so Cayu
cannot reserve their spend. Reservation-based limits are refused before setup.
The gateway records final upstream token usage using the normal price book for
cost estimation. It serializes model calls across all capabilities for the same
run, commits each receipt, and checks the budget before admitting another call.
Direct guest HTTP requests follow the same accounting path. Each response is
buffered up to 64 MiB before delivery so provisional stream counters cannot admit
more work. SDK counters are ignored when the gateway owns accounting.
For custom launchers without this gateway, Claude consumes final per-message
stream usage and Codex converts cumulative SDK counters to deltas; these
observations cannot enforce admission of calls made independently by the guest.
Claude's final USD total enters the existing reported-cost projection,
which retains provider receipts separately from estimated budget authority and
does not add a second charge to the token-derived ledger. Missing or interrupted
upstream receipts remain unknown. A fully received non-success response records
zero usage, so a transient rejection does not prevent a budgeted retry. The
gateway declares the upstream usage dialect and persists normalized counters,
including authoritative zero counts. A single admitted call can exceed the budget;
the gateway refuses further model and native-tool calls after its receipt is
recorded. If receipt publication fails, every capability for that run stops
admitting work; the run cannot complete successfully with unsettled accounting.
Limits with
`action="notify"` report the threshold without interrupting the run.

## Sandbox and credential gateway

The whole CLI must run in a sandbox because its built-in tools run beside it.
The SDK client stays in the service and talks to the CLI over tunneled stdio.
For Codex inside an isolated container, select `sandbox="external"` so its
app-server uses that existing boundary instead of starting a nested bubblewrap
sandbox. Native approval callbacks still apply. Set `workspace_read_only=True`
on the Docker launcher when the guest must not modify the checkout; the mount
is then enforced read-only by Docker. The launcher remains responsible for
filesystem and network containment in external sandbox mode.

`docker_agent_launch` allocates a container, mounts only the prepared checkout,
executes the CLI as the checkout's numeric owner, and removes the whole container
on exit. It verifies that the selected Docker network is `--internal`; host,
bridge and default networks are refused. The application owns the image, network,
and the service endpoint reachable from that network. An internal network blocks
external routing but does not by itself restrict host services or peer containers:
deploy a host firewall allowing only the gateway and disable inter-container
communication. Treat Docker access as privileged service authority.

The image must contain the exact SDK-compatible CLI at the configured absolute
path, plus the tools needed by the workload. Do not mount a host home, Docker
socket, credentials or shared repository mirrors. The launcher's environment is
explicit, and both adapters blank inherited host variables. Guest API keys and
subscription tokens are refused. A custom launcher must provide the same isolation
and process-tree cleanup contract through `ExternalAgentLaunch.close`.

The gateway forwards only configured paths and models, refuses redirects, adds
service-owned authentication headers and issues a short-lived bearer capability
per run. Revocation cancels and joins active model and MCP requests. It also
closes idle upstream streams, and deadlines cover the entire request and stream.
Mount the router only on the private sandbox interface; it is not a general proxy.
It supports stateless Messages and Responses requests with client-executed tools
and inline attachments. Anthropic inline text/PDF citations can be carried into
later requests. Prompt-cache writes support the five-minute lifetime; one-hour
writes are rejected because their distinct price is not yet accounted for.
Provider-hosted tools, remote attachment URLs, stored
response/file references, background execution and unknown request fields are
refused. Guest beta headers are discarded; any required beta header belongs in
the trusted route configuration. Codex hosted web search is disabled. Guest
capability values travel through the launcher's environment, not host command
arguments.

```python
from contextlib import asynccontextmanager
from pathlib import Path
from cayu.external_agents import ClaudeAgentAdapter, docker_agent_launch
from cayu.external_agents.server import ExternalAgentGateway, ExternalModelRoute

# The application owns http_client and mounts gateway.router on this interface.
gateway = ExternalAgentGateway(
    client=http_client,
    public_base_url="http://172.30.0.1:8766",
    routes={"anthropic": ExternalModelRoute(
        base_url="https://api.anthropic.com",
        headers={"x-api-key": service_api_key},
        paths=("/v1/messages",),
    )},
)

@asynccontextmanager
async def launch(context):
    async with gateway.grant(
        context, route="anthropic", models=frozenset({model})
    ) as capability:
        async with docker_agent_launch(
            image="my-agent-image:version",
            cli="/usr/local/bin/claude",
            workspace=Path(prepared_checkout),
            network="agent-internal",
            environment={
                "ANTHROPIC_BASE_URL": capability.model_base_url,
                "ANTHROPIC_AUTH_TOKEN": capability.token,
            },
        ) as prepared:
            yield prepared

adapter = ClaudeAgentAdapter(model=model, launch=launch)
```

For Codex, pass the fixed capability route base URL to `CodexAgentAdapter` and put
its token in `CAYU_MODEL_PROXY_TOKEN`. The adapter configures an OpenAI Responses
provider requiring only that scoped proxy token. `ExternalAgentLaunch` can also
supply per-run `mcp_servers` for Claude and `codex_config_overrides` for Codex.
Use service-side API credentials. Products should not expose subscription login
flows without the provider's required authorization. Consult the current
[Claude Agent SDK documentation](https://platform.claude.com/docs/en/agent-sdk/overview)
and [Codex app-server documentation](https://developers.openai.com/codex/app-server/).

## Native tools and permission decisions

`CayuToolBridge(app, name=..., tools=..., policy=..., environment_name=...)` must be
registered before app startup. Each call enters a native child session using a
deterministic tool dispatcher: there is no model request. Native tool policy,
argument validation, approvals, effects, environments and deadlines still apply.
Child calls inherit the parent's request budget limits. Process-isolated tools
retain the native process supervisor, and MCP discovery validates the registered
manifest policy before exposing descriptions or schemas to the vendor harness.
These control steps have a runtime-owned zero price under every budget book, and
the dispatcher records authoritative zero-token usage with its canonical billing
identity. `external_tool_price_book(book)` can also include that explicit price
in an application-owned book for inspection.

For built-in permissions, register `ExternalPermissionTool("Read",
requested_effect=...)` or the vendor-specific names you offer. Pass an `authorize`
callback to `run_external_agent` that calls `bridge.authorize` with the active
external session and deadline. Policy sees the declared external effect and exact
arguments. The permission tool itself records only an authorization receipt:
Cayu does not execute, roll back or attest completion of a vendor built-in tool.
The Claude adapter gates both `PreToolUse` and `can_use_tool`; Codex gates command
and file approvals and denies unknown requests. Requests containing known secrets,
including the run's gateway capabilities, are refused before native tool records
are created. Without an authorization callback, permission requests are denied.
Vendor-internal tools that never request approval
remain subject to the guest sandbox, not native tool execution guarantees.

For native tools over MCP, bind `external_tool_scope(bridge, session_id=...,
execution_deadline=...)` and pass its endpoint to `gateway.grant(..., tools=...)`.
Configure the vendor MCP client with `capability.mcp_url` and the bearer token.
The endpoint supports initialize, ping, tools/list and tools/call over HTTP JSON-RPC.
An approval can be resolved through Cayu's normal dashboard/API while the live
SDK waits. Revocation joins in-flight calls; later approvals cannot execute a
native tool once the external caller has left or the service has restarted.
