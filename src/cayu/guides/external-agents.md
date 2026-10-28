# External agents: Claude Code and Codex as Cayu sessions

Use this guide when you want to run a vendor's whole coding agent, such as
Claude Code or Codex, and still see each run in Cayu. Every run becomes a Cayu
session and task. Its token usage and estimated cost are recorded like any other
session, it stops at a deadline, a USD cap is checked after each model call, and
known secrets are scrubbed from the events and transcript Cayu stores.

If you want Cayu to run the model loop, tools and recovery itself, build a normal
Cayu agent instead (`cayu guide authoring`). These adapters are separate from
`ModelProvider`, which is one model step.

## What you need

Required:

- The vendor SDK, `cayu[claude-agent-sdk]` or `cayu[codex]`, and the vendor's CLI.
  `cayu[claude-agent-sdk]` accepts `claude-agent-sdk>=0.2.168,<0.3`, and Cayu's
  package CI installs the newest version in that range and runs the adapter
  against it. `cayu[codex]` pins `openai-codex==0.163.0` exactly, because the
  Codex adapter's bounded transport depends on that SDK's internals. To hold
  `claude-agent-sdk` at one version, pin it in your own requirements, inside the
  range.
- A `CayuApp` with a task store. Use SQLite or Postgres stores to inspect runs
  after the process exits.
- A finite `execution_deadline` on the `RunRequest`.
- A launcher. It starts the CLI for one run and, when the run ends, stops it and
  everything it started. Cayu ships four, and you can [write your
  own](#write-your-own-launcher):

| Launcher | Where the CLI runs | Needs |
| --- | --- | --- |
| `local_agent_launch` | On this host, as the current user. Not a sandbox: use it for trusted code and tests. | Nothing else |
| `linux_agent_launch` | On this Linux host, with Landlock and seccomp confinement | Landlock ABI 6 and seccomp; no Docker, root or namespaces |
| `docker_agent_launch` | In a container on an internal Docker network | Docker, an image with the CLI, an internal network |
| `runner_agent_launch` | In a runner that can open interactive processes, such as `LambdaMicroVMRunner` | That runner |

Choose the launcher for the host you actually have. `local_agent_launch` does
not confine untrusted code. If Docker and a suitable remote runner are unavailable,
use an existing sandbox through a [custom `ExternalAgentLaunch`](#write-your-own-launcher).
`LaunchedRunner` can apply your launcher to runner commands, but it only offers
interactive processes when the wrapped runner does; wrapping `LocalRunner` does
not add that capability. See `cayu guide references#environments`.

Optional:

- The credential gateway, `ExternalAgentGateway` in `cayu.external_agents.server`
  (needs `cayu[server]`). The CLI sends its model calls to the gateway, which adds
  the service's API key, records each call and refuses calls once a USD cap is
  reached.
- A sandbox. The launcher alone decides where the CLI and its built-in tools run.

Without the gateway:

- The CLI uses its own model credentials: its existing login, or an API key you
  pass in the launcher's `environment`. In a sandbox, the CLI and the tools it
  runs can read that key.
- Cayu records the usage the SDK reports. It checks USD caps after each reported
  model call and before each tool approval, but it can't refuse a model call
  before the CLI makes it. A call the SDK never reports, for example after a
  crash, is recorded as unknown usage.
- The CLI can reach whatever network its launcher allows.
- Native Cayu tools over MCP (`external_tool_scope`) need the gateway, which
  serves their MCP endpoint. Approving the CLI's built-in tools through Cayu
  policies (`CayuToolBridge.authorize`) works without it.

With or without the gateway, every run gets a Cayu session and task, recorded
events and usage, the deadline, tool authorization through `authorize`, and secret
scrubbing in what Cayu stores.

## Linux confinement without Docker

`linux_agent_launch` runs the CLI on a Linux x86_64 or aarch64 host using
Landlock and seccomp. It needs neither a Docker daemon nor user namespaces.
It grants write access to the checkout and a fresh private HOME/TMPDIR, read
access to system binaries/libraries and the CLI, and no network access by
default. Add installation directories with `read_only_paths` when the CLI
loads files next to its executable. A script whose interpreter is installed
outside system directories also needs that interpreter installation granted
(for example a Python installation under /opt). Host home directories and credentials are
not inherited. This is host-kernel confinement, not a VM boundary.

```python
from contextlib import asynccontextmanager
from cayu.external_agents import linux_agent_launch


@asynccontextmanager
async def confined_launch(context):
    async with linux_agent_launch(
        cli="/opt/vendor/bin/claude", cwd="/srv/review/checkout",
        read_only_paths=["/opt/vendor"],
    ) as prepared:
        yield prepared
```

Use this launcher with the same adapters and `run_external_agent` as any
other launcher. Stdio stays interactive, so approvals and usage observations
continue to work. Descendants inherit confinement and cannot escape cleanup
with `setsid`, signal the supervisor, or change another process's resource
limits or scheduling. TCP Fast Open is refused, so granted ports are the
only TCP destinations. Closing confirms no process in the
owned group can still execute, including when container PID 1 retains zombies.
File mode, ownership, timestamp and extended-attribute changes are blocked,
even inside writable grants. Filesystem ioctls are blocked too; only terminal
queries and nonblocking/close-on-exec descriptor controls are allowed.
Missing kernel support refuses the launch; it never falls back to an
unconfined local process. The launcher adds no CPU, memory, attempt or execution
time quotas; choose run deadlines and resource policy for the application.

For model access, keep real provider credentials in `ExternalAgentGateway`
and pass only its per-run token in `environment`. Grant its TCP port with
`allowed_tcp_ports=[port]` and use an IP literal such as `127.0.0.1` in the
gateway URL. **A port grant permits that destination port on any address**;
Landlock does not provide an IP/hostname allowlist. UDP/DNS and external UNIX
sockets are blocked. Use a host firewall or a network-isolated runner when
the application's contract requires host-based egress. Do not grant broad
filesystem paths containing host secrets or writable launcher/interpreter files.

## Quick start without Docker or a gateway

This example runs the `claude` CLI on your machine with your Claude Code login and
records the run in SQLite. `local_agent_launch` is not a sandbox. Claude Code and
the tools it runs can read and change anything your user can, and reach any
network. Use it for trusted code, local development and tests. For untrusted
repositories or production, use a sandboxed launcher, as in the
[production setup](#production-setup-docker-and-the-gateway).

```bash
pip install 'cayu[claude-agent-sdk]'
```

```python
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from cayu import CayuApp, Message, RunRequest
from cayu.deadlines import ExecutionDeadline
from cayu.external_agents import ClaudeAgentAdapter, local_agent_launch, run_external_agent
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore

CHECKOUT = Path.cwd()  # The directory Claude Code starts in.


@asynccontextmanager
async def launch(context):
    # Runs the `claude` on PATH as you, with your login. Not a sandbox.
    async with local_agent_launch(cli="claude", cwd=CHECKOUT) as prepared:
        yield prepared


async def allow_reads(name, arguments, call_id):
    return name in {"Read", "Grep", "Glob"}  # Without a callback every tool call is denied.


async def main():
    app = CayuApp(
        session_store=SQLiteSessionStore("sessions.db"),
        task_store=SQLiteTaskStore("tasks.db"),
    )
    prompt = "Review the code in the current directory and list the three riskiest problems."
    try:
        result = await run_external_agent(
            app,
            RunRequest(
                agent_name="claude-local",
                messages=[Message.text("user", prompt)],
                execution_deadline=ExecutionDeadline.after(600),  # Required. Ten minutes.
            ),
            ClaudeAgentAdapter(model="claude-sonnet-5-5", launch=launch),
            prompt=prompt,
            authorize=allow_reads,
        )
        usage = await app.get_session_usage(result.session_id)
        print(result.output)
        print(
            f"session {result.session_id}: {usage.usage.input_tokens} input and "
            f"{usage.usage.output_tokens} output tokens"
        )
    finally:
        await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

`cayu session usage SESSION_ID --sqlite sessions.db` shows the run's model calls
after the script exits. Use a `claude` CLI version that the installed
`claude-agent-sdk` supports. `sessions.db` keeps every run until something removes
it; for a long-lived service, schedule storage retention with
`run_storage_retention_worker` or `cayu storage prune --apply`
([storage retention](https://github.com/cayu-dev/cayu/blob/main/docs/storage-retention.md)).

`local_agent_launch` gives the CLI `PATH`, `HOME` and the locale settings of this
process, so a CLI that is already logged in keeps using its login. No other host
variable reaches it. Pass more with `environment=`, for example
`{"ANTHROPIC_API_KEY": key}`. Values of variables whose names contain `KEY`,
`TOKEN` or `AUTHORIZATION` are scrubbed from what Cayu stores. When the run ends,
the launcher kills the CLI's process group and waits until it is empty. A process
that starts its own session leaves that group and is not stopped.

For Codex, run the `codex` CLI and leave out `base_url`, so Codex uses its own
login and provider settings:

```python
from cayu.external_agents import CodexAgentAdapter


@asynccontextmanager
async def launch_codex(context):
    async with local_agent_launch(cli="codex", cwd=CHECKOUT) as prepared:
        yield prepared


adapter = CodexAgentAdapter(model="gpt-6-sol", launch=launch_codex)
```

Codex asks `authorize` about `codex_command` and `codex_file_change` requests. Its
default `sandbox="read-only"` keeps Codex's own sandbox for the commands it runs,
where the host supports one.

## Production setup: Docker and the gateway

Install Cayu with the server extra and the vendor SDK you use:

```bash
pip install 'cayu[server,claude-agent-sdk]'   # Claude Code
pip install 'cayu[server,codex]'              # Codex
```

The example below runs Claude Code in a container that can reach only the Cayu
gateway. Before running it you need:

- Docker. `docker_agent_launch` starts the CLI in a container (see
  [Launchers](#launchers)). The example assumes a Linux host.
- An internal Docker network, created with
  `docker network create --internal --subnet 172.30.0.0/24 --gateway 172.30.0.1 agent-internal`.
  On Linux the network's gateway address, `172.30.0.1`, belongs to the host, so
  the container can reach the Cayu gateway there but has no route to the internet.
  In production, also firewall the host so the network reaches only the gateway
  (see [Sandbox and credential gateway](#sandbox-and-credential-gateway)).
- An image, here `claude-cli:local`, with the Claude Code CLI version that
  matches the installed `claude-agent-sdk` at `/usr/local/bin/claude`.
- `ANTHROPIC_API_KEY` set in the service's environment. It stays in the service:
  the container only gets a short-lived token for the Cayu gateway.
- `CAYU_ALIAS_KEY` set to a persistent secret: generate it once with
  `python -c "import secrets; print(secrets.token_urlsafe(32))"` and retain it
  across restarts alongside the database. It signs public authority aliases.

```python
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI
from pydantic import SecretStr

from cayu import CayuApp, Message, RunRequest
from cayu.budgets import BudgetLimit, default_price_book
from cayu.deadlines import ExecutionDeadline
from cayu.external_agents import ClaudeAgentAdapter, docker_agent_launch, run_external_agent
from cayu.external_agents.server import ExternalAgentGateway, ExternalModelRoute
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.vaults import SecretRedactor

MODEL = "claude-sonnet-5-5"  # Priced in default_price_book(), so the USD cap applies.
GATEWAY_HOST = "172.30.0.1"  # The host's address on the internal network.
CHECKOUT = Path("checkout").resolve()  # The only host directory the container sees.


async def allow_reads(name, arguments, call_id):
    return name in {"Read", "Grep", "Glob"}  # Without a callback every tool call is denied.


async def review(app, adapter):
    prompt = "Review the code in the current directory and list the three riskiest problems."
    result = await run_external_agent(
        app,
        RunRequest(
            agent_name="claude-review",
            messages=[Message.text("user", prompt)],
            execution_deadline=ExecutionDeadline.after(600),  # Required. Ten minutes.
            budget_limits=(BudgetLimit.usd(2),),  # Checked after each model call.
        ),
        adapter,
        prompt=prompt,
        authorize=allow_reads,
    )
    usage = await app.get_session_usage(result.session_id)
    cost = await app.get_session_cost(result.session_id, default_price_book())
    print(result.output)
    print(
        f"session {result.session_id}: {usage.usage.input_tokens} input and "
        f"{usage.usage.output_tokens} output tokens, about ${cost.total_cost}"
    )
    return result


def build_app(api_key):
    keyring = PublicAuthorityAliasKeyring(
        active_key_id="service", keys={"service": SecretStr(os.environ["CAYU_ALIAS_KEY"])}
    )
    return CayuApp(
        session_store=SQLiteSessionStore(
            "sessions.db", public_authority_alias_codec=PublicAuthorityAliasCodec(keyring)
        ),
        public_authority_alias_keyring=keyring,
        task_store=SQLiteTaskStore("tasks.db"),
        secret_redactor=SecretRedactor(api_key),  # Scrubbed from what Cayu stores.
    )


async def main():
    api_key = os.environ["ANTHROPIC_API_KEY"]
    app = build_app(api_key)
    async with httpx.AsyncClient() as http_client:
        gateway = ExternalAgentGateway(
            client=http_client,
            public_base_url=f"http://{GATEWAY_HOST}:8766",
            routes={
                "anthropic": ExternalModelRoute(
                    base_url="https://api.anthropic.com",
                    headers={"x-api-key": api_key},
                    paths=("/v1/messages",),
                )
            },
        )
        surface = FastAPI()
        surface.include_router(gateway.router)
        server = uvicorn.Server(uvicorn.Config(surface, host=GATEWAY_HOST, port=8766))
        serving = asyncio.create_task(server.serve())

        @asynccontextmanager
        async def launch(context):
            async with gateway.grant(
                context, route="anthropic", models=frozenset({MODEL})
            ) as capability:
                async with docker_agent_launch(
                    image="claude-cli:local",
                    cli="/usr/local/bin/claude",
                    workspace=CHECKOUT,
                    network="agent-internal",
                    workspace_read_only=True,
                    environment={
                        "ANTHROPIC_BASE_URL": capability.model_base_url,
                        "ANTHROPIC_AUTH_TOKEN": capability.token,
                    },
                ) as prepared:
                    yield prepared

        try:
            while not server.started:
                await asyncio.sleep(0.05)
            await review(app, ClaudeAgentAdapter(model=MODEL, launch=launch))
        finally:
            server.should_exit = True
            await serving
            await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

What each part gives you:

- `run_external_agent` creates a fresh Cayu session and task for the run and
  returns an `ExternalAgentResult` with `session_id`, `task_id` and `output`.
  The run fails, and the task with it, when the deadline passes. A run without a
  finite `execution_deadline` is refused.
- The gateway calls the model with the service's key and records each call's
  tokens on the session, so `app.get_session_usage`, `app.get_session_cost`,
  `cayu session usage SESSION_ID --sqlite sessions.db` and the dashboard show
  them. Once the cap is reached the gateway refuses further model calls and the
  run fails with `ExternalAgentBudgetExceeded`. One call can still go over the
  cap, because the vendor CLI makes its own calls and Cayu cannot reserve them.
- `SecretRedactor(api_key)` scrubs the key from stored events, transcripts and
  request metadata. The gateway token, and any `docker_agent_launch` environment
  variable whose name contains `KEY`, `TOKEN` or `AUTHORIZATION`, are scrubbed too.
- The CLI only sees the checkout, mounted read-only here, and can only reach the
  gateway. `ClaudeAgentAdapter` offers Claude Code the `Read`, `Grep` and `Glob`
  tools by default, and every tool call goes through `authorize`. To run tool
  decisions through Cayu policies and approvals instead, see
  [Native tools and permission decisions](#native-tools-and-permission-decisions).

For Codex, change the route, the token variable and the adapter in the example
above:

```python
from cayu.external_agents import CodexAgentAdapter

CODEX_MODEL = "gpt-6-sol"  # Priced in default_price_book().
openai_api_key = os.environ["OPENAI_API_KEY"]  # Pass this one to SecretRedactor.
routes = {
    "openai": ExternalModelRoute(
        base_url="https://api.openai.com/v1",
        headers={"authorization": f"Bearer {openai_api_key}"},
        paths=("/responses",),
    )
}
# In launch(): route="openai", models=frozenset({CODEX_MODEL}), cli="/usr/local/bin/codex"
# and environment={"CAYU_MODEL_PROXY_TOKEN": capability.token}.
# Codex asks `authorize` about "codex_command" and "codex_file_change" requests.
adapter = CodexAgentAdapter(
    model=CODEX_MODEL,
    base_url=f"http://{GATEWAY_HOST}:8766/models/openai",
    launch=launch,
    sandbox="external",  # The container is the sandbox; don't nest Codex's own.
)
```

## Launchers

A launcher starts the vendor CLI for one run and stops it again. It is an async
context manager that takes the run's `ExternalAgentContext` and yields an
`ExternalAgentLaunch`. Pass it to either adapter as `launch=`.

Cayu ships `local_agent_launch`, `docker_agent_launch` and `runner_agent_launch`.
The local launcher runs the CLI on the host without a sandbox, as in the
[quick start](#quick-start-without-docker-or-a-gateway). The Docker launcher
starts one container per run on an existing `--internal` network (it refuses
`host`, `bridge` and `default`), mounts only the checkout, runs the CLI as the
checkout's owner, and removes the container when the run ends.

Without Docker, use `runner_agent_launch` with a runner that supports interactive
processes, or write your own launcher. See the examples under
[Launchers without Docker](#launchers-without-docker). Yield
`ExternalAgentLaunch(executable, cwd, close, environment, secret_values)`, where
`executable` is a local command that runs the CLI inside your sandbox over
stdin and stdout, and `close` stops and waits for every process the CLI started.
Put any token you hand to the CLI in `secret_values` so Cayu scrubs it.
`mcp_servers` (Claude) and `codex_config_overrides` (Codex) are optional
per-run settings.

## Reference

The rest of this guide is the detailed contract.

`cayu.external_agents` exports a vendor-neutral `ExternalAgentAdapter` protocol,
`ExternalAgentEvent`, `ExternalAgentRecorder`, `run_external_agent`,
`ClaudeAgentAdapter`, `CodexAgentAdapter`, the local, Docker and runner launchers,
and the native tool bridge. SDK imports happen only when an adapter starts. The HTTP gateway lives in
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

When an adapter, a launcher or `runner_port_forward` can't confirm that a guest
process, sandbox or forward closed, it raises `ExternalCleanupIncomplete`, a
subclass of `ExternalAgentError`. Await its `retry()` to try the close again
before treating the guest as stopped.

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

To contain the CLI, run the whole CLI in a sandbox, because its built-in tools
run beside it. `local_agent_launch` does not do this. The SDK client stays in the
service and talks to the CLI over tunneled stdio.
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
explicit, and both adapters blank inherited host variables. The Docker and runner
launchers refuse guest API keys and subscription tokens; `local_agent_launch`
accepts them, because there is no guest boundary to keep them behind. A custom
sandbox launcher must provide the same isolation
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

The [production setup](#production-setup-docker-and-the-gateway) shows the
gateway and the Docker launcher wired together, including the uvicorn server that mounts `gateway.router` on the
internal network's host address.

For Codex, pass the fixed capability route base URL to `CodexAgentAdapter` and put
its token in `CAYU_MODEL_PROXY_TOKEN`. The adapter configures an OpenAI Responses
provider requiring only that scoped proxy token. `ExternalAgentLaunch` can also
supply per-run `mcp_servers` for Claude and `codex_config_overrides` for Codex.
Use service-side API credentials. Products should not expose subscription login
flows without the provider's required authorization. Consult the current
[Claude Agent SDK documentation](https://platform.claude.com/docs/en/agent-sdk/overview)
and [Codex app-server documentation](https://developers.openai.com/codex/app-server/).

## Launchers without Docker

A launcher is an async context manager that receives the run's
`ExternalAgentContext` and yields an `ExternalAgentLaunch`; pass it to an adapter as
`launch=`. `docker_agent_launch` is one launcher. Where there is no Docker daemon,
use `runner_agent_launch`, `local_agent_launch` for trusted code and tests, or
write your own.

### Run the CLI in a Cayu runner

`runner_agent_launch(runner=..., cli=..., environment=..., cwd=None, timeout_s=None)`
runs the CLI inside any runner whose `interactive_process_capability()` returns a
`RunnerInteractiveProcessCapability`. `LambdaMicroVMRunner` provides one through
its sidecar. Each CLI process runs in its own PID namespace in the MicroVM,
so closing it confirms that every process the CLI started has exited, including
processes that left its process group. `BrokeredSandboxRunner` provides one
through Cayu Cloud's sandbox broker, so a Cloud Agent can run the CLI in a
sandbox it gets from Cloud, with no cloud-provider credential. That needs a Cayu
Cloud whose broker supports interactive commands, which its sandbox records say
(`interactive_commands`); with any other, the runner offers no capability and
`runner_agent_launch` refuses it before starting anything. A brokered sandbox
has no port forwarding (see
[Model access from a runner](#model-access-from-a-runner)), so the CLI there
reaches the model over the sandbox's own egress. Another sandbox plugs in
by implementing the capability on its runner; the launcher and the adapters stay
the same.
If a failed start cannot confirm process cleanup, the capability must call
`retain_failed_start(error, process)` before re-raising the original error.
Callers recover the handle with `failed_start_process(error)` and retry `close()`
before treating the guest as stopped; this also preserves cancellation identity.

The runner is borrowed: you create it, put the CLI in its image, and close it.
The SDK starts a local stand-in that relays the CLI's arguments and stdio to the
runner over a private Unix socket. The launch's `cwd` is a host directory, because
the SDK starts the stand-in there; in the guest the same path links to the
runner's working directory. `timeout_s` bounds each guest process if the service
disappears without closing it, so derive it from the run's deadline. As with the
Docker launcher, model keys are refused in `environment`. The CLI reaches the
model through the gateway over a port forward; see
[Model access from a runner](#model-access-from-a-runner). The adapters
install a secret protector before starting the CLI; the launcher registers each
process's newly issued secrets before forwarding output. Direct launcher users
that publish output must call `prepared.protect_secrets(protector)` first.

```python
import math
from contextlib import asynccontextmanager

from cayu.external_agents import ClaudeAgentAdapter, runner_agent_launch

@asynccontextmanager
async def launch(context):
    # runner: a runner you own, such as a LambdaMicroVMRunner whose image has the CLI.
    async with runner_agent_launch(
        runner=runner,
        cli="/usr/local/bin/claude",
        environment=model_route_environment,
        timeout_s=(None if (remaining := context.execution_deadline.remaining_seconds()) is None
                   else math.ceil(remaining) + 60),
    ) as prepared:
        yield prepared

adapter = ClaudeAgentAdapter(model=model, launch=launch)
```

### Model access from a runner

A CLI in a MicroVM has no route to your service, and you don't need to give it
one. `runner_port_forward(runner, host=..., port=..., timeout_s=None)` opens a
listening address inside the guest and relays each connection to `host`:`port` on
your host, over the same owner-fenced channel the CLI's stdio uses. Serve
`gateway.router` on this host's loopback only, open a forward to it, and grant
with `public_base_url=forward.guest_url` so the capability's URLs name the guest
address. The real model key stays in the gateway; the guest gets only the run's
token. No public gateway URL, no inbound route to the guest and no extra Lambda
egress connector are needed.

The forward is a provider-neutral runner capability:
`Runner.port_forward_capability()` returns a `RunnerPortForwardCapability` whose
`open_port_forward(host=..., port=..., timeout_s=None)` returns a
`RunnerPortForward` with `guest_host`, `guest_port`, `guest_url` and `close()`.
`LambdaMicroVMRunner` provides it through sidecar protocol 7: the guest listens on
its agent network's gateway address, `192.0.2.1`, at a port from 18500 to 18563,
reachable only from processes in the MicroVM. Another sandbox plugs in by
implementing the capability on its runner. `BrokeredSandboxRunner` doesn't:
Cayu Cloud's broker relays no connections into its sandboxes, so
`runner_port_forward` refuses a brokered sandbox, and a CLI there needs a
gateway URL it can reach through the sandbox's `public_web` egress.

Put the per-run URL in the CLI's configuration. Claude Code takes it from the
environment. Codex takes it from `codex_config_overrides`, which replaces the
adapter's `base_url`, and native tools over MCP take `capability.mcp_url` through
`mcp_servers`:

```python
import math
from contextlib import asynccontextmanager

from cayu.external_agents import (
    ClaudeAgentAdapter,
    CodexAgentAdapter,
    codex_proxy_overrides,
    runner_agent_launch,
    runner_port_forward,
)

GATEWAY_PORT = 8766  # uvicorn serves gateway.router on 127.0.0.1:8766 only.


def guest_timeout(context):
    remaining = context.execution_deadline.remaining_seconds()
    return None if remaining is None else math.ceil(remaining) + 60


@asynccontextmanager
async def claude_launch(context):
    async with (
        runner_port_forward(
            runner, host="127.0.0.1", port=GATEWAY_PORT, timeout_s=guest_timeout(context)
        ) as forward,
        gateway.grant(
            context,
            route="anthropic",
            models=frozenset({MODEL}),
            public_base_url=forward.guest_url,
        ) as capability,
        runner_agent_launch(
            runner=runner,
            cli="/usr/local/bin/claude",
            environment={
                "ANTHROPIC_BASE_URL": capability.model_base_url,
                "ANTHROPIC_AUTH_TOKEN": capability.token,
            },
            timeout_s=guest_timeout(context),
        ) as prepared,
    ):
        yield prepared


@asynccontextmanager
async def codex_launch(context):
    async with (
        runner_port_forward(
            runner, host="127.0.0.1", port=GATEWAY_PORT, timeout_s=guest_timeout(context)
        ) as forward,
        gateway.grant(
            context,
            route="openai",
            models=frozenset({CODEX_MODEL}),
            public_base_url=forward.guest_url,
        ) as capability,
        runner_agent_launch(
            runner=runner,
            cli="/usr/local/bin/codex",
            environment={"CAYU_MODEL_PROXY_TOKEN": capability.token},
            codex_config_overrides=codex_proxy_overrides(capability.model_base_url),
            timeout_s=guest_timeout(context),
        ) as prepared,
    ):
        yield prepared


claude = ClaudeAgentAdapter(model=MODEL, launch=claude_launch)
# The launch's overrides replace this placeholder base URL for each run.
codex = CodexAgentAdapter(
    model=CODEX_MODEL,
    base_url="http://127.0.0.1/models/openai",
    launch=codex_launch,
    sandbox="external",
)
```

Leaving the block closes the CLI, revokes the token and closes the guest listener,
in that order. `timeout_s` bounds the listener if the service disappears first.

What to know about this route:

- **No TLS inside the guest.** The CLI talks plain HTTP to the guest address, the
  bytes cross to your host inside the runner's authenticated HTTPS channel, and the
  host connects to the gateway on loopback. The image needs no Cayu CA. The relay
  passes bytes unchanged, so an HTTPS target would stay end to end.
- **A leaked token is limited to the run.** Anything in the guest can read the
  token, because the CLI must. It works only for the granted route and models, only
  on the Messages or Responses path, only within the run's budget checks, and only
  until the run's deadline or the end of the grant, whichever comes first. It never
  reveals the provider key, which the gateway adds upstream. It is also scrubbed
  from what Cayu stores.
- **Budgets hold.** The gateway checks the run's budget before every model call, so
  a call that would go over a reached cap is refused before it is sent.
- **Streaming.** The relay forwards each chunk as soon as the gateway writes it.
  Measured with a fake streaming upstream and a simulated 50 ms round trip to the
  MicroVM endpoint, events reached the guest a median of about 50 ms after the
  upstream sent them, 80 ms at worst; with no added round trip the median was about
  1 ms. Expect roughly one endpoint round trip per chunk. The gateway itself buffers each model
  response until it ends, so it can commit usage before the CLI sees a finished
  response; the CLI receives the whole stream then.
- **Claude Code** (2.1.306, the CLI bundled with `claude-agent-sdk` 0.2.170)
  uses `ANTHROPIC_BASE_URL` with `ANTHROPIC_AUTH_TOKEN` and sends the token as
  `Authorization: Bearer`, which the gateway accepts. It may substitute a newer
  model for a retired name (for example it sent `claude-opus-5-5` when asked for
  `claude-opus-4-1`), so grant the model name the CLI actually sends.
- **Codex** (0.163.0, the version the pinned `openai-codex` uses) reads the token
  from `CAYU_MODEL_PROXY_TOKEN`, as `codex_proxy_overrides` configures, and sends it
  as `Authorization: Bearer` to the overridden `base_url`. Leave `CODEX_HOME` unset
  or point it at an existing directory; Codex exits if it is missing.
- **Proxies.** If the runner's environment sets `HTTP_PROXY`, add the forward's
  `guest_host` to `NO_PROXY` in `environment` so the CLI connects to it directly.

To check this on real AWS, build a MicroVM image from this release's sidecar with
the CLIs at the versions above, for example by adding
`RUN python3.11 -m pip install claude-agent-sdk==0.2.170 openai-codex-cli-bin==0.163.0`
to the exported Dockerfile and linking the bundled `claude` and `codex` binaries
into `/usr/local/bin`. Then run the quick start with the launches above, a small
prompt and `BudgetLimit.usd(0.05)`, and confirm that:

- the run completes and `app.get_session_usage` shows the call's tokens;
- with a cap already reached, the next model call returns HTTP 403 from the
  gateway and the run fails with `ExternalAgentBudgetExceeded`;
- the session's events and the guest's environment contain no provider key;
- after the run, a request with the old token to the gateway returns 401, and
  `runner.exec(ExecCommand.shell("curl -sS http://192.0.2.1:18500/"))` fails to
  connect.

### Write your own launcher

Yield `ExternalAgentLaunch(executable, cwd, close, environment, secret_values)`:

- `executable` is a local command that runs the whole CLI inside your sandbox and
  relays its stdin and stdout. The SDK starts it with `environment` (inherited host
  variables are blanked) and the CLI's arguments.
- `cwd` must exist on the host, where the SDK starts `executable`, and name the
  checkout in the sandbox, because Codex also uses it as its working directory.
- `close` stops every process the CLI started and returns only once they have all
  exited. Removing a container or ending a PID namespace gives that guarantee;
  killing one process does not. It may be called more than once. If it raises,
  Cayu keeps the run's task live and retries it.
- `secret_values` lists every token you hand to the CLI, so Cayu redacts them.
- `mcp_servers` (Claude) and `codex_config_overrides` (Codex) are optional
  per-run settings.

This launcher uses Podman, which needs no daemon. It runs one container per run on
an existing internal network whose only reachable service is the gateway, mounts
only the checkout at the same path, and removes the container on close:

```python
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from cayu.external_agents import ExternalAgentLaunch

async def podman(*args: str) -> str:
    process = await asyncio.create_subprocess_exec(
        "podman", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    stdout, _ = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"podman {args[0]} failed")
    return stdout.decode().strip()

@asynccontextmanager
async def podman_launch(checkout: Path, environment: dict[str, str]):
    container = await podman(
        "run", "--detach", "--network", "agent-internal", "--userns", "keep-id",
        "--volume", f"{checkout}:{checkout}", "--workdir", str(checkout),
        "my-agent-image:version", "sleep", "infinity",
    )

    async def close() -> None:
        # Removal kills and waits for every process in the container, and
        # --force tolerates a container that is already gone.
        await podman("rm", "--force", "--time", "0", container)

    try:
        with TemporaryDirectory() as state:
            cli = Path(state) / "cli"
            flags = " ".join(f"--env {key}" for key in environment)
            cli.write_text(
                "#!/bin/sh\n"
                f'exec podman exec --interactive {flags} {container} '
                '/usr/local/bin/claude "$@"\n'
            )
            cli.chmod(0o700)
            yield ExternalAgentLaunch(
                executable=str(cli),
                cwd=str(checkout),
                close=close,
                # Rootless Podman needs its own home and runtime directory.
                environment={
                    "PATH": "/usr/bin:/bin",
                    "HOME": os.environ["HOME"],
                    "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", ""),
                    **environment,
                },
                secret_values=(environment["ANTHROPIC_AUTH_TOKEN"],),
            )
    finally:
        await close()
```

Use it inside a per-run `launch(context)` like the Docker example above, passing
the gateway capability's `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN`.

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
