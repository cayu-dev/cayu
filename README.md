# Cayu

Cayu is an open-source Python framework for building long-running agents, multi-agent workflows, and sandboxed tool runtimes.

## Design Goals

- Build real agent applications, not just hosted prompt/config definitions.
- Support multiple agents collaborating through shared state, channels, tasks, triggers, and workflows.
- Treat tool execution as a first-class runtime concern with explicit workspace, runner, vault, and sandbox contracts.
- Store every important run action as structured events for CLI output, dashboard inspection, webhooks, and replay.
- Run locally, in containers, on hosted infrastructure, or behind an application server.
- Make MCP an interoperability layer, not the only custom tool model.

## Status

Cayu is in early development. The current codebase is a framework foundation/runtime slice: it includes core contracts, environment registration, local workspace/runner/artifact-store implementations, framework-native file, artifact, command, and stdio MCP tool adapters, first-class tool policies for scoped authority and durable tool approvals, in-memory and SQLite session/event/transcript stores, explicit session resume, resumable session interruption, session-level usage/cache summaries, hard token/tool/time run limits, and session fork with persisted provider/model identity, in-memory and SQLite task stores, event sinks and structured runtime logging, model-provider contracts, model-facing context policies, checkpoint-backed context compaction, initial Anthropic Messages API and OpenAI Responses API providers with certifi-backed TLS verification, structured message/tool-call handling, tool execution, tool-result feedback to the model, max-step protection, validation for framework boundary data, and an optional FastAPI server with a packaged dashboard for inspecting runs, sessions, tasks, transcripts, and events.

It does not yet include hosted deployment adapters, vector search, or higher-level task orchestration.

## Contract Rules

Cayu treats payloads, metadata, tool arguments, tool results, model options, checkpoints, task data, and event data as JSON data. These fields must contain JSON-compatible values: objects, arrays, strings, integers, finite floats, booleans, and null. Tuples, arbitrary Python objects, non-string object keys, circular references, NaN, and Infinity are rejected. Task input, result, error, and metadata fields are top-level JSON objects with JSON-compatible nested values.

Framework objects are copied at runtime boundaries. Mutating an agent, environment, or tool object after registration is not part of the public contract. To change a registered declaration, register a new configuration or use an explicit update API once one exists.

Framework-native tools receive runtime services through `ToolContext`: workspace, artifact store, runner, vault, and MCP server specs. Those service references are runtime-only and are excluded from serialized context data.

Tool policies authorize registered tool calls before execution. Denied calls emit `tool.call.blocked`, do not run the tool, and are returned to the model as error tool results so the session can continue.

## Initial Layout

```text
src/cayu/
  core/        framework primitives: events, messages, agents, tools, workflows
  artifacts/   uploaded/generated file storage contracts
  environments/ execution context contracts
  runtime/     app runtime, sessions, event sinks
  runners/     command execution backends
  workspaces/  filesystem workspace contracts
  storage/     storage contracts and SQLite implementations
  providers/   model provider contracts
  mcp/         MCP client/server integration contracts
  vaults/      secrets access contracts
  cli/         developer/admin CLI
```

`LocalRunner` is for trusted local development. `MicrosandboxRunner` is available
behind the optional `cayu[microsandbox]` extra for local microVM-backed command
execution. `E2BRunner` and `E2BWorkspace` are available behind the optional
`cayu[e2b]` extra for E2B cloud sandbox execution and native E2B filesystem
access.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src/ tests/ examples/
uv run ruff format src/ tests/ examples/
uv run ty check src/cayu examples
```

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

The OpenAI provider uses Responses API streaming by default. Tune the ordinary HTTP
timeout and the no-provider-event stall timeout separately:

```python
from cayu import OpenAIProvider

provider = OpenAIProvider(
    timeout_s=600,
    stream_idle_timeout_s=300,
)
```

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

The raw provider `usage` payload remains available on each durable
`model.completed` event for dashboards, audits, and provider-specific
diagnostics. `usage_metrics` is Cayu's stable summary shape; raw `usage` is the
exact provider payload. If a provider reports usage fields Cayu does not
understand yet, the event still keeps raw `usage` even when `usage_metrics` is
absent.

For dashboards, CLIs, and audit views, the optional server exposes paginated
durable events at `GET /api/sessions/{session_id}/events`. It supports
`after_sequence`, `limit`, `event_type`, `tool_name`, `agent_name`,
`environment_name`, and `workflow_name` query parameters. Responses include each
event's durable `sequence` plus `has_more` and `next_sequence`, so clients can
poll or page without loading the full session transcript.

The provider-neutral transcript is exposed separately at
`GET /api/sessions/{session_id}/transcript`. It supports `offset`, `limit`, and
`role` filters and returns each message with its zero-based transcript `index`.
Use events to inspect what happened; use transcript to inspect the conversation
state Cayu will use for resume, compaction, and provider requests.

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
        model="gpt-5.5",
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

Cayu blocks provider options that would replace framework-owned fields such as
messages, tools, `store`, or `previous_response_id`, but it preserves
provider-specific cache controls as request options. OpenAI prompt caching is
mostly automatic and can use `prompt_cache_key` / `prompt_cache_retention`.
Anthropic automatic prompt caching uses top-level `cache_control`. Explicit
Anthropic block-level cache breakpoints are not a Cayu-level abstraction yet;
they require block-level provider metadata and should be added deliberately if
the transcript contract grows that capability.

Provider options configured on `AgentSpec` are copied into each normal model
request for that agent. Provider-backed compaction can use the existing
`ModelCompactor(options=...)` argument for compaction-model cache settings.

Estimate session cost from the same durable usage events by passing your own
pricing table. Cayu does not ship hardcoded provider prices because those change
outside the framework:

```python
from decimal import Decimal

from cayu import ModelPricing, PricingCatalog

pricing = PricingCatalog(
    prices=(
        ModelPricing(
            provider_name="openai",
            model="gpt-5.5",
            match="prefix",
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

Cost estimation walks each durable `model.completed` event, matches its
provider/model against the pricing catalog, and returns line items. A pricing
entry can match an exact model name or a provider model-name prefix so callers
can handle provider snapshot suffixes. Missing pricing is reported as unpriced
line items instead of being silently treated as free. If cache read/write prices
are omitted, Cayu falls back to the normal input-token price for those counters;
provide explicit cache prices when your provider or account charges them
differently.

The optional server exposes the same estimate at
`POST /api/sessions/{session_id}/cost`. The request body supplies the pricing
catalog because Cayu does not hardcode provider prices:

```json
{
  "pricing": {
    "prices": [
      {
        "provider_name": "openai",
        "model": "gpt-5.5",
        "match": "prefix",
        "input_per_million": "2.00",
        "output_per_million": "8.00",
        "cache_read_input_per_million": "0.50"
      }
    ]
  }
}
```

Run `examples/usage_cost_summary.py` for a deterministic local example that
prints normalized usage, cache counters, and estimated cost without calling a
real provider API.

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

Set hard run limits with `RunLimits` and estimated-cost budgets with
`CostBudget` on `RunRequest`, `ResumeRequest`, `DispatchRequest`, or
tool-approval continuation requests:

```python
from decimal import Decimal

from cayu import CostBudget, Message, RunLimits, RunRequest

request = RunRequest(
    agent_name="assistant",
    messages=[Message.text("user", "Analyze these invoices.")],
    limits=RunLimits(
        max_total_tokens=50_000,
        max_tool_calls=25,
        max_elapsed_seconds=300,
        scope="session",
    ),
    cost_budget=CostBudget(
        max_estimated_cost=Decimal("0.50"),
        pricing=pricing,
        scope="session",
    ),
)
```

Token and tool-call limits are evaluated from durable session events, so they
apply across resume and dispatch paths by default. `scope="session"` is the
default and treats token/tool-call limits as lifetime session budgets.
`scope="run"` evaluates token/tool-call limits against only the current
`run(...)`, `resume(...)`, dispatch, or approval-continuation invocation. Elapsed
time is always evaluated for the active runtime invocation and resets on each
call. Cost budgets use the same scope names: `scope="session"` enforces a
lifetime estimated-cost budget, while `scope="run"` compares only estimated cost
added during the current invocation.

Cost budgets are estimates derived from normalized usage metrics and the pricing
catalog supplied by your app. They are not provider invoices. By default,
`CostBudget` fails closed when a newly observed model step has no matching
pricing entry, because Cayu cannot prove that the budget is still safe. Set
`allow_unpriced=True` only when your app intentionally allows missing prices for
that run.

When a limit is reached, Cayu emits `session.limit_reached`, marks the session
`interrupted`, emits `session.interrupted` with
`interruption_type="limit_reached"`, and leaves the session resumable. Resuming
with the same exhausted session-scoped token or tool-call budget will interrupt
again immediately; pass a higher budget, omit that limit, or use `scope="run"`
if "continue" should mean "give this invocation a fresh token/tool/cost budget."
In a limit event, `actual` is the value evaluated for the selected scope, while
`usage_summary` remains the cumulative session summary. Cost-limit events also
include the cumulative `cost_summary`; decimal cost values are serialized as
strings for JSON stability. If the model requested tools in the same step, Cayu
records skipped tool results before interrupting so the provider-neutral
transcript remains valid for resume.

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

Use durable local task storage for optional background work tracking:

```python
from pathlib import Path

from cayu import SQLiteTaskStore, TaskCreate

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
    model="gpt-5.5",
)
```

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

Use `strip_old_file_attachments(...)` inside custom context policies when you build your
own transcript projection and want the same bounded native-file behavior.

Scope tool authority per agent:

```python
from cayu import AgentSpec, ExecCommandTool, ListFilesTool, ReadFileTool, StaticToolPolicy

app.register_agent(
    AgentSpec(name="reviewer", model="gpt-5.5"),
    tools=[ReadFileTool(), ListFilesTool(), ExecCommandTool()],
    tool_policy=StaticToolPolicy(allow=["read_file", "list_files"]),
)
```

Custom policies can also require caller approval before a tool round runs. The runtime checkpoints the pending approval, emits `tool.call.approval_requested`, marks the session `interrupted`, and waits for `resolve_tool_approval(...)`:

```python
from cayu import ToolApprovalDecision, ToolApprovalRequest

async for event in app.resolve_tool_approval(
    ToolApprovalRequest(
        session_id="sess_123",
        approval_id="approval_123",
        decision=ToolApprovalDecision.APPROVE,
    )
):
    print(event.type)
```

While a tool approval is pending, `app.resume(...)` rejects normal message resume. Approving or denying the pending approval writes the required tool results and clears the pending checkpoint atomically, then the model loop continues with valid provider-neutral history. If approval close is retried, Cayu reuses durable terminal tool events instead of running completed tools again. If a side-effecting tool started but Cayu cannot prove whether it completed, record the known external outcome and continue without re-running the tool:

```python
from cayu import ToolApprovalRecoveryOutcome, ToolApprovalRecoveryRequest

async for event in app.recover_tool_approval(
    ToolApprovalRecoveryRequest(
        session_id="sess_123",
        approval_id="approval_123",
        tool_call_id="call_123",
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
