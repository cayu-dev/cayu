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

Cayu is in early development. The current codebase is a framework foundation/runtime slice: it includes core contracts, environment registration, local workspace/runner/artifact-store implementations, framework-native file, artifact, command, and stdio MCP tool adapters, first-class tool policies for scoped authority and durable tool approvals, in-memory and SQLite session/event/transcript stores, explicit session resume and session fork with persisted provider/model identity, in-memory and SQLite task stores, event sinks, model-provider contracts, model-facing context policies, checkpoint-backed context compaction, initial Anthropic Messages API and OpenAI Responses API providers with certifi-backed TLS verification, structured message/tool-call handling, tool execution, tool-result feedback to the model, max-step protection, validation for framework boundary data, and an optional FastAPI server with a packaged dashboard for inspecting runs, sessions, tasks, transcripts, and events.

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
behind the optional `cayu[microsandbox]` extra for microVM-backed command
execution when agents need an isolated sandbox boundary.

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

The OpenAI provider uses Responses API streaming by default. Tune the ordinary HTTP
timeout and the no-provider-event stall timeout separately:

```python
from cayu import OpenAIProvider

provider = OpenAIProvider(
    timeout_s=600,
    stream_idle_timeout_s=300,
)
```

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
