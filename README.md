# Cayu

Cayu is an open-source Python framework for building long-running agents, multi-agent workflows, and sandboxed tool runtimes.

## Design Goals

- Build real agent applications, not just hosted prompt/config definitions.
- Support multiple agents collaborating through shared state, channels, tasks, triggers, and workflows.
- Treat tool execution as a first-class runtime concern with explicit workspace, runner, and sandbox contracts.
- Store every important run action as structured events for CLI output, dashboard inspection, webhooks, and replay.
- Run locally, in containers, on hosted infrastructure, or behind an application server.
- Make MCP an interoperability layer, not the only custom tool model.

## Status

Cayu is in early development. The current codebase is a framework foundation/runtime slice: it includes core contracts, environment registration, local workspace/runner implementations, framework-native file and command tools, in-memory and SQLite session/event stores, in-memory and SQLite task stores, event sinks, model-provider contracts, an initial Anthropic Messages API provider with certifi-backed TLS verification, structured message/tool-call handling, tool execution, tool-result feedback to the model, max-step protection, and validation for framework boundary data.

It does not yet include dashboard UI, hosted deployment adapters, vector search, isolated runners, higher-level task orchestration, or streaming provider adapters.

## Contract Rules

Cayu treats payloads, metadata, tool arguments, tool results, model options, checkpoints, task data, and event data as JSON data. These fields must contain JSON-compatible values: objects, arrays, strings, integers, finite floats, booleans, and null. Tuples, arbitrary Python objects, non-string object keys, circular references, NaN, and Infinity are rejected. Task input, result, error, and metadata fields are top-level JSON objects with JSON-compatible nested values.

Framework objects are copied at runtime boundaries. Mutating an agent, environment, or tool object after registration is not part of the public contract. To change a registered declaration, register a new configuration or use an explicit update API once one exists.

Framework-native tools receive runtime services through `ToolContext`: workspace, runner, vault, and MCP server specs. Those service references are runtime-only and are excluded from serialized context data.

## Initial Layout

```text
src/cayu/
  core/        framework primitives: events, messages, agents, tools, workflows
  environments/ execution context contracts
  runtime/     app runtime, sessions, event sinks
  runners/     command execution backends
  workspaces/  filesystem/artifact workspace contracts
  storage/     storage contracts and SQLite implementations
  providers/   model provider contracts
  mcp/         MCP client/server integration contracts
  vaults/      secrets access contracts
  cli/         developer/admin CLI
```

## Development

Standard Python:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Or with `uv`:

```bash
uv sync
uv run pytest
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

Run the live Anthropic example with local tools:

```bash
export ANTHROPIC_API_KEY=...
PYTHONPATH=src python examples/anthropic_local_tools.py
```

Use durable local session/event storage:

```python
from pathlib import Path

from cayu import CayuApp, EventQuery, SQLiteSessionStore

store = SQLiteSessionStore(Path(".cayu") / "sessions.sqlite")
app = CayuApp(session_store=store)

async def inspect_session(session_id: str):
    return await store.query_events(EventQuery(session_id=session_id))
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
