# Runtime Contracts

This document names the first contracts that must stabilize before the framework grows higher-level features.

## Event

Append-only event emitted by runtime, providers, tools, workflows, memory, runners, and sessions.
Framework event types are enumerated. Extension events must use the `custom.` namespace so typos do not silently become durable event names.

Events power:

- terminal output
- dashboard
- webhooks
- session replay
- hosted platform adapters
- tests and debugging

## SessionStore

Creates sessions, stores events, and checkpoints resumable state.

Local default can be SQLite. Hosted use can be Postgres or another durable store.

## EventSink

Receives events and forwards them somewhere:

- stdout
- dashboard websocket
- webhook
- log file
- database
- hosted platform adapter

## Agent

Turns messages into event streams using:

- model providers
- tools
- memory
- workflows
- runtime services

## Tool

Runs a capability and returns `ToolResult`.

Tool results must support:

- model-facing text
- structured output
- artifacts
- error state

String-only tool results are not enough for the final framework.

## Workflow

Coordinates deterministic or agent-assisted multi-step execution.

Workflows need durable step state, retries, pause/resume, failure modes, and event emission.

## Runner

Executes commands/code and returns stdout, stderr, exit code, timeout/cancel flags, and artifacts.

Runner commands use `ExecCommand`:

- `process`: explicit argv list for normal command execution
- `shell`: explicit shell script for bash-like behavior

The framework should not pass a single ambiguous command string to runners.

Remote runners may talk to a runner service inside EC2/ECS/Daytona/etc.

## Workspace

Filesystem/artifact boundary. For coding agents this is often a target repo. For document/data agents this may be uploaded files and generated outputs.

## Vault

Secrets abstraction. Raw secret values should be injected into tools/runners by runtime and should not be placed in model prompts.

MCP config separates plain and secret values:

- `env` / `headers`: non-secret strings
- `secret_env` / `secret_headers`: `SecretRef` values resolved by runtime

The framework should not guess whether a key name is sensitive.

## KnowledgeStore

Searchable memory/knowledge interface. Default local implementation should eventually support file indexing plus SQLite FTS.
