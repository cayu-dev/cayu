# Runtime Contracts

This is a design/maintainer document for the current framework foundation. It names the first contracts that must stabilize before the framework grows higher-level features.

## Boundary Data

Framework boundary data should be portable across local processes, remote runners, hosted runtimes, event stores, dashboards, and replay tools.

Payloads, metadata, tool arguments, tool results, model options, checkpoints, and event data are JSON data. They must contain JSON-compatible values: objects, arrays, strings, integers, finite floats, booleans, and null. Tuples, arbitrary Python objects, non-string object keys, circular references, NaN, and Infinity are not valid boundary data.

Runtime APIs copy framework objects at boundaries. User code should not mutate registered specs, request objects, message parts, event payloads, tool results, or provider events and expect those mutations to change already-registered or already-emitted runtime state.

## Agent, Environment, Session

Cayu separates agent definition, execution environment, and session state:

- `AgentSpec`: model, system prompt, tool declarations, and metadata.
- `Environment`: workspace, runner, vault, MCP servers, and execution metadata.
- `RunRequest` / `Session`: one run of an agent, optionally in a named environment, with messages, status, events, and checkpoints.

This mirrors the useful Managed Agents separation of brain, hands, and durable run history without copying any one provider API. A run may omit an environment for simple provider/tool tests, but concrete file, command, sandbox, vault, or MCP-backed tools should hang off an environment.

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

Creates sessions, stores events, and checkpoints runtime state.

`RunRequest.session_id` is an optional caller-provided id for a new session. It must be unique. Resume, replay, and idempotent continuation should be explicit APIs later, not implied by session creation.
`RunRequest.environment_name` optionally selects a registered environment. If omitted, the runtime may use the default registered environment; if no environment is registered, simple runs can still execute without one.
Events emitted for an environment-backed run carry `environment_name` as a top-level event identity field, not as payload data. Runtime code owns this field and normalizes provider events before emitting them.

Local default can be SQLite. Hosted use can be Postgres or another durable store.
`InMemorySessionStore` exists for tests, local examples, and the first runtime slice.

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

The initial `CayuApp` runtime registers agent specs, model providers, and tools, then emits and persists events for one session run. A run may make multiple model requests: model output can request tools, the runtime executes those tools, appends assistant `tool_call` messages and matching `tool_result` messages, and calls the model again until the model completes without tool calls or `RunRequest.max_steps` is exceeded. Multiple tool calls from one model step are grouped into one assistant message and one tool-result message in Cayu's internal transcript. Provider adapters must emit a `completed` stream event for each model step; a stream that ends silently is treated as a failed runtime contract.

`CayuApp.run()` is an event-stream API. Runtime failures are represented as terminal `session.failed` events rather than re-raised exceptions from the iterator. A stricter programmatic API can be added later on top of the same runtime path.

## Provider

Model providers translate model-specific APIs into Cayu runtime contracts.

Provider adapters must:

- receive a copied `ModelRequest`
- yield `ModelStreamEvent` values
- emit a `completed` stream event for each model step
- stop emitting after `completed`
- convert stream events into `Event` values for the current session

Provider errors should become model error events and failed sessions. Tool calls should be emitted as structured tool-call stream events so the runtime can execute tools and feed structured results back into the next model step.

## Tool

Runs a capability and returns `ToolResult`.

Tool declarations are captured when an agent is registered with `CayuApp`.
The registered name, description, and input schema are the public contract shown to the model for that agent.
Changing `tool.spec` after registration does not update the registered agent or the model-facing tool declaration.
To change a tool's public contract, create/register a new agent configuration or re-register the tool through an explicit runtime API once one exists.

Tool results must support:

- model-facing text
- structured output
- artifacts
- error state

String-only tool results are not enough for the final framework.

Tool failures are recoverable by default. They are recorded as `tool.call.failed` events and returned to the model as structured `tool_result` message parts with `is_error=true`. The session itself should fail for provider errors, runtime contract violations, max-step exhaustion, storage failures, or unrecoverable infrastructure problems.

## Workflow

Coordinates deterministic or agent-assisted multi-step execution.

Workflows need durable step state, retries, pause/resume, failure modes, and event emission.

## Runner

Executes commands/code and returns stdout, stderr, exit code, timeout/cancel flags, and artifacts.

Runner commands use `ExecCommand`:

- `process`: explicit argv list for normal command execution
- `shell`: explicit shell script for bash-like behavior

The framework should not pass a single ambiguous command string to runners. Use process mode unless shell parsing, expansion, and quoting are intentional.

Remote runners may talk to a runner service inside EC2/ECS/Daytona/etc.
`LocalRunner` is available for development and trusted local execution. It is not a sandbox. By default it inherits the parent process environment and overlays any explicit `env` values; set `inherit_env=False` when commands should only receive the explicit environment passed to the runner.

## Workspace

Filesystem/artifact boundary. For coding agents this is often a target repo. For document/data agents this may be uploaded files and generated outputs.
`LocalWorkspace` is available for local filesystem-backed work. It resolves paths under one root and rejects path traversal outside that root.

## Vault

Secrets abstraction. Raw secret values should be injected into tools/runners by runtime and should not be placed in model prompts.

MCP config separates plain and secret values:

- `env` / `headers`: non-secret strings
- `secret_env` / `secret_headers`: `SecretRef` values resolved by runtime

The framework should not guess whether a key name is sensitive.

## KnowledgeStore

Searchable memory/knowledge interface. Default local implementation should eventually support file indexing plus SQLite FTS.
