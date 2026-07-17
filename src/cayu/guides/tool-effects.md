# Choosing a ToolEffect

`ToolEffect` tells Cayu what replay can do to externally meaningful durable
state. Classify the operation's behavior, not its implementation, transport,
price, or name.

- `NONE`: replay creates no externally meaningful durable mutation and needs no
  downstream idempotency identity.
- `IDEMPOTENT`: the operation may mutate state, but a stable downstream
  idempotency contract or key collapses repeated execution.
- `EXTERNAL`: the operation has a non-idempotent or outcome-ambiguous external
  effect that generic retry must never assume is safe to repeat.

## Canonical decisions

| Operation behavior | ToolEffect | Why |
| --- | --- | --- |
| Pure computation | `NONE` | Replay creates no externally meaningful durable mutation. |
| Stable local file, database, search, or public HTTP read, including a paid or logged read | `NONE` | Transport, billing, and observability do not turn a read into a mutation. |
| Consuming read, dequeue, or one-time-token invalidation without a stable replay contract | `EXTERNAL` | The call mutates durable state and replay may consume again. |
| Read that creates a durable snapshot or artifact without a stable replay contract | `EXTERNAL` | Snapshot creation is a durable mutation and replay may create another snapshot. |
| Write or other mutating operation with a stable downstream idempotency key or equivalent replay contract | `IDEMPOTENT` | The downstream system collapses repeated execution through a stable operation identity or equivalent idempotency contract. |
| Ordinary file write, email, payment, or other mutation without such a contract | `EXTERNAL` | Replay may repeat the mutation. |
| Mutating request whose outcome is unknown after a timeout or disconnect and has no stable replay contract | `EXTERNAL` | The caller cannot prove whether the first mutation committed or safely collapse replay. |

An ordinary public HTTP GET used for remote research is therefore `NONE`. It
does not become `IDEMPOTENT` merely because it crosses a network, is logged, is
rate-limited, or costs money. Conversely, a method or tool named "read" is not
necessarily `NONE`: dequeueing work, consuming a one-time token, or creating a
durable artifact snapshot mutates state.

When behavior is uncertain, use `EXTERNAL` until the application can name and
test a stable downstream replay contract.

The classification describes replay safety; it does not mean Cayu will
automatically retry the tool.

## Check a `NONE` declaration before deployment

`verify_tool_effect(...)` is an explicit deployment-readiness test seam. It
invokes one registered tool against a bounded temporary Cayu workspace, then
reports the declared effect and any created, updated, or deleted paths:

```python
from cayu.testing import ToolEffectVerificationStatus, verify_tool_effect

evidence = await verify_tool_effect(
    app,
    agent_name="reporter",
    tool_name="calculate_report_total",
    arguments={"source": "input.json"},
    workspace_files={"input.json": b'{"total": 42}'},
    unobserved_systems=("reporting_database",),
)
assert evidence.status is ToolEffectVerificationStatus.CONSISTENT
```

For `NONE`, an unchanged workspace is `consistent`; any observed create,
update, or delete is a `mismatch`. This is scoped evidence, not proof that the
tool is universally pure. This first observer compares regular-file paths and
content only. Empty directories, symlinks, other non-regular entries,
permissions, timestamps, and filesystem metadata are outside its mutation
evidence, although every traversed entry counts toward the observation limit.
The result always names systems outside the boundary, including network
services, databases outside the workspace, artifact stores, runner execution,
process state, and host paths outside the temporary workspace. Add
application-specific systems through `unobserved_systems`.

The verifier supplies no runner, artifact store, vault, proxy, or knowledge
store and runs the tool directly without policy, approvals, hooks, events, or
the model loop. Build a fresh application with controlled adapters for this
test: the current Python process is not a security sandbox, and tool-instance
state is not observed. One cooperative asyncio deadline covers workspace
seeding, tool execution, both snapshots, and cleanup checks. If it expires, the
helper raises `TimeoutError` and returns no verdict because observation did not
complete. A tool or filesystem operation that blocks the event loop can delay
that failure; enforcing a hard wall-clock stop requires a killable process
boundary. Snapshots stop at configured traversed-entry and regular-file caps,
and bound per-file and total content bytes. Deadline and observation-limit
failures therefore fail closed.

`IDEMPOTENT` and `EXTERNAL` declarations require the explicit
`allow_effectful_execution=True` opt-in. They execute once and return `observed`,
which records workspace changes but does not claim replay safety. Use a
domain-specific test for the downstream idempotency or reconciliation contract.
`cayu check` remains structural and never invokes this verifier or application
tools.

## Keep other controls separate

`ToolEffect` does not authorize execution. Authorization and approval belong in
tool, command, and network policy. Billing, budgets, quotas, and rate limits are
cost-governance controls. Taint tracks information flow. Events and telemetry
provide observability and audit evidence.

A `NONE` tool can still be expensive, sensitive, denied by policy, or heavily
audited. An `IDEMPOTENT` tool is not automatically authorized. Classify replay
semantics here, then configure those orthogonal controls explicitly.
