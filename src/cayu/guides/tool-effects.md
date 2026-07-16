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

## Keep other controls separate

`ToolEffect` does not authorize execution. Authorization and approval belong in
tool, command, and network policy. Billing, budgets, quotas, and rate limits are
cost-governance controls. Taint tracks information flow. Events and telemetry
provide observability and audit evidence.

A `NONE` tool can still be expensive, sensitive, denied by policy, or heavily
audited. An `IDEMPOTENT` tool is not automatically authorized. Classify replay
semantics here, then configure those orthogonal controls explicitly.
