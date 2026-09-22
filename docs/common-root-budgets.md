# Common-root budget bindings

Common-root budget admission lets an application bind model and auxiliary
inference work to one trusted accounting authority. It is opt-in: configure a
`budget_binding_receiver` on `CayuApp` and set
`enable_common_root_budget_binding=True`.

The receiver is the authority boundary. It must register or resolve an
immutable `BudgetBinding`; caller-supplied IDs, digests, limits, and allowance
values are not authority. A binding is registered by `binding_id` and its full
authority digest. Repeating the exact authority is idempotent; changing any
authority field conflicts before reservation or provider dispatch.

`allowance` is a positive durable integer and means the maximum number of
atomically admitted model or auxiliary dispatches for that binding. Each
operation consumes one allowance unit in the same durable transaction as its
multi-ceiling reservation. Replay of the same operation identity is
idempotent; a new operation is refused once the allowance is exhausted.
Values are bounded by Cayu's durable signed-integer limit.

The binding's root and ancestor causal ceilings must each have reservations.
All ceilings are admitted atomically. Opaque external tool adapters are
refused before dispatch while strict common-root admission is enabled because
their downstream work cannot provide the required qualified accounting
evidence.

Memory, SQLite, and PostgreSQL budget ledgers support registration and
allowance consumption. SQLite and PostgreSQL persist both the registration and
operation-consumption identities, so restart and competing workers preserve
the same authority and exhaustion decisions.

Cancellation, timeout, publication failure, and recovery retain reservation
ownership until settlement or explicit cleanup. Cancellation is not treated as
proof that an opaque provider stopped; unresolved work remains fenced and does
not receive a fresh allowance through replay.

This capability is intended for trusted application/runtime receivers. It does
not make arbitrary tool or service adapters common-root qualified.
