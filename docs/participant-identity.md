# Participant identity administration

A participant is a durable logical identity, not an agent definition, session,
worker, or credential. Multiple participants may name the same application
configuration without becoming the same participant. An alias is a mutable
lookup name; retain the returned `ParticipantRef` for identity-sensitive work.

This API administers identities, lifecycle admission, namespace retention and
permit responsibility. It does not send collaboration requests, create sessions,
dispatch agents, or stop remote work. A retained permit is bookkeeping for a
qualified receiving owner, not a public execution grant.

## Initialize explicitly

Configure `CayuApp(collaboration_store=store, collaboration=registration)` and
then await `app.initialize_collaboration()`. Neither construction nor a failed
participant call implicitly provisions an owner. Other participant operations
raise `CollaborationNotInitialized` until initialization succeeds.

`CollaborationRegistration` holds a bootstrap binding, a trusted application
access policy, and a bounded tuple of supported `ParticipantConfiguration`
values. Each configuration contains versioned references for the agent
definition, routing policy, and admission policy. These are registered names,
not serialized executable callbacks or secrets.

The bootstrap binding includes application scope, provisioning scope, owner
name, contract version, and all capacity limits. Concurrent initialization with
the same binding returns the same durable owner and namespace incarnation.
Changing that binding for an existing application scope conflicts; it does not
silently replace the owner. Reopening a persistent store and initializing the
same binding reconstructs the same authority.

## Authenticate at the application boundary

An application constructs `CollaborationAccessContext` from its authenticated
principal. Do not construct it from model output or expose a raw principal
parameter as an agent tool. The context is not a bearer token and does not grant
access by itself: every operation consults `CollaborationAccessPolicy`.

The synchronous, side-effect-free policy returns a `CollaborationAccessGrant`
for the exact application scope and either the whole scope (`participants=None`)
or a tuple of exact participant references. An empty tuple permits no
participants. Creation and creation-receipt readback require whole-scope access;
configuration and alias changes require access to every affected participant.
Moving an alias requires both its old and new target.

Discovery and event pages are filtered by the current grant. Cursors bind the
scope, principal and exact selection, and cannot be reused after that selection
changes. They are pagination positions, not authority. A hidden alias returns
no binding. Events concerning multiple participants require access to all of
them. Scope initialization events are visible only to whole-scope grants.

Pages are not a frozen cross-page snapshot. Participant pages order immutable
IDs and read the current configuration on each page; concurrent insertion before
the cursor may require a fresh scan. Event pages advance stable sequences, so
new matching events after the cursor appear on later pages. Each page uses one
transactional view and rechecks the current access selection.
Event cursors additionally bind the retention revision. Pruning invalidates old
event cursors with `CollaborationHistoryUnavailable`; a fresh page explicitly
reports `history_complete=False` rather than presenting retained events as a
complete history.
The requested count is an upper bound: pages also fit the aggregate contract
limits, including their cursor. A shortened page continues after its last
returned record. If even one record plus its required cursor cannot fit, the
query reports unavailability rather than returning an empty continuation loop.

## Exact mutations and replay

Use `initialization.operation("application-chosen-key")` for a mutation's
operation reference. `create_participant`, `configure_participant`, and
`change_participant_alias` return immutable `ParticipantReceipt` values. Their
participant state, configuration history, alias mutation, receipt, event, and
capacity accounting commit in one transaction.

The exact operation tuple contains scope, namespace incarnation, generation,
caller key, kind, schema version, mode, source and destination owner, initiator
binding, receipt stage, complete request, and bootstrap limits. A matching key
with different intent conflicts. A receipt contains the historical result of
that operation; `inspect_participant` returns current participant state.

Configuration changes compare the participant's expected configuration revision.
Alias changes compare a scope-wide alias revision and the exact previous target.
Removing and recreating an alias does not reuse its earlier revision. Distinct
operation keys do not turn a stale revision into an authorized update.

Exact replay checks retained evidence before requiring a currently registered
configuration or new-mutation permission. It still requires current readback
authorization. `lookup_participant_operation(receipt.expected, context=...)`
offers explicit full-tuple readback. A missing or contradictory event or
configuration snapshot cannot be silently treated as successful replay.

## Ownership and capacity

Memory, SQLite and PostgreSQL implement the same typed `CollaborationStore`
contract. Memory provides in-process durability only. Persistent adapters use
the ordinary Cayu schema lifecycle and need schema revision 94. PostgreSQL
serializes transactions within each application scope; SQLite uses a native
write transaction. Distinct application scopes do not share operation keys.

Cancellation or an observation timeout does not prove a database operation
aborted. A dispatched mutation stays owned until it settles; exact local
retries join that work, and persistent transactions decide cross-instance
retries. Preliminary exact readback also uses retained, bounded observation:
a stalled transaction cannot leave a retry waiting indefinitely on its lock.
No more than 64 mutation and exact-readback tasks combined are retained per
store instance. Each owned operation's observation allowance is ten seconds.
Cancellation preserves sanitized failure evidence already available from the
completed operation; it does not wait indefinitely for future failures.
`store.close()` stops new owned operations and observes pending work for a bounded
interval; if it reports
`CollaborationUnavailable`, retain the store and retry close after settlement.
Do not dispose its connection pool while an operation is still owned.
SQLite shutdown retains one task through active transaction settlement and
physical connection closure. Timeout or cancellation stops only observation;
repeated close calls join pending shutdown, and a settled close failure may be
retried. New reads are refused once shutdown starts.

Capacity limits are explicit and immutable. Ordinary mutations cannot consume
reserved control operation, event, or byte capacity. The retained-byte limit
counts canonical document payloads, with one 64 KiB reservation for the mutable
anchor. It is not a physical database-file-size promise. Immutable receipt and
configuration history remain charged; replacing a current-state row counts
only its size delta. Exhaustion rejects a new mutation atomically, without
evicting exact replay evidence. New control operations also require capacity;
their reserved allowance is not an unlimited administrative history. A registered
permit reserves its settlement operation slot, event and bounded byte allowance
before admission, so later optional work cannot consume those resources.
Admission also checks the mandatory settlement receipt, snapshot, event and
receiving-readback envelopes using worst-case JSON expansion of receipt identifiers
and maximum generated counters. Aggregate reserved bytes do not replace these
individual record bounds.

All durable identity values pass bounded schema validation and the configured
workload-secret checks. Known secrets are rejected rather than redacted into a
different authority. Keep application-supplied identity/configuration names
secret-free.

See [the runnable identity example](../examples/collaboration/identity.py).

## Lifecycle admission and responsibility

`change_participant_lifecycle` compares the exact participant incarnation and
expected lifecycle revision. Active participants admit internal permits;
draining and disabled participants reject new permits while permitting settlement.
Re-enabling advances the admission generation. Replaying an old registration or
active-state receipt cannot revive excluded work. Retirement is terminal and
requires all outstanding obligations to be settled. An existing alias may be
removed after retirement, but cannot grant the retired identity new authority.

Disable captures the issued-permit frontier in the same transaction that closes
admission. The returned receipt proves acceptance, not remote quiescence.
`inspect_participant` separately reports current state, outstanding obligations
and settled/unsettled status. `list_participant_obligations` provides bounded
participant/position-ordered responsibility, pending-only by default. Its cursors
bind principal, scope, participant incarnation, filter and retention revision.
Settlements between pages can remove pending results; restart the scan when a
complete current inventory is needed. Pruning invalidates stale cursors.

Permit registration and settlement are internal store-owner entrances. Registration
serializes with lifecycle elections and records the complete source operation,
participant generation, destination owner/target, effect scope and settlement
requirement. Settlement queries a trusted receiving-owner reader outside the
transaction, then revalidates and commits exact positive receiving evidence inside
the transaction. An absent target, expired worker claim, or raw caller-shaped
receipt does not prove quiescence. Actual destination fencing adapters are not
provided by this API.

## Namespace maintenance and retained evidence

Initialization always replays the original bootstrap; obtain the current generation
with `inspect_collaboration_namespace`. Build new keys using that generation's
`NamespaceRef.operation(...)` after rotation, not the original initialization.
`seal_collaboration_namespace` closes new admission. Exact replay and already
admitted settlement remain allowed. `rotate_collaboration_namespace` atomically
seals the old generation and elects one successor.

`retire_collaboration_namespace` requires the later current control generation, exact
namespace revision, and the expected retired floor. It advances only the next
contiguous settled generation; an unresolved older generation cannot be skipped.
Retained receipts remain historical and readable after retirement.
The current control generation may be open or sealed for retirement and pruning.
This lets maintenance reclaim capacity before rotation even when the retained
generation limit is full. Sealing still rejects new participant mutations and
permit registration; maintenance does not reopen admission.

`prune_collaboration_namespace` removes at most the requested operation-record
count (1–32) from the next retired generation. Registration and settlement form an
indivisible two-record permit bundle; a batch reaching that bundle needs two
remaining slots. A batch too small to make any progress is rejected atomically.
Each batch uses a fresh operation key and exact retention revision; retrying the
same batch returns its immutable receipt. Events, indexes and unreferenced history
are reclaimed in the same transaction. Current participant history and history
referenced by another retained receipt remain available. Completed generations
compact to monotonic rejection floors; event sequences are never reused.

Exact lookup retains its four outcomes. Removed content returns unavailable, never
not-found authority to execute again. `inspect_collaboration_retirement` separately
returns retirement floors and `retained`, `partial`, or `pruned` content evidence
for the exact namespace reference (or `None` for an elected non-retired generation).
Unknown/future generations conflict. Retirement evidence is not an exact receipt.
There is no hidden maintenance worker: applications invoke bounded batches and
inspect their `complete`, `removed_records` and retention-frontier results.
