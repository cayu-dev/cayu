# Exact immutable input resources

`LocalArtifactResourceOwner` acquires an exact, revision-pinned artifact or a bounded
`FolderInputManifest`, records the complete command before reading, and keeps
the artifact-store pin until an owner-issued receipt is released. A receipt is
not a capability by itself: readback, transfer, and release revalidate the
durable command and the current material commitments.

```python
from cayu.artifacts import (
    LocalArtifactResourceOwner,
    LocalArtifactStore,
    MandateResourcePreparationReader,
    ResourceAcquisitionCommand,
)

store = LocalArtifactStore(".cayu/artifacts")
# Host registration supplies these values, never request payloads. The exact
# command/permit pair comes from the application's trusted participant receiver.
registered_resource_receiver = MandateResourcePreparationReader(
    owner=registered_selector_owner,
    resolver=registered_mandate_resolver,
    context=authenticated_context,
    redactor=redactor,
    registration=receiver_ref,
    policy=pinned_policy,
    artifact_store=store,
    collaboration_store=collaboration_store,
    initialized=collaboration_initialization,
    responsibilities=((command, preparation_permit),),
)
owner = LocalArtifactResourceOwner(
    ".cayu/resource-owner",
    owner=resource_owner,
    artifact_store=store,
    preparation_reader=registered_resource_receiver,
)
# `preparation_permit` must come from the registered collaboration receiver;
# a caller-shaped PermitCommand is not authorization.
preparation = await owner.authorize(command, permit=preparation_permit)
receipt = await owner.acquire(command, preparation=preparation)
try:
    # Use the exact receipt only while its command remains authorized.
    await use_immutable_input(receipt)
finally:
    await owner.release(receipt)
```

Resolve artifact selectors with
`LocalArtifactResourceOwner.artifact_selector(store, resource_owner, artifact_id)` before
registering the exact command with the trusted receiver. Local artifacts have
one immutable revision (`1`); their incarnation is the digest of store-observed
immutable publication metadata, including its creation timestamp. Arbitrary
revision/incarnation labels are rejected. Republishing a deleted artifact ID
does not satisfy the old selector. Acquisition rechecks the identity after
pinning, closing the selector-to-pin replacement window. Folder members use
these same artifact identities.
Synchronous mandate canonicalization validates the exact identity format, not
live artifact existence. Acquisition observes live publication metadata in its
retained asynchronous worker before dispatch; blocked filesystem observation
does not block the event loop or release the operation fence on caller timeout
or cancellation. The synchronous host discovery/manifest-registration APIs
should be run off the event loop when filesystem latency matters.
Owner construction is also synchronous host setup. Reopening binds the journal
to the artifact store ID, resolved root, and physical root identity; a same-ID
replacement store cannot settle the original store's cleanup responsibilities.
Async owner operations perform journal locking, reads, writes, and synchronization
in retained filesystem workers. A timeout or cancellation stops observation, not
an in-flight transaction or its mutation fence. Unchanged read transactions do
not rewrite the journal.
Mutation-fence setup and teardown also run in a retained worker, including their
filesystem metadata operations. Cancellation or timeout before fence acquisition
prevents the delayed invocation from starting resource work. Once work has begun,
the fence remains owned through settlement and lock teardown.
Graceful event-loop shutdown also retains dispatched pin and unpin work until
the actual filesystem operation settles. Aborted journal transactions preserve
lock-cleanup failures alongside the original rejection; a clean abort adds no error.

Exact-resource pins occupy a separate internal store namespace. Generic
`pin()` and `release_pin()` cannot remove them, even with the complete operation
digest or destination pin label. Labels also include the receiver identity, so
distinct owners cannot release one another's retention using the same operation key.
Normal and session-closure deletion both
honor this namespace; only resource-owned cleanup removes its retention.

The command must carry an exact `ResourceSelector`, a pinned policy, explicit
material and byte bounds, and the cleanup owner. `authorize()` also records a
validated `PermitCommand` in the owner journal; passing a matching digest
or a caller-shaped receipt without that durable authorization is rejected.
The concrete mandate receiver compares the complete command and permit against
its trusted registration, including the participant, lifecycle/admission
generations, settlement key, limits and target state. Before pin dispatch it
registers that exact obligation with the collaboration store. Registration
failure or a lost acknowledgement retains the original operation for recovery.
Cleanup publishes local quiescence before collaboration settlement; a failed
settlement remains capacity-counted and is retried by `reconcile()`.
Acquisitions and transfers share one event budget. Admission reserves seven
stage-event slots before dispatch, including cleanup and responsibility settlement.
Each operation stage has one immutable event; retry diagnostics update the
retained snapshot without spending more stage slots. Unused slots are released
only after responsibility settlement, so retries cannot consume another
operation's mandatory cleanup capacity.
Every journal publication also preserves a worst-case byte and JSON-node
envelope for each unsettled operation's receipts, diagnostics and remaining
events. Pending admission reserves this space before responsibility registration
or pin dispatch; unrelated authorization and manifest writes cannot consume it.
Cleanup evidence separately proves quiescence and permanent admission exclusion
from the released operation tombstone. A missing registration acknowledgement
does not downgrade quiescence to exclusion-only evidence. The collaboration
transaction either settles the exact registered obligation or durably fences a
registration that never committed, including quiescence-required permits.

Leases retain the complete resolver/receiver/policy/context/store registration,
as well as identity-bound revocation generations for every mandate-chain entry.
Replacing a resolver at the same revision does not renew existing leases.
The command policy must match the registered pinned policy.

Public release requires the registered resolver's current `release` action;
neither an acquisition lease nor possession of a serialized receipt grants that
action. Owner-internal reconciliation of already-retained cleanup obligations
does not require renewed acquisition authority.
Every acquisition must include `release` in its allowed operations. Revocation
alone does not retire a successfully owned resource: its retention remains live
until an authorized release or authenticated handoff establishes cleanup work.
Public release always compares against the original durable acquisition receipt,
including after interrupted or completed cleanup. Callers retry that original
`owned` receipt; they cannot choose a release stage or substitute its commitments.
Internal recovery reads the preserved original receipt separately from mutable
release progress.
Release and release recovery validate the exact durable command and original
receipt without requiring the old material to still exist. Once unpinning has
succeeded, deletion or replacement under the same artifact ID cannot prevent
the old responsibility from settling. Public release still requires current
cleanup authority; recovery cannot remove another owner's retention.

Custom `ResourcePreparationReader` implementations must implement a
`revalidation_guard(command, lease)` which validates the complete lease and
serializes revocation until guard exit. `revalidate()` is only a point-in-time
observation; it cannot authorize a subsequent effect. The owner holds the guard
while synchronously submitting each pin to the filesystem executor and while
dispatching a prepared final-receipt commit to its retained journal worker.
It does not hold the resolver guard while waiting
for dispatched filesystem work to settle. Revocation can prevent the next
member or publication without blocking on a stalled pin. Already-dispatched
work remains fenced by the resource owner until it settles, including if guard
exit fails. Internal local pin adapters must forward the supplied `dispatch`
callback; submitting work outside it bypasses preparation authority.

Custom `ResourcePreparationReader` implementations must also implement current
release authorization, durable responsibility registration, and authenticated
settlement in addition to lease acquisition and revalidation. `transfer_permit()`
resolves a destination permit from trusted registration; lease acquisition and
revalidation accept both acquisition and transfer commands. No-op
implementations are test doubles, not qualified production receivers.
Operation identity is the
scoped `OperationRef`; reusing it with any different decision-bearing field
returns a conflict rather than allocating a second resource. A cancelled or
timed-out foreground call does not release a dispatched read. The owner keeps
the pending/uncertain operation fenced and `reconcile()` can finish it after a
late settlement or process restart.
Preparation also carries an invocation-local stop flag. Cancellation or timeout
prevents a late resolver/registration result from dispatching another pin or
publishing a newly usable acquisition/transfer receipt. Already-dispatched work
is not cancelled: after it settles, partial preparation moves through durable
cleanup and responsibility settlement. Failed cleanup remains fenced by its
durable state and can be reconciled without renewing acquisition authority.
Recovery applies the same stop checks when it would start new acquisition;
mandatory cleanup is allowed to finish. A later `drain()` reports the retained
failure or stopped-preparation unavailability, not a replay of the original
caller cancellation. Worker failures are carried as typed outcomes until
explicit observation, preventing abandoned shield futures from logging raw
exception text.

Folders are immutable manifests of artifact members. The owner verifies every
member's object identity, metadata/content commitment, size, aggregate byte
bound, and manifest hash before publishing one complete receipt. A failure
after an earlier member was pinned retains those pins until reconciliation or
explicit cleanup; it never reports a complete folder.
Deterministic member-count, aggregate-size and manifest-size bounds are checked
before preparation or responsibility registration. A permanently over-limit
operation already retained by an interrupted path is cleaned and settled during
reconciliation, rather than retried indefinitely.

Transfer is a two-owner protocol. The destination durably records its pending
request and pins every member, then publishes an accepted transfer receipt.
The destination mandate receiver must register the exact `(transfer, permit)`
pair in its `transfers` configuration. Its current `prepare` grant covers the
source's exact selector; its pinned policy matches the source acquisition's
policy. The permit records the destination participant, source transfer operation,
`transfer` effect scope, and destination-owned `resource_transfer` target whose
object ID is `resource_operation_digest(transfer)`, incarnation is the destination
incarnation, and revision is the acceptance generation. This target identifies
responsibility, not a grant derived from the source receipt.
Before any destination pin, the owner persists the exact preparation lease and
registers responsibility. It revalidates before each member and acceptance
publication. Partial transfers retain their pins and capacity until owner-internal
cleanup and collaboration settlement; definite mandate denial triggers cleanup
without waiting for lease expiry. Reopening uses the same registered transfer
tuple and operation identity, never a new allocation key.
Only after the destination receipt is read back by the registered destination
owner may the source release its pin. S3, Docker mounts, mutable workspaces,
and transformed or unqualified artifact-store adapters are not advertised by
this capability; they must fail closed instead of being treated as local
durable ownership.
Public `read_transfer()` requires current registered preparation authority and
a currently accepted durable receipt. Pending, uncertain, releasing and released
transfers return `ExactUnavailable`, not acceptance evidence.
Exact private settlement readback is separate and
used only for source cleanup under both owner mutation fences; revoked authority
does not prevent cleanup of an already-accepted handoff.
Specifically, destination `reconcile(source_owners=(source,))` authenticates its
durable accepted transfer and completes source cleanup under both owner fences,
without renewing the source release mandate. The destination pin remains live.
