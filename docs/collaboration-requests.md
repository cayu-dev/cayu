# Durable collaboration request acceptance

The request owner retains questions and informational contributions. Acceptance
does not create a session, invoke an agent, deliver a message, or start background
work. Admission planning and authenticated answer publication are separate
capabilities.

When a request is answered from a retained session export, register a
`SessionExportRequestReceivingOwner` as the receiving owner. Its acceptance
reader must implement both exact operations: `lookup(receipt)` authenticates
the source export, and `settlement(receipt, expected_permit)` returns a
`ReceivingSettlementReceipt` issued by the receiving boundary. Acceptance is
not converted into settlement locally, and a missing settlement result fails
closed. The request coordinator holds receiving authorization through store
publication, so a lost acknowledgement can replay the answer without
rerunning the export.

## Registration and authority

Configure the existing collaboration store and participant registration, plus
`collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=...)`.
The resolver implements the existing held `MandateResolver` contract. Optional
resource-selector owners validate canonical resource references. The maximum
request lifetime is explicit and finite.

The participant access policy recognizes `request_accept`,
`request_readback`, and `request_control`. New acceptance requires current
consultation authority for the exact initiating participant. Readback and
administration remain separate permissions. Original initiating identity and
the accepted mandate/configuration evidence are retained in the receipt;
they are not replaced by the identity of a later authorized reader.
Every observer enters its own access and mandate checks, including concurrent
identical calls. Atomic store replay prevents duplicate acceptance; sharing a
store does not share another application's authorization.
Participant-scoped read permission is checked against both submitted selections
and the retained operation's actual participants before reporting exact conflicts.
An unauthorized caller receives access denial, not an existence/conflict result.
Fresh acceptance returns the same access denial for missing and inaccessible
aliases, without loading an inaccessible recipient's snapshot.

## Owner-level example

This function uses an application with initialized participants and registered
authority. Its arguments are exact participant references, pinned contract
references, and an application-authenticated mandate context:

```python
from cayu.collaboration import CollaborationRequest, RequestControl


async def retain_review(
    app, *, sender, recipient, context,
    delivery_contract, output_contract, independence_policy, disclosure_policy,
):
    initialized = await app.initialize_collaboration()
    request = CollaborationRequest(
        operation=initialized.operation("review-request"),
        kind="question",
        sender=sender,
        target=recipient,
        content="Review the selected material.",
        inputs=(),
        context_hint=None,
        delivery_contract=delivery_contract,
        output_contract=output_contract,
        independence_policy=independence_policy,
        disclosure_policy=disclosure_policy,
        ttl_ms=60_000,
        cancellation="detach",
    )
    receipt = await app.accept_collaboration_request(request, context=context)
    snapshot = await app.inspect_collaboration_request(
        receipt.expected, context=context
    )
    return receipt, snapshot
```

An informational contribution uses `kind="contribution"` and
`output_contract=None`. It does not require a fabricated answer.

For cancellation, provide a distinct operation key in the accepted namespace
generation and the exact accepted command:

```python
control = RequestControl(
    operation=receipt.expected.operation.model_copy(
        update={"caller_key": "cancel-review-request"}
    ),
    expected=receipt.expected,
    expected_revision=snapshot.revision,
    kind="cancel",
)
terminal = await app.control_collaboration_request(control, context=context)
```

The authority resolver must authorize a new control. Replaying its exact committed
receipt requires current read permission, not a new administration grant.

For an admitted session-export request, admission and terminal commands carry
the exact source export receipt. This is required for continue, fork, fresh,
answer, cancellation, and failure paths; a source reference without its
authenticated receipt is not enough to authorize or settle the request.

## State, receipts, and recovery

The request lifecycle retains `open`, `answered`, `failed`, `declined`,
`cancelled`, and `expired` states. While a request is open, its admission has
an independent state machine: it begins `undecided`, may enter `planning` or
`preparing`, and can become `deferred`, `clarifying`, `admitted`, or `closed`
through the corresponding authenticated admission decision and terminal
commands. Admission state is not the same as the request's terminal outcome.
Delivery is pending, then excluded when unstarted responsibility is closed.
Neither receipt stage claims that a recipient ran or read the contribution.

Acceptance atomically records the resolved participants, original addressing
intent, owner-time deadline, permit, receipt, event, and mandatory control
reservation. Disabling a recipient prevents fresh acceptance without erasing
already accepted responsibility. Replay does not resolve an alias again.

Mandatory controls have an explicit 8 KiB canonical-JSON limit on their complete
initiating identity (`RequestControlCommand.initiator`), in addition to individual
identifier and 64 KiB contract limits. The bound includes JSON escaping. An
otherwise authorized control exceeding it is rejected without mutation; use an
authorized identity within this bound. Acceptance reserves the serialized room
for any control within this limit, a maximum escaped operation key, and either
cancellation or expiry. Capacity reservations do not relax individual record
limits.

At the deadline, expiry wins over a new cancellation classification. A premature
`expire` control is rejected. Exact terminal replay preserves the prior election.

`lookup_collaboration_request` accepts the complete acceptance or control command and returns
the shared match/conflict/not-found/unavailable alternatives. Access denial and
retired namespaces are separate outcomes. The immutable acceptance receipt and
the current request snapshot are intentionally different values.

An authorized scope-wide maintainer can call
`list_due_collaboration_requests`. Pages are bounded by count and bytes and
contain pending admission responsibilities. Their cursors are positions in current
due inspection, not snapshot guarantees, observation registrations, or work
claims. Reading a page does not execute or settle its records.

Cancellation or timeout can stop observing a mutation while it remains owned.
Use exact readback rather than assuming that the operation was aborted. During
shutdown, call `drain_collaboration_requests()` before closing the collaboration
store. Draining closes new operations on that shared collaboration owner and
waits boundedly for retained operations; an unavailable result means draining
must be observed again.

Observation reads reconcile against the current owner event frontier in the
same store transaction as readback. Publication before registration, during
registration, or after a previous read is returned at the next current
frontier; the registration event itself is metadata and is not reported as
request work. Missing frontier evidence returns unavailable rather than
silently reporting a complete page.

## Finite collaboration waits

`CollaborationWait` registers a bounded, finite election over exact request
commands. It supports `ALL_SUCCESS`, `ALL_SETTLED`, `ANY_SUCCESS`, and
`QUORUM_SUCCESS`; registration stores the predicate, deadline, source pins,
and retention responsibility without starting a model, tool, session, or
background worker.

Use `register_collaboration_wait()` once, then
`observe_collaboration_wait()` to catch up the authenticated request-source
frontiers and record evidence. The first qualifying evidence manifest is
durable and replayable. `inspect_collaboration_wait()` is a read-only exact
readback; `lookup_collaboration_wait()` returns the shared exact
match/conflict/not-found/unavailable registration result. `cancel_collaboration_wait()` records cancellation or an
owner-time expiry without changing the target requests.

An external observer can use these operations without creating a session. A
session-bound wait carries an exact continuation ticket. After election,
`deliver_collaboration_wait()` passes the authenticated predicate/result
binding to a `CollaborationWaitLatchReceiver` registered with the destination;
cancellation and expiry keep
their source responsibility pending until an explicit authenticated exclusion
receipt is recorded. Acknowledgement loss is reconciled by repeating the same
exact operation, never by rerunning a request producer.
