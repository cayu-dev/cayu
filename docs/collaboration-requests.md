# Durable collaboration request acceptance

The request owner retains questions and informational contributions. Acceptance
does not create a session, invoke an agent, deliver a message, or start background
work. Admission planning and authenticated answer publication are separate
capabilities.

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

## State, receipts, and recovery

This acceptance slice retains `open`, `cancelled`, and `expired` states.
Admission remains `undecided` until closed by a control. Delivery is pending,
then excluded when unstarted responsibility is closed. Neither receipt stage
claims that a recipient ran or read the contribution.

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
