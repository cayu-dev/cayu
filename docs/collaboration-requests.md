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

### Clarification source references

For a previously published text export, `app.inspect_clarification_source()`
resolves a `ClarificationSource` using the registered session-export owner:

```python
source = await app.inspect_clarification_source(
    export_request,
    sender=question_author,
    audience=permitted_responder,
    context=export_context,
)
```

This is read-only. It checks the source session incarnation and immutable
participant binding, exact export selection and current disclosure permission.
The result contains identities and commitments, not the text or private provider
state. Supplying `expected=source` checks the complete previous selection on a
later read. A mismatch conflicts; revoked disclosure is denied even when the
historical source reference matches. The source reference is not a transferable
authorization: later publication, peer delivery and provider exposure must still
use their current authorization boundaries.

`RequestRegistration.clarification_policies` is an explicitly configured tuple
of `ClarificationPolicy` values (at most 32). Each policy has a unique versioned
reference and finite ceilings. The registration is defensively copied; a caller
cannot change its ceilings by changing a supplied policy object. An empty tuple
does not supply an implicit clarification policy. Registering a policy alone
does not execute or admit a service turn.

### Opening an exact clarification question

`app.open_clarification(command, source=export_request, context=export_context)`
accepts a `ClarificationOpenCommand` for a request already admitted with the
`clarify` decision. The command freezes the parent request and admission, current
effective-input revision, registered receiver and finite policy, question key,
responder, lineage, source commitment, budget identity and deadline. The source
must be an export from a session bound to the selected request recipient, with
the original sender as its audience. The export context must carry the current
recipient mandate authenticated by the registered resolver.

The coordinator holds the existing export/mandate guard through the request
transaction. It does not reacquire the same resolver recursively. The native
collaboration transaction compares the exact parent frontier, participant
incarnations and lifecycle/configuration generations, deadline and capacity.
Fixed-key changes conflict without publishing another question. If disablement
wins before publication, new publication is denied. An earlier committed question
remains historical evidence; replay does not launch work.

Question publication requires common-root budgeting to be enabled and resolves
the binding through the application's registered budget receiver. Its reference
has kind `budget_binding`, the collaboration owner, binding ID as `object_id`,
the complete binding authority digest as `incarnation`, and revision `1`.
`budget_authority_sha256` is that same complete digest. Root and ancestor
ceilings must be reserved, priced, all-time USD limits; the root ceiling cannot
exceed the registered clarification policy's cumulative-spend ceiling. These
checks do not create a second balance or replace dispatch-time ledger admission.

Opening is distinct from peer delivery, exposure, temporary service, reply
acceptance and final-result continuation. It does not append transcript content,
invoke a provider or consume the original wait's final latch. Current export
authorization is required even when calling the opening entrance with a previously
committed command; an old receipt cannot renew disclosure permission.

### Explicit clarification delivery

`app.deliver_clarification(intent, context=export_context)` accepts a complete
`ClarificationDeliveryIntent`: the question (and accepted reply when applicable),
operation and initiator, exact source export request, sender and recipient, and
the existing `PeerContentAppendRequest`. The target must be an existing exact
participant-session incarnation. Delivery uses `wake_policy="none"`; it does not
start temporary service or consume the original final-result wait.

The registered export owner revalidates the selected text and current mandate.
The collaboration transaction reserves pending responsibility and settlement
capacity before calling the peer API. The peer API separately authenticates the
producing occurrence through the registered export policy and owns queue admission.
The two source guards are acquired sequentially, not recursively; revocation
between them prevents append and does not erase the retained responsibility.

Retries preserve the complete intent, including attempt deadline and replacement
identity. Exact receiving readback precedes another append attempt. Cancellation,
timeout or a lost acknowledgement does not prove exclusion or release capacity.
Only receiving-store append/exclusion evidence settles delivery. Queue acceptance,
transcript delivery and provider exposure remain separate facts.

The returned `ClarificationDeliveryReceipt` contains operation/question identity,
kind, status, and queue or exclusion metadata, not the source text, private
authority or complete receiving request. Source authority is checked even on a
delivery retry; a historical export receipt is not a new disclosure grant.

### Explicit temporary clarification service

`app.prepare_clarification_delivery(intent, context=export_context)` registers the
same exact durable responsibility used by `deliver_clarification`, but does not
attempt peer append or dispatch execution. Repeating preparation is exact replay;
later delivery uses the same intent and reacquires current disclosure authority.
Preparation alone is neither append, exclusion, nor permission to execute.

`app.service_clarification(request, context=export_context)` accepts a
`ClarificationServiceRequest` containing the service operation and initiator,
complete question delivery, original continuation ticket, service generation,
optional parent service, and an explicit host `instruction`. The nonblank
instruction is ordinary user input, bounded to 16 KiB of UTF-8, and is part of
the exact retry identity. It is not a replacement for authenticated peer content.
The runtime does not synthesize a tool result or copy the question into user input.

The receiving session must already exist and match the delivery's participant and
session incarnations. The question must have a settled authenticated append, or
a prepared delivery supplied with a separate current sender `delivery_context`.
For prepared delivery, the runtime admits the exact invocation first, then invokes
the existing delivery owner before provider work. A responder context cannot stand
in for the sender. The immutable service record retains the complete required peer
append; dispatch checks reconcile that exact native receipt, including after
restart. An in-process callback or a caller-shaped success response cannot replace
the receiving evidence.
The service receipt's `released_session_status` is present only after return and
records the session status in that invocation's native release receipt. It is
historical evidence, not the session's current status. Return proves discharge of
the service responsibility; it does not by itself mean that execution completed
successfully. In particular, a returned failed or interrupted service is not an
accepted clarification reply and does not automatically retry.
The responder needs current source-row read/exposure and consultation authority;
an earlier delivery receipt does not grant that access. The coordinator resolves
the actual session binding and execution profile, then uses the existing durable
participant permit and native temporary-service admission. The registration guard
is owned through the actual registration transaction and ends before provider
serialization obtains its independent disclosure guard.

When multiple peer occurrences remain in the receiving history, the registered
export policy must implement `acquire_peer_exposures` for the complete batch.
Authenticate each occurrence under one shared revocation guard held through
serialization; do not recursively acquire a non-reentrant single-item guard.
The existing default supports only a single occurrence and rejects larger
batches rather than silently skipping authorization.

Peer append can admit inert content to a released clarification wait only after
the receiving transaction authenticates its indexed native wait and writer-release
evidence. This exception does not reopen ordinary completed/interrupted sessions,
enable automatic wakeup, or grant execution permission. A final latch and temporary
service still compete through their existing receiving owner.

Native service admission rejects both expired questions and expired original
waits. Active service is bounded by the earliest of those deadlines and its finite
`service_timeout_ms`. This timer cancels owned runtime execution; it neither changes
the parent session's lifetime deadline nor proves external work has stopped.
Cancellation settlement uses the ordinary runtime cleanup owners. An unresolved
admission stays fenced and must be reconciled rather than dispatched again.

The returned `ClarificationServiceReceipt` contains bounded identities and state,
not private permit material. `returned` means native execution responsibility has
returned; it is not proof that a reply was exported, accepted, or that the original
request completed. Pending observation or acknowledgement loss requires readback
with the same complete selection; never mint a replacement operation merely
because the caller stopped waiting. Reply acceptance and final-result continuation
remain separate operations.

`app.reconcile_clarification_service(request, context=collaboration_context)` is
an explicit maintenance entrance for scope-wide `request_control` authority.
It requires the same complete service selection and authenticates native
receiving records and the existing permit owner. It can settle a proven return
after source disclosure is revoked without reading the exported text or launching
new work. A prepared or still-active record remains pending; absence, interruption
and timeout do not prove exclusion. This operation does not replace fresh
authorization on `service_clarification` or reply publication.

Native return or exclusion does not by itself permit erasing the service's
receiving evidence. The source session retains a deletion fence until its owner
records acknowledgement of both collaboration permit settlement and lineage
settlement. A failure between those writes is reconciled with the same service
identity, including after the original final-result wait has completed.
If namespace maintenance has already pruned the foreign receipts, maintenance
requires both the exact retained native terminal receipt and positive retirement
evidence from that same collaboration namespace incarnation and generation.
Retirement certifies settlement before pruning; absence alone is not sufficient.
This recovery path never grants execution or recreates a service permit.

An authorized scope-wide `request_readback` maintainer can discover registered
CollaborationStore service debt with `list_pending_clarification_services(context=..., cursor=None,
limit=32)`. Pages have a hard limit of 64 items and the shared bounded JSON
envelope. The cursor orders by deadline and exact operation, not a snapshot or
work claim. Expired services remain discoverable until positively settled.
Start a later sweep without a cursor to include newly inserted responsibilities.

Each item carries a `ClarificationServiceRecovery` selector containing the
original operation, waiting-session incarnation, public selection commitment and
complete dispatch commitment. It contains no host instruction, peer payload or
permit. After a restart, pass this selector to `reconcile_clarification_service`
instead of rebuilding the original request. Reconciliation compares the selector
against the full immutable native dispatch and registered collaboration permit;
its hashes are expected identities, not authorization. Changed commitments or
incarnations conflict. This selector is never accepted by the execution entrance.

`inspect_clarification_services(original_ticket, context=..., cursor=None, limit=32)`
reads the existing SessionStore ticket's bounded service index. This includes
native preparations that have not yet registered a CollaborationStore permit,
and historical terminal services. It verifies the immutable ticket, indexed
service commitment and exact native readback before returning status and recovery
selectors. It does not disclose peer content or private admission material.

An administrator can explicitly call `exclude_clarification_service(selection,
context=...)` for unadmitted preparation. The existing receiving owner atomically
arbitrates exclusion against admission and then settles the existing permit
responsibility. In side-session mode the original wait's reservation and target
preparation are created together in one native SessionStore transaction, before
the separate CollaborationStore permit registration. Both capacities and session
incarnations are checked before either reservation is written. Exclusion compares
the exact retained target preparation; delayed admission cannot replace that
exclusion. Already-admitted work cannot use this operation. Ordinary
reconciliation never implicitly excludes a preparation merely because it timed out
or its observer disappeared. Both maintenance operations require scope-wide
existing permissions; historical evidence does not grant new execution authority.

Delivery maintenance uses `list_pending_clarification_deliveries(context=...,
cursor=None, limit=32)` and `reconcile_clarification_delivery(recovery,
context=...)`. Discovery returns at most 64 bounded entries per page, including
expired unresolved attempts. Each selector binds the original operation and the
complete intent commitment; it contains no source payload or private append key.
Scope-wide `request_readback` is required for discovery and `request_control` for
reconciliation. Exact native append/exclusion readback can settle existing debt
after restart without the caller retaining its original payload-bearing intent.
Missing or pending receiving evidence remains pending, never excluded by inference.
These maintenance calls neither append nor expose content: actual delivery and
provider exposure still require their existing current export-owner authorization.
Historical evidence and a recovery selector do not grant disclosure or execution.

`exclude_clarification_delivery(recovery, context=...)` additionally obtains an
exact native exclusion fence, even if disclosure was permanently revoked before
any peer attempt existed. It requires scope-wide `request_control` and independent
registered `SessionExportPolicy.acquire_peer_exclusion` approval of the owner-read
`ClarificationDeliveryRecord`. The default denies. A competing append that already
won remains appended, never reclassified as excluded. Lost acknowledgement keeps
responsibility pending until exact native readback settles it; replay cannot append.

Every new `RequestAdmissionCommand` supplies `expected_input_revision` and
`expected_input_sha256`. Revision zero commits the original accepted request
command; later revisions use the current clarification input commitment. The
owner compares both atomically, even when the ordinary request revision has not
changed. Already-committed admissions replay their original complete command.

Active temporary service samples time from the registered collaboration owner
before entering runtime execution and subtracts the complete clock-read round
trip. Its monotonic duration is bounded by the service policy, question deadline,
and original wait deadline; worker UTC offsets cannot extend those bounds. Clock
read failure prevents execution, not responsibility retention or reconciliation.

Question expiry uses `list_due_clarification_questions(context=..., cursor=None,
limit=32)` and `expire_clarification_question(request, context=...)`. Discovery
uses native owner time and returns bounded, content-free question identities and
original question/request commitments. The limit is 1–64. Start each later sweep
without a cursor: a previously ineligible question can become due behind an old
cursor. Discovery requires scope-wide `request_readback`; expiry requires
scope-wide `request_control`. A `ClarificationExpiryRequest` supplies a distinct
stable operation key in the question's namespace and the exact discovery
selector. Preserve that request across cancellation or lost acknowledgement;
the same public entrance reconciles exact committed expiry after reconstruction.
Expiry before the native deadline is rejected. A competing reply or other
terminal decision is not overwritten. This settles only the question decision
and its reserved capacity, never a pending service, peer delivery, or provider
effect. Those responsibilities retain their separate recovery entrances.

Retired namespace pruning reclaims terminal request and clarification history in
bounded batches. A durable private cursor binds both histories, so reconstruction
does not restart deletion or mistake partial history for a new request.
Ordinary request readback is unavailable while that history is being reclaimed.
Pending clarification service/delivery responsibility prevents reclamation.
Shared lineage counters remain unchanged while another retained question uses
them; the final lineage record can be removed only after its last question and
all pending responsibility are gone. Pruning does not release foreign source
exports or native session resources by inference.

### Accepting a clarification reply

`app.reply_to_clarification(request, context=export_context)` accepts a
`ClarificationReplyRequest` containing the exact service selection, original
request, expected effective-input revision and commitment, a source-owned
`assistant_visible_text_v1` export, and the producing model stage ID. The stage ID
is a selector, not authority: the receiving coordinator checks native service
return, provider dispatch, and committed model-publication evidence. Exported
rows must belong to that exact service invocation and publication, and their
visible-text commitment must match the authenticated export. Private provider
state and thinking are not reply material.

Current responder/source disclosure authority is required on every call,
including replay. The request transaction elects one reply and appends an input
revision without rewriting the original request. `ClarificationReplyAcceptance`
returns bounded election evidence; it does not dispatch a provider, deliver the
reply to another session, or complete the original request.

Final-result publication remains a separate request outcome. When using the
registered session-export receiver, the final export must match the original
request's initiating identity, pinned output projector and disclosure policy.
An explicitly authorized requester may export the responder's actual completed
record; that operation preserves the producing session, incarnation and source
commitment. A responder's unrelated export receipt cannot simply be relabeled.
Observe and deliver the original wait only after the request owner has elected
its terminal outcome. Clarification reply acceptance alone does not satisfy that
final wait or authorize its continuation.

### Question inspection and cleanup

`app.lookup_clarification(expected, context=mandate_context)` accepts the complete
opening or closing command and returns the shared exact lookup result: match,
not-found, conflict or unavailable. A match contains that operation's original
receipt. `app.inspect_clarification(opening_command, context=mandate_context)`
uses the same comparison and current read authorization, but a match contains
the current `ClarificationQuestionState`. Thus an opening receipt can remain
unchanged while inspection reports that its question is now closed.

These entrances require current request readback authority, including access to
the participants in retained evidence. They do not disclose the exported text or
renew publication/execution authority. Historical metadata remains inspectable
without the old publication policy or receiving adapter installed. Access denial
remains an exception, not not-found; missing receipt/state/history evidence is
unavailable rather than proof of absence.

`app.close_clarification(command, context=mandate_context)` requires a complete
`ClarificationCloseCommand` and current request administration authority. Its
initiator must match the authenticated caller. Expiry, supersession and parent-
terminal dispositions require the corresponding owner-store evidence; cancellation
is an explicit semantic decision. Closing is exact and replayable, including
after participant disablement or source-disclosure revocation when the administrator
still holds cleanup authority. It does not complete the original request, consume
its final wait, cancel dispatched work or settle any service/delivery obligation.

### Retaining a review request

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

An authenticated host can explicitly stop a participant-owned root execution at
its completed assistant/tool-turn boundary with
`execute_participant_session_to_wait(execution, wait, participant=..., context=...,
wait_context=...)`. Creation remains inert; the execution uses the existing
participant permit, and the complete wait is part of its exact execution identity.
No scheduler or model-facing orchestration is implied.

If that execution releases its writer before parking the prepared wait, or the
original wait is cancelled/expired after parking and temporary service, the host
can explicitly call `exclude_participant_session_wait()` with the same arguments.
For a service-bearing wait, every service must have exact terminal receiving
evidence and acknowledged foreign settlement. Native return alone is insufficient.
The retirement transaction compares the complete settled-service commitments and
current released writer frontier; a new service or final-continuation claim fences
cleanup. Cleanup grants no execution and creates no replacement wait.
Cleanup requires current scope-wide request-control and participant-administration
permission, and authenticates the complete original execution/wait commitments
and native permit consumption and admission/release receipts. The original
`wait_context` remains exact selection data, not renewed disclosure authority:
its mandate may have expired or been revoked. Cleanup returns a bounded,
content-free `ParticipantSessionWaitExclusionReceipt`, not historical request
content or a new execution grant. Terminal status or cancellation alone is not
evidence of release. It does not
dispatch another model request. A released retirement remains a deletion fence
until the registered wait receiver reads back its exact durable exclusion and the
session owner records acknowledgement. A failed or lost acknowledgement is retried
with the same execution and wait identities, not a replacement execution key.
If interruption happened before the foreign wait registration committed, cleanup
records an exact cancelled (or expired) registration under the same operation key
before the native exclusion handshake. Late registration cannot reopen that wait;
cleanup does not require the original deadline still to be in the future.
Sealing or rotating a still-retained namespace does not prevent this exact
cancellation handoff. Retired or pruned generations cannot acquire a new wait
cleanup responsibility. If foreign history has already been pruned, positive
namespace-retirement evidence and the exact native released-retirement receipt
finish the acknowledgement without recreating foreign records. Repeating the
same cleanup returns the same compact receipt before and after pruning.
