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

### Prepared recipient admission

`RequestRegistration.prepared_admission=PreparedAdmissionRegistration(receiver=...)`
opts into the production native recipient receiver. Its `ObjectRef` must carry
an explicit revision. Register the ordinary mandate resolver and enable a trusted
common-root budget binding receiver as well. No permissive receiver is installed
by default. An optional existing `receiving_owner` continues to handle export-backed
operations; it does not authenticate the prepared branch.

Before acquiring material or creating a FRESH child, an explicit driver can call
`prepare_recipient_creation(creation, context=...)` to obtain a
`FreshRecipientPreparation`. It freezes the original bounded request, exact native
creation target, resolved profile, historical definition commitment and complete
sponsor binding without creating a session, acquiring a permit or registering
budget-ledger capacity. The requested session ID may be `None`; no future session
incarnation is invented. This proposal is data, not execution or creation authority.
Initial attachment references may be frozen in that input, but do not prove
retention or access. A planning proposal containing attachments requires explicit
resource recipes. Native creation checks complete transfer coverage and exact
attachment metadata against the qualified static artifact environment before
committing the child; preflight alone cannot bypass those checks.

The existing registered budget receiver receives a preparation request with
`kind="request_recipient_preparation"`, `schema_version=1`, `creation_target` and
`execution_profile_fingerprint`. It must explicitly authenticate this future
operation; it must not treat a requested public ID as a created incarnation.
Passing the proposal as `create_recipient_session(..., preparation=proposal)`
compares current preflight, definition, sponsor and the complete native creation
target before the receiving handoff mutates either store. Changed expectations
fail closed. Creation still uses its ordinary current authorization and native
permit; the proposal does not replace either. Historical creation replay checks
the original native material rather than resolving new application defaults.
The request snapshot has the named finite ceiling
`MAX_PREPARATION_REQUEST_BYTES` (16 KiB), in addition to profile, budget and
complete contract envelope limits.

After `create_recipient_session` creates an inert FRESH or FORK child, call
`prepare_recipient_admission(creation, context=...)`. This read-only entrance
reconstructs the exact native creation target and retains the full resolved
profile and sponsor binding. The returned proposal is data, not authority.
Submit it as `RequestAdmissionCommand.prepared` with the matching `decision="fresh"`
or `decision="fork"`, empty
`evidence`, no source export, and the exact request revision, effective-input
revision/commitment and next admission generation. The receiving owner checks
the native creation decision, incarnation, immutable receipt and current mandate
and budget receiver before admission. A FORK target additionally binds the exact
historical selection commitment, manifest commitment, view ID and source session
incarnation from the native child creation receipt. It does not renew source
disclosure or require an already-transferred source pin to remain active. Changed
selection evidence is rejected without admission mutation.

For a child carrying qualified immutable local resources, register its exact
destination `LocalArtifactResourceOwner` in `RequestRegistration.resource_owners`.
The admission target carries bounded `ResourceMaterialReference` values binding
the owner, operation, transfer template, accepted receipt and preparation receipt.
The native child receipt proves adoption; the registered owner independently
authenticates continued accepted retention and holds its existing mutation fence
through the admission transaction. A missing/released pin, conflicting commitment
or unavailable native owner refuses admission. This does not acquire new material,
renew the original source's disclosure/preparation grant or expose resource bytes.
Current receiving mandate, participant admission and sponsor checks still apply.
Caller cancellation does not release that fence while the owned admission worker
is still running; exact replay reconciles the original durable operation.

For an existing participant-owned session, use
`prepare_recipient_continuation(RecipientContinuationRequest(session_id=...,
session_instance_id=..., participant=...), context=...)`. This authenticates
participant administration and returns a CONTINUE proposal for one coherent,
released completed whole-turn boundary. Submit it with `decision="continue"`
and the same exact request/input/admission fields described above. The target
binds the participant/session incarnations, current run epoch, binding/checkpoint
commitments, release receipt, completed interaction/model step/event and transcript
frontier, and complete profile. It never appends input, queues a turn or acquires
a writer. Busy, human-paused, incomplete, queued-input and closure-owned sessions
are refused; an arbitrary idle flag is not completion evidence.

The registered receiver independently recaptures that exact selection. A newer
turn cannot silently replace it. Receiver capability 2 qualifies FRESH and
CONTINUE; capability 1 qualifies only FRESH. Capability 3 additionally qualifies
resource-free FORK. Capability 4 also qualifies exact adopted-resource evidence
for FRESH/FORK; earlier receivers cannot silently ignore those fields.
Constructing an identical target
does not authenticate it. Selection is a read, not a cross-store lease: later
execution must still admit against the exact target and current native gates.
An already-committed admission remains historically replayable after the session
advances, without granting permission to execute the old boundary again.

The existing participant permit protocol orders admission against disablement:
registration and local settlement of the admission-only permit commit in the
same CollaborationStore transaction as the request receipt. Disablement first
rejects new admission; an already committed admission remains exactly replayable.
This is not a cross-store transaction, a recipient execution permit or a guarantee
that the child will remain available for a later execution owner.

`lookup_collaboration_admission(expected, context=...)` and the application's
`collaboration_admission_reader().lookup(expected, context=...)` authenticate
current read access and compare the complete expected command. Historical
readback does not re-resolve a budget binding or launch work. Same-key changed
input, target, profile, sponsor or contract evidence is not an exact replay.
Keep the same command/key after cancellation, timeout or lost acknowledgement
and reconcile before attempting another admission.
Committed admission retry authenticates current read access before comparing
durable evidence, even if its receiver registration changed or was removed.
New admission still requires the exact currently qualified receiver.

An inert prepared admission can be cancelled or expired through the existing
request control entrance, including after participant disablement. That settles
the request's unstarted obligation, not child deletion or unrelated execution.
Prepared producer progress and outcomes require a later qualified output-owner
attachment and are refused here. Receipt and proposal data never grant launch
authority. This shared-contract slice does not implement the complete admission
planner.

Prepared profile and budget snapshots have canonical JSON ceilings of 16 KiB
and 8 KiB respectively; the complete prepared command is limited to 48 KiB,
within the existing 64 KiB contract envelope. Request capability version 2 and
schema revision 106 fence writers that cannot preserve this evidence. The existing
typed request, event and permit records remain the durable owners; there is no
second admission database.

## Explicit deterministic planning

`RequestRegistration.planning_policies` registers immutable
`ConfiguredRequestPlanningPolicy` values. The supported algorithm,
`input_revision_rules_v1`, selects a typed decision for an exact effective-input
revision, or its required default. Policies have a pinned reference, schema
version and complete configuration commitment. Arbitrary callbacks and
model-assisted evaluation are not supported. Configuration is strategy, not
permission to execute or disclose content.

See `examples/collaboration/planning.py` for an immutable decline policy and a
single-call host driver. The host preserves operation keys; the example does
not install permissive authority or launch a background retry loop.

`app.plan_collaboration_request(request, context=mandate_context)` takes a
`RequestPlanningRequest`. It binds the complete accepted request, request and
effective-input revisions/commitment, planning and admission operations and
generations, initiator, policy reference/commitment, finite limits, deadline and
optional exact predecessor. Planning requires current `prepare` mandate
authority and participant lifecycle/configuration checks. It does not replace
the existing request admission or participant execution permit protocols.

The native collaboration owner retains the resolved policy and exact intent
before evaluating it. A retained decision is read on retry, not reevaluated
against a replacement deployment policy. Same-key differences conflict. The
returned `RequestPlanningRecord` separates its business state from pending
receiving-stage responsibility; a proposal or record is never a launch permit.
The record stores the selected decision's commitment alongside the immutable
policy, rather than duplicating the complete proposal. Reconstruction checks
that commitment against the policy's selection for the retained input revision.
Similarly, a terminal control disposition binds the exact retained request by
commitment; the record's `control` property reconstructs its complete expectation.

The currently qualified decision families are:

- `RequestPlanningDefer`: an absolute owner-time timer or an exact admission
  prerequisite. Register prerequisite readers as `RequestPlanningAdmissionReader`
  entries in `planning_readers`; they use the existing authenticated
  `RequestAdmissionReader` contract. A positive receipt permits an explicit
  successor, not automatic execution. Missing, unavailable or conflicting
  prerequisite evidence cannot satisfy it.
- `RequestPlanningDecline`: native request admission, terminal outcome and
  permit settlement. The owner can prove local non-dispatch for an initial
  request or a complete history of planner-owned, source-free deferrals.
  Terminal planner-owned questions can also settle after the native owner
  verifies complete question history and no pending delivery/service handoffs.
  Earlier independent receiving responsibility still requires its settlement;
  a later local deferral does not erase it.
- `RequestPlanningClarify`: an exact existing question opening and export
  selection. Planning retains a native stage before acquiring the existing
  source/budget-authorized question owner. Question opening and stage settlement
  commit together. A matching raw opening cannot consume a planner-owned stage.
  Question delivery, service, reply acceptance and final-result continuation
  remain separate operations with their existing authorization requirements.
- `RequestPlanningContinue`: the exact prepared CONTINUE proposal from the
  read-only entrance above. The frozen configuration selects this boundary, not
  a mutable session-ID lookup performed inside policy evaluation. Planning
  retains its exact admission stage before acquiring the registered receiver.
  Native admission, stage settlement and the admitted planning record commit
  together under CollaborationStore. Cancellation can fence an unopened local
  admission but never cancels unrelated work in the selected session. Individual
  stage settlement/exclusion sizes are checked before retention; aggregate
  reserved bytes alone do not establish that a terminal record fits.
- `RequestPlanningFresh`: a frozen `FreshRecipientPreparation` from the explicit
  native preflight entrance above. Planning atomically registers the existing
  creation permit and its exact pending stage in CollaborationStore before
  handing off to SessionStore. Native creation remains inert and owns the actual
  session incarnation. Exact receiving readback settles the same permit and
  records the child in the first stage; a second local stage submits its prepared
  admission to the existing receiver. Final admission rechecks current participant
  and effective-input authority. A successful creation followed by rejection is
  retained, not deleted or silently recreated. Cancellation seals new local
  admission and reconciles creation versus native exclusion; a cancelled observer
  or absent receiving receipt alone never releases that responsibility. These
  are separate durable owner transactions, not a cross-store atomic transaction.
  The maximal creation and final admission receipt envelopes are checked before
  retaining permission to create. Neither stage launches recipient execution.
- `RequestPlanningFork`: a frozen `ForkRecipientPreparation` from
  `prepare_recipient_fork(base_creation, source_selection, source_participant=...,
  context=..., deadline_at_ms=...)`. The base freezes explicit FRESH input and
  preflight; its creation permit is not dispatched as a FORK. Attachment
  references require the separately registered resource recipes described below.
  The view stage retains the full selection and source/recipient permit tuple
  before native acquisition. The native owner durably reserves selection-control
  capacity before the stage atomically registers either permit. If reservation
  fails, cancellation can exclude the still-unregistered stage in the same
  CollaborationStore transaction that fences delayed permit registration. This
  local non-admission proof does not claim that an admitted selection stopped.
  After registration, cleanup requires native exclusion or release evidence and
  uses the already-reserved control capacity. A lost reservation acknowledgement
  is reconciled with the original target; an inert control record alone cannot
  authorize a selection. The stage remains pending through selection, explicit
  adoption and child creation. A separate creation stage binds the resolved
  material to that exact blueprint; it is accepted only from the application-wired
  native receiving owner, not caller-shaped resolution evidence. The original
  view stage settles after native release/exclusion and both source permit
  acknowledgements. A third stage performs final admission. Retained history is
  historical data, not current disclosure or execution permission. This branch
  does not execute the child or provision external resources.

FRESH and FORK policies may include an ordered, bounded `resources` tuple of
`RequestPlanningResource` values. Each recipe contains an exact acquisition
permit, an immutable `ResourceTransferTemplate` (including its complete
acquisition command), and the destination's exact transfer permit. Register the
corresponding `LocalArtifactResourceOwner` instances in
`RequestRegistration.resource_owners` and their concrete mandate preparation
receivers before planning. The recipe is not a resource receipt or authorization.
Each resource has a finite absolute deadline no later than the planning deadline.

The planner retains an acquisition stage before calling the source owner, then a
transfer stage bound to the authenticated acquisition receipt before calling the
destination. Only after every transfer is accepted does it retain the exact
material-bound creation stage. Creation adopts the destination pins; source
acquisitions are discharged separately before final request admission. Cleanup
first reconciles creation or its exact exclusion so it cannot release material
that a delayed child creation may still adopt. Reconstructed recovery uses the
same native operations and registered owners, never a new resource key. Partial
preparation remains pending and capacity-counted until native adoption or
independently authorized cleanup supplies positive settlement evidence.

A successor uses a new explicit planning/admission identity and the exact
predecessor operation/revision. Timer/prerequisite eligibility or an accepted
clarification input is checked transactionally alongside the current request
frontier. Reusing the old input after a reply is rejected. Successors cannot
extend the original deadline, evade earlier generation ceilings, replace admitted
work or abandon unresolved stages. Historical prerequisite receipts prove the
decision's provenance; they do not grant current access.
An absolute timer that is already due remains eligible; it does not renew the
deadline or implicitly create another planning generation.
An explicitly cancelled preparation may be replaced only after every foreign
stage has positive terminal evidence, local admission is fenced, and no native
request admission exists. A created child retains its native ownership: it is
not an exclusion and is never implicitly adopted by a new generation.
The prior cancellation receipt remains immutable; the new operation binds its
exact predecessor revision and inherits the original finite deadline/ceilings.

`lookup_collaboration_plan(expected, context=...)` compares the full expectation
and returns current authenticated readback. `reconcile_collaboration_plan` makes
one bounded progress pass for that same retained operation; it cannot create a
missing intent or select replacement keys. Cancellation, an observation timeout
and lost acknowledgement are not proof of exclusion. Preserve the complete
original request and reconcile it after restart.
FRESH/FORK recovery checks the planning deadline against CollaborationStore owner time,
even if no explicit expiry control was submitted. An elapsed preparation is
fenced through the existing expiry control under current administrative authority;
without that authority recovery refuses rather than starting new preparation.
Pending native creation, view and resource responsibilities remain retained until
exact receiving evidence proves creation or exclusion and the corresponding
retention settlement. A child already created before acknowledgement loss is
reconciled, not discarded or created again. Read-only lookup does not expire plans.
For FRESH recovery, `max_recovery_items` bounds stage progress per call. A limit
of one can return `preparing` after adopting creation; the next exact call handles
final admission. An already-settled creation stage is reconstructed from its
authenticated retained receipt, not recreated or reselected.
FORK recovery similarly bounds preparation, creation settlement and pin settlement
as distinct progress items. It preserves the same native selection and creation
identities; cleanup reconciles a late selection or creation before allowing a
successor. A released source pin does not require the already-created child to be
recreated. The creation-stage receipt derives admission data from its retained
command and native child evidence instead of storing duplicate profile and budget
authority. Complete typed records remain subject to the existing 64 KiB ceiling.

`list_pending_collaboration_plans(context=..., after=None, limit=32)` requires
scope-wide request-read authority. Its typed page contains at most 32 bounded
records, including terminal records with unsettled stages. A cursor is a position,
not a snapshot or ownership claim; begin a later sweep without one to discover
new work behind it. Discovery starts no background worker.

`control_collaboration_plan` accepts an exact `RequestPlanningControl` under
current administrative authority. It can fence a not-yet-opened native question
stage or close an existing question without renewing source disclosure. It does
not settle separate delivery/service debt. Native root request cancellation and
expiry also fence eligible local plans in their transaction, after the original
request's receiving responsibility is proven settled. A failed plan-control write
rolls back that request control rather than leaving contradictory terminal state.
The complete control initiator has the same 8 KiB canonical-JSON ceiling as
native request control. Before retaining or growing a plan, the owner checks
that its maximum-shape bounded cleanup record fits the individual record limit,
as well as reserving aggregate storage. A plan that fits initially but cannot
retain its cleanup evidence is rejected before that responsibility is committed.

Planning capability version 1 uses schema revision 107 and typed native plan,
stage and event tables in Memory, SQLite and PostgreSQL. Named hard ceilings in
`cayu.collaboration.planning` bound generations (32), stages (128), resource
intents (32), page size (32), configured rules (32) and record bytes (64 KiB).
Every request supplies explicit narrower `RequestPlanningLimits`; aggregate
storage and reserved terminal evidence use the existing collaboration quotas.
Pending responsibility blocks unsafe namespace retirement and pruning. Settled
planning evidence is reclaimed before its original request history, not by a TTL
that silently discards unresolved work.
Retired-namespace maintenance may reclaim a settled plan in bounded ascending
stage batches. Its native plan record retains the pruning cursor until the final
batch; partially pruned plans are unavailable to ordinary planning readback or
reconciliation. The same maintenance operation replays its durable receipt,
including after restart. Later creation evidence remains retained until earlier
resource-adoption stages that depend on it have been reclaimed.
FORK selection/exclusion arbitration additionally requires native SessionStore
revision 108. SQLite and PostgreSQL validate its decision table and exact owner
index before accepting work; a damaged current schema is not silently repaired
by inventing missing selection or exclusion history.

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
