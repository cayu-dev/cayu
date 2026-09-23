# Registered session exports

Session exports publish bounded deterministic projections or exactly reviewed
prose from an ordinary session. No participant enrollment or participant/session
binding is required for the ordinary-session route. Optional registered mandate
resolution narrows permission; it does not supply a transport to the receiving
audience, automatic content review, or external-resource retention.

Run the [ordinary-session example](../examples/collaboration/session_export.py)
from the repository with `PYTHONPATH=src python examples/collaboration/session_export.py`.
It uses `InMemorySessionStore`, performs no model calls, and exports a row count,
an exact host-approved report reference/decision, and an explicitly reviewed
standalone announcement with no selected private source. Its report projector rejects
extra fields, attachments, changed revisions and unrelated reports rather than
copying schema-shaped content. Its review owner holds an exact approval, not an
automatic text sanitizer. Both example authorities are process-local; in-memory
state does not survive process restart.

## Registration and public entrances

Import the user-facing contracts from `cayu` or `cayu.collaboration`.
Register trusted host implementations with
`CayuApp(session_store=store, session_exports=registration)`, where
`registration` is a `SessionExportRegistration` containing an owner, policy,
projectors, `ExportLimits`, and optional receiving-owner acceptance readers.
`release_readers` registers exact reviewed-content owners; `mandates` registers
a trusted mandate resolver; `resource_owners` registers qualified subtree owners.
This API is separate from the existing session snapshot/backup export API.
Readiness requires native store capability attestation, not merely a caller-set
capability version or boolean. Registration alone cannot enable an unsupported
store.

All context arguments below are keyword-only:

| Public call | Result and purpose |
| --- | --- |
| `initialize_session_exports(session_id, context=context)` | Stable `SessionExportNamespace` for the exact session instance. |
| `export_session(request, context=context)` | `SessionExportReceipt` for a new publication or exact historical replay. |
| `export_session(request, context=context, invocation=ctx)` | Runtime export from a live runtime-issued `ToolContext`, with a separately authorized runtime policy. |
| `lookup_session_export(request, context=context)` | Exact lookup result; inspect its `status` before accessing a matching `receipt`. No output payload. |
| `read_session_export(request, context=context)` | Authorized retained output as a `dict`; unavailable or conflicting requests raise. |
| `settle_session_export(settlement, context=context)` | `SessionExportSettlementReceipt` for release or retirement. |
| `reconcile_session_export(request, context=context)` | Finish participant admission settlement for a published export, or permanently exclude a prepared export before settling its permit. Never reruns projection. |
| `drain_session_exports()` | Seal the export owner against new calls and drain owned work, including work still running after its observer was cancelled. |

Construct `SessionExportRef` using the returned session ID, session instance ID,
and namespace. Its operation reference contains the namespace owner's
`application_scope`, `namespace_incarnation`, `generation`, and a stable
`caller_key`. Nested reference values can be supplied as dictionaries through
the public models' `model_validate()` methods; private store schemas are not
part of the public API.

`SessionExportRequest` also binds selected `source_indices`, the exact audience,
and registered projector and policy references. Indices are unique, canonicalized,
and limited to 16. The current source adapter admits text and structured tool
results, not hidden thinking or assets. The registered projector must select only
approved fields; source admission alone is not disclosure permission.

`source_selection` defaults to `whole_records`, preserving that record-level
restriction. Explicit `assistant_visible_text_v1` selection is available only for
deterministic exports of assistant records containing nonblank visible text.
Those records may also contain provider-state and thinking parts; the source
owner removes these private parts before either projector callback. Other part
types, other roles, empty selection, and records without eligible text are denied.
The callback view preserves record indices, interaction attribution, text order,
and text-part attribution; it does not create a new transcript record or tool result.

The immutable request and receipt bind the selection/version, original session
incarnation and record indices, audience, policy and projector. The public
`source_commitment` commits the selected text view, not private provider state.
Complete original records are independently validated inside native publication;
that validation commitment stays internal, including during participant admission.
Changing selection under the same operation key conflicts. Historical replay
returns the original output under fresh authorization, without re-projecting
source material or renewing revoked access.

## Authorization is a held guard, not a boolean

`SessionExportAccessContext` is authenticated host input, supplied separately from
the request. Constructing one does not authenticate a principal. Never populate it
from model arguments or an unauthenticated request body.

`SessionExportPolicy.acquire()` returns an async context manager yielding
`SessionExportAuthorization`, or raises `SessionExportDenied`. The implementation
must check every requested action, the exact session instance, principal, and
audience. Deny empty and unknown action sets. Serialize revocation with the entire
guarded operation; an authorization boolean or an earlier access check cannot
replace this guard. The example uses one process-local lock for both acquisition
and revocation, not a distributed authorization guarantee.

Initialization requests `initialize`; new publication jointly requests `readback`,
`source`, and `export`; receipt lookup/replay requests `readback`; payload reads
request both `readback` and `expose`; new settlement jointly requests `readback`
and `release` or `retire`. Combined guards retain readback permission during
concurrent reconciliation without nesting policy locks.
Publication and settlement first check for replay under fresh readback permission.
Authorization evidence has a finite expiry, checked against the store clock
after receipt reads as well as before publication and payload exposure.

`SessionExportProjector.project()` is deterministic and must not call a model.
Its independent `validate(source, output, audience)` must return literal `True`
for the exact approved output. That output-validation boolean is not policy
authority. Registered references must identify the implementation and
configuration; do not silently change behavior under the same reference.

### Runtime requester provenance

A trusted tool implementation may pass its runtime-issued `ToolContext` as
`invocation`. Cayu derives the requesting session incarnation, run epoch, root
invocation, interaction, model/tool identities, argument commitment and execution
profile from the existing private runtime authority. It checks the live owner
and stored requester before admitting work. Copying or reconstructing the
context does not retain that authority, and a context from an ended invocation
cannot start another export. Admission ends when the originating tool call
finishes, including for child tasks holding a copied execution context. Exports
already handed to the retained owner continue settling independently of that
admission lifetime. The requester may differ from the exported source.

The registered policy must explicitly implement `acquire_runtime(context,
origin=..., session_id=..., session_instance_id=..., actions=..., audience=...)`.
The default refuses. Authenticate the requested principal against this origin
and apply the same held source/action guard; host-asserted origin is not
server-verified identity. Policy-returned values cannot inject runtime provenance.
The durable receipt retains historical origin, not a reusable runtime grant.
An authorized host can inspect that receipt after restart without recreating a
live tool context. Host-only read, settlement and administrative APIs still
require trusted application context; do not expose them as unauthenticated tools.

### Mandates and participant admission

When `mandates` is registered, supply a `MandateAccessContext` in the access
context. Its references are proposals, not credentials. The registered
`MandateResolver.acquire()` must authenticate the issuer/principal mapping and
entire root-to-leaf chain, holding current revocation until the operation settles.
Cayu checks its pinned resolver identity, action/audience/scope permissions,
ancestor expiry at store time, exact lineage, shrinking depth and permissions,
unchanged sponsor, and preservation of inherited budget references. These
references do not replace `BudgetLedger` admission or authorize external spend.
Resolver `MandateDenied` and participant `CollaborationAccessDenied` refusals
surface as `SessionExportDenied`, not transient readback unavailability.

Exact `ResourceSelector` values require pinned owner revisions. A bounded tuple
expresses a union. Subtree containment requires a registered
`ResourceSelectorOwner`; aliases must already match its canonical result and
`contains()` must return literal `True`. Textual prefixes, globs and regexes
provide no containment authority. Selected transcript rows use source-owner
references with kind `session_transcript_row` and revision equal to row index
plus one. Reviewed exposure occurrences also undergo channel, resource and
exclusion checks, including when no transcript rows are selected.
Every ancestor's excluded sources must also match the registered resource
owner's canonical identity. Cayu rejects aliases rather than rewriting
authenticated restrictions.

Participant-attributed operations use the actual participant access boundary and
durable permit admission before projection. The source retains an exact prepared
handoff so acknowledgement loss can be reconciled. A participant identity does
not prove ownership of a session, and this route does not create or backfill a
participant/session ownership binding. A registered admission can settle after
disable; a new admission cannot bypass current lifecycle fencing.
If native settlement commits before its source acknowledgement, reconciliation
can discharge the published export's admission using authenticated retirement
evidence for its exact collaboration namespace, even after permit pruning.
Missing evidence or another namespace's evidence leaves the source fenced;
retirement does not manufacture an exact permit settlement receipt.

If native permit admission was rejected at capacity, source reconciliation still
records permanent source exclusion using its reserved envelope. A new negative
permit record cannot spend namespace maintenance reserves. If ordinary native
capacity is exhausted, reconciliation reports capacity and keeps the source
deletion fence. An authorized operator can rotate and retire that collaboration
namespace, then retry the exact source reconciliation: permanent namespace
retirement proves the old permit can never register, without inventing a receipt
or rerunning projection. Registered permits instead settle using capacity reserved
when they were admitted. Do not delete the source while either responsibility is
pending.

### Exactly reviewed prose

Set `mode="reviewed_prose"` and supply a `ContentReleaseRequest` referencing the
authorized decision. Its commitments bind exact UTF-8 text, selected source and
bounded source/channel exposure manifest; the export request binds audience and
policy/validator versions. In this mode `projector` identifies a registered
`ContentReleaseReader`, not a deterministic projector. Its held `acquire()` must
load positive application-owned review evidence and return `ReleasedContent`.
Echoing a caller's approval-shaped value does not authenticate it.

Cayu compares the full release expectation, source commitment, exact text digest,
issuer, expiry and audience before publishing `{"text": ...}`. Reads require
current release permission and exact retained approval, as well as source
exposure permission. A byte edit, different exposure or broader audience requires
a different authorized decision. Receipt-only lookup does not re-review or
regenerate the output. This verifies binding to an approval; the application's
reviewer remains responsible for the content's suitability and confidentiality.

## Retention, replay, and settlement

Retain the original request and stable operation key for acknowledgement-loss
reconciliation. Exact replay returns the original receipt without re-running the
projector. Reusing the identity for different request material or a different
initiating identity (issuer, including its incarnation, and principal) conflicts.
Authorized receipt inspection and payload reads do not require the original
exporter; their current policy guard decides access. Historical receipt equality does not establish current
permission: lookup, replay, and exposure acquire fresh policy guards.

The export transaction retains output, receipt, metadata event, and pending
responsibility together. Limits cover export count, pending count, and retained
bytes; admission reserves capacity for settlement. Output is bounded to 8 KiB,
selected source to 32 KiB, and complete contract representations remain bounded.
An individually small payload does not guarantee admission.

Pending responsibility fences session erasure. Use a distinct operation key in
`SessionExportSettlementRequest`:

- `release` requires exact acceptance resolved by a registered
  `SessionExportAcceptanceReader` for the receiving owner. A caller-supplied
  receipt or boolean is not acceptance. Session exports do not deliver the
  output itself.
- `retire` requires current retirement permission and no receiving acceptance.
  Retired output cannot be read through `read_session_export`.

An authorized administrator may settle another principal's export. The settlement
receipt binds its own `initiator`; exact settlement replay compares that complete
identity and validates the retained source owner and namespace. It does not
rewrite the export's original initiator.

Receipt lookup returns `match`, `conflict`, `not_found`, or `unavailable`.
Malformed retained evidence and unavailable authorized readback produce
`ExactUnavailable`; access denial and caller cancellation remain raised signals.
Payload exposure is a separate API and still raises when output is retired.

Settlement clears pending responsibility but is not pruning: retained output and
historical evidence are not reclaimed, and export/retained-byte capacity is not
reset. Exact export and settlement replay remain subject to current readback
permission. Stop issuing new export API calls before calling
`drain_session_exports()` during shutdown. Drain seals this export owner; it is
not a reusable flush, and later calls are rejected. If work remains in flight,
keep the host and store alive and retry draining rather than assuming it stopped.
Cancellation of an awaiting caller does not prove a dispatched projector or
store mutation stopped.

## Privacy and failures

Do not put secrets into identity fields, requests, logs, or example output.
Do not assume a projector automatically removes secrets: explicitly select and
validate approved output. Payload exposure belongs exclusively to the authorized
read API, not events or receipt lookup.

The `session.export.published`, `session.export.released`, and
`session.export.retired` event payloads contain only `export_commitment` and
`output_commitment` (64-character lowercase SHA-256 hex). They contain no inline
output, source content, or principal. Public metadata and historical receipts
are not disclosure grants or replay permission; authority must be verified at
the owning store/policy boundary.

Handle `SessionExportDenied`, `SessionExportConflict`,
`SessionExportCapacityExceeded`, and `SessionExportUnavailable` separately.
Malformed contracts can raise `CollaborationContractError`. Missing registration
or an unsupported store fails closed. After interruption or uncertain delivery,
reconcile the exact request rather than inventing a new key or reconstructing
authority from event metadata.

## Trusted JSONL history restoration

The trusted storage JSONL import boundary restores exact export-event history;
ordinary callers cannot append forged or modified export events. This is history
restoration, not transfer of source-owned export authority. JSONL does not carry
the private output/receipt operation records, so import rejects a checkpoint with
pending export responsibilities. Settle them at the source before restoring.
For a settled namespace, import omits its private root, retains its historical
events, and confers no payload, receipt-replay, or settlement authority at the
destination. Keep the source store for those operations. Never import untrusted
JSONL.
