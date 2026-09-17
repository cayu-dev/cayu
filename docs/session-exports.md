# Registered session exports

Session exports publish bounded, deterministic projections of selected transcript
rows from an ordinary session. No participant enrollment or participant/session
binding is required. The supported contract does not provide released prose,
model-generated export content, delegation, pruning, or a transport to the
receiving audience.

Run the [ordinary-session example](../examples/collaboration/session_export.py)
from the repository with `PYTHONPATH=src python examples/collaboration/session_export.py`.
It uses `InMemorySessionStore`, performs no model calls, and exports only a row
count. In-memory state does not survive process restart.

## Registration and public entrances

Import the user-facing contracts from `cayu` or `cayu.collaboration`.
Register trusted host implementations with
`CayuApp(session_store=store, session_exports=registration)`, where
`registration` is a `SessionExportRegistration` containing an owner, policy,
projectors, `ExportLimits`, and optional receiving-owner acceptance readers.
This API is separate from the existing session snapshot/backup export API.
Readiness requires native store capability attestation, not merely a caller-set
capability version or boolean. Registration alone cannot enable an unsupported
store.

All context arguments below are keyword-only:

| Public call | Result and purpose |
| --- | --- |
| `initialize_session_exports(session_id, context=context)` | Stable `SessionExportNamespace` for the exact session instance. |
| `export_session(request, context=context)` | `SessionExportReceipt` for a new publication or exact historical replay. |
| `lookup_session_export(request, context=context)` | Exact lookup result; inspect its `status` before accessing a matching `receipt`. No output payload. |
| `read_session_export(request, context=context)` | Authorized retained output as a `dict`; unavailable or conflicting requests raise. |
| `settle_session_export(settlement, context=context)` | `SessionExportSettlementReceipt` for release or retirement. |
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
