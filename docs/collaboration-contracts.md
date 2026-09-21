# Collaboration contract foundation

## Implemented versus planned

Cayu has internal shared collaboration value types, bounded preparation,
complete-value comparison, typed exact-lookup outcomes and versioned capability
descriptors under `cayu.collaboration`. These private modules are implementation
infrastructure, not a public agent API. Importing the package starts no work and
registers no owner. The [participant identity API](participant-identity.md) builds
on this foundation with a production CollaborationStore for scoped identities,
lifecycle admission, permit responsibility and namespace retention. It does not
dispatch collaboration requests. Participant-owned inert-session creation and
historical context-view retention are documented separately in
[`context-views.md`](context-views.md); those APIs remain capability-gated by
the configured native SessionStore.

Existing SessionStore, TaskStore, execution/resource owners and BudgetLedger keep
their current authority. Colocating them in one database does not make separate
API calls one transaction. The handoff matrix below specifies future owner
integration obligations, **not implemented operations**.

## Values, preparation and authority

`OperationRef` identifies one operation by application scope, namespace
incarnation, generation and caller key. Kind, schema version and mode are compared
under that identity rather than creating new key domains. `OwnerRef` and
`ObjectRef` name logical owners and object incarnations, not database URLs or
Python object addresses. `InitiatorBinding` retains the original issuer,
principal, participant, mandate and originating invocation/interaction.

`ExpectedOperation[Intent]` combines those values with a concrete owner-typed
intent and expected receipt stage. Each owner must include **all** operands,
flags, ordered inputs, declared sets, selectors, policies/profiles, semantic
generations, deadlines/bounds, retained references and transfer/cleanup/budget
obligations that affect its operation. Already-frozen selections are compared;
unknown owner-generated selections are authenticated from the original receipt
and adopted once. A digest alone is neither complete expectation nor provenance.

`TransportClaimRef` is a current servicing reference, not proof of a live lease.
It stays outside immutable business identity. Existing owner-time lease and
fencing mechanisms remain responsible for permission to service work. Replacing
a worker does not replace the original initiator or the operation key.

All types are **values, not authorization tokens**. Construction, private imports,
copying, JSON round trips and equality do not authenticate a caller. The receiver
must authenticate access and originating authority and reconstruct trusted
evidence from its own exact durable boundary. A trusted Python host can execute
arbitrary code; private module names are not a sandbox against that host.

`prepare_contract` accepts a trusted concrete schema and trusted SecretRedactor.
It bounds and detaches supported model fields and primitive containers without
calling input serializers, checks original values for known secrets, validates
the concrete schema, checks resulting values again, and enforces aggregate
canonical limits. Errors at this preparation entrance are content-free and do
not attach raw validation exceptions. Model construction alone does not perform
the workload-secret check; every receiving boundary must prepare its inputs.

Owner schemas extend `ContractValue` with strict scalar, concrete nested-model
and tuple fields. Mutable dictionaries/lists are not accepted as immutable
authority. Use strict scalar annotations; the shared JSON encoder deliberately
normalizes numeric values and cannot replace an integer field's strict check.
Fields declared in `unordered_fields` are sorted by canonical value, with
duplicates rejected. Ordered fields stay ordered; concrete schemas explicitly
define defaults and absent-value semantics. No generic metadata bag supplies
missing authority.

Declared schema keys and validated finite literal controls are structural.
Caller-selected identity/text values and arbitrary container keys are not
structural merely because they have the same spelling. Known-secret input is
rejected, not redacted into a different accepted operation. This is not universal
semantic detection of confidential business content.

### Development bounds

| Dimension | Inclusive bound |
| --- | --- |
| Complete canonical envelope | 64 KiB |
| Nodes, including object keys | 8,192 |
| Traversal nesting | 64 |
| Entries in any object/array | 64 |
| Identity text | 512 UTF-8 bytes |
| Protocol/family/stage code | 128 ASCII identifier characters |
| New generations/schema versions | Positive strict integers through the portable JSON integer maximum |

These are development limits, not measured throughput guarantees. Existing
store-owned counters keep their own contracts. Owners still need explicit finite
configuration for participants, active requests, fan-out, wait lifetimes,
retention, pins, pending/draining work, control reserves and spend. Missing limits
must not mean unlimited. Nested members fitting individually do not exempt the
complete envelope from its bound.

## Lookup and exact replay

| Internal result | Meaning |
| --- | --- |
| `ExactMatch[Receipt]` | Carries the owner's exact typed receipt; the owner must verify expected input and source authority |
| `ExactNotFound` | No record in the authoritative snapshot; never positive exclusion or proof of non-dispatch |
| `ExactConflict` | The scoped key conflicts with expected material |
| `ExactUnavailable` | Authoritative readback cannot currently be established |

Only a match carries a receipt. A positive exclusion has its own receipt stage;
queue acceptance is not completion, and stop acceptance is not quiescence.
Immutable receipts and current progress/disposition remain distinct.

Receiving owners authenticate namespace/read access, prepare expected input,
perform exact lookup and compare the full tuple before returning a receipt or
conflict. Only new operations resolve new policy/defaults and require new-effect
capabilities. Historical readback still requires current permission and valid
receipt/effect invariants; disabling new writes does not itself erase receipts.
Authentication denial is separate and may conceal existence on public surfaces.

The value/preparation layer performs no I/O or retries. Owner adapters must not
map observer cancellation or fatal control signals into ordinary unavailability.
Cancelling an observer does not establish that dispatched work stopped, nor does
it cancel the business operation. Pending ownership remains until positive
settlement or exclusion.

## Child keys and capabilities

`HandoffSlot` names an immutable source scope/incarnation, parent operation and
slot. `HandoffIntent[Intent]` binds that slot to an exact child command. Source
stores must atomically elect an opaque child key and full intent before dispatch,
with unique slot and scoped child-key constraints. A losing proposal adopts the
stored winner only after comparing the remaining complete tuple. Recovery reloads
that key. Fresh retry keys and keys derived from changed input can create duplicate
work instead of conflicts. A legitimate successor is explicitly elected.

Public input cannot choose runtime-owned key namespaces; actual namespace
authentication and key election belong to future receiving/source owners. This
foundation creates neither a key service nor a durable namespace store.

`CapabilityDescriptor` identifies an owner and canonical sets of supported
mutation/readback family versions. Every advertised mutation requires its matching
readback contract. A family version denotes its entire required atomic publication,
expected-input readback, applicable exclusion, owner-time and retention/frontier
guarantees—not independently switchable safety bits. `require_capability` compares
against an explicit supported family definition and expected owner. An arbitrary
family does not become supported by declaring version one.

Descriptors must come from trusted application registration bound to the real
implementation/wrapper. They are not accepted as caller-supplied permission.
Wrappers may not retain a declaration while dropping its guarantees. Shape
validation cannot prove a custom adapter truthful; real conformance is required.
No production collaboration family is advertised by the test receiver.

## Future owner handoff matrix

Every row includes the common tuple above: kind/version/mode, scoped operation
identity, original authenticated principal/issuer/participant/scope, source and
destination incarnations, complete intent and frozen selections,
policy/profile/mandate/generations, deadlines/bounds, receipt stage and exact
quota/retention/cleanup obligations. Rows add the following specific authority.

C means collaboration owner; S session owner; T task owner; X existing
resource/execution/source owner. The labels below identify design rows, not APIs.
All rows remain future production capabilities.

| Handoff | Destination and additional exact tuple | Receipt stage and retained responsibility |
| --- | --- | --- |
| H1 Context selection | S: source participant/session incarnation, boundary selector, projection, freshness, pin/acquisition limits | Exact retained manifest/pin; C preparation retains acquisition/release until adoption |
| H2 Resource acquisition | X: resource owner/kind/selector, key, allowed operations, isolation, count/bytes/cost, cleanup owner | Revision/allocation/reservation; X owns late acquisition and cleanup |
| H3 Child materialization | S: recipient, exact fresh/retained input manifest and pin, creation key, profile/tool/exposure/budget lineage, first input | Store-minted child incarnation; C freezes selection before launch |
| H4 Invocation | S/T: exact target, invocation/interaction/dispatch, input/output contract, profile, launch generation | Queue/invocation or exclusion; execution owner and C outcome index |
| H5 Wait registration | C: S ticket/session/interaction generation, canonical targets, predicate/version/threshold or absence, deadline/failure/service policy, edge revision | Exact wait receipt; S retains registration until C owns observation |
| H6 Source observation | Source: incarnations, filter/projection, frontier selector, replay/retention requirements | Frontier/coverage/pin; C catch-up and source retention |
| H7 Producer export | C: binding/admission, producer interaction/effect/publisher, output contract, kind/sequence/payload, source revisions/exposure, projection/validator/audience and exact release | Immutable occurrence/output and election; bounded fan-out, no producer rerun |
| H8 Peer append | S: occurrence/consumer/target/projection key, payload, interest/deadline, attempt generation, wake contract | Queue/append; S owns exposure and continuation |
| H9 Outcome latch | S: ticket/generation, exact wait target/predicate/threshold receipt, immutable outcome/manifest, disclosure, latch key | Latch; S indexed consume or retire |
| H10 Continuation | S/T: ticket/consume generation, continuation identity, input/profile/budget/purpose | One continuation acceptance; inline/queued execution owner |
| H11 Interest detachment | C: request terminal receipt, binding/attachment/membership, last-interest disposition, independent obligation/sponsor/retention | Detach/closed-membership disposition; indexed stop or continuation |
| H12 Stop | X: execution/descendant/effect scope, lifecycle generation, exact close/disposition, stop policy/bound and settlement allowance | Stop acceptance then separate exclusion/quiescence; X cleanup and ledger settlement |
| H13 Lifecycle fence | S/T/X: election, permit frontier, target/operation generations and settlement scope | Exclusion/stop including absent targets; C retains unsettled frontier |
| H14 Delivery exclusion | S: interest/occurrence, immutable attempt generation, append key and withdrawal scope | Append-before-fence or exclusion; C reconciles before successor |
| H15 Publisher fence | C: task/contract, old/new assignment/attempt, old binding/publisher and supersession intent | Exact fence; T retains successor activation responsibility |
| H16 Member settlement | X/C: group/decision, exact member roles/winner/loser set, start/stop obligations and finalizer/non-interference contract | Exclusion/quiescence or independent disposition; T owns finalizer barrier |
| H17 Retention transfer | Source: exact pin/reference set, old/new owners, material commitments and transfer stage | Transfer/release; indexed source cleanup until confirmed |
| H18 Remote operation | Adapter: verified issuer/audience/peer, endpoint/config, complete operation tuple, capabilities/evidence and allowance | Authenticated exact receipt; local projection preserves evidence class |
| H19 Clarification | S: question/input/responder, ticket/side-session, service generation/profile/mandate/spend/deadline, reply key and return policy | Service/return; S single writer and final-outcome latch |
| H20 Commission/link | T: mode/authority, source/candidate commitments, task/objective/criteria/acceptor, recipient offer/assignment/graph revisions, resource/budget/deadlines/acceptance window, attachment/cancellation | Held offer/link and retention; C relationship/admission, not recipient consent |
| H21 Activation | T: H20 link/hold, exact current H22 responsibility, task/contract/assignment, permit/lifecycle/mandate, graph/ordinary gates | Activation/exclusion; readiness can remain blocked |
| H22 Responsibility | T: offer/task/contract/assignment/recipient, decision and authenticated recipient or delegated policy/version, scope/limits/revocation, expected state/deadline | Responsibility decision with hold/event update; C reconciles before activation |

Source intent/event/pending index and destination mutation/event/receipt are
separate owner-atomic transactions. Acknowledgement transfers only the named
responsibility. Unknown foreign state cannot justify release, redispatch or new
spend. Later owner implementations must qualify public, internal, callback,
worker and reconstruction entrances as applicable.

## Verification boundary

The repository-private `tests/core/collaboration_conformance.py` defines typed
fixture operations and assertions. A limited in-process receiver self-tests
replay, conflicts, source election, exclusion, partial batches, permission shape
and six failure windows. Deliberately defective variants demonstrate that the
assertions detect duplicate mutation, changed-input replay and fabricated
exclusion. Real `Task.cancel()` exercises observer behavior with retained pending
work. These checks are not durable backend or process-restart qualification.

Persistent owner adapters must implement the explicit reopen/independent-worker
fixture facilities and supply real SQLite/PostgreSQL process-loss and concurrency
evidence. Do not replace those facilities with an in-memory reset. Existing
session publication adapters can reuse the
[session-operation fault harness](session-operation-fault-harness.md); task and
resource owners keep their own transaction and fault boundaries. No public
conformance package or adapter certification is exported by this foundation.
