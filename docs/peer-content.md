# Typed peer content

Peer content is authenticated historical data delivered to existing or durably
identified future participant-owned sessions. It is not user steering, a tool result, a system
instruction, or permission to execute work.

The public append boundary accepts a `PeerContentAppendRequest` only with a
`CollaborationAccessContext`. The immutable append key binds the collaboration
generation, source occurrence, consumer, projection/schema, and target session
incarnation. Retries with the same complete request replay the durable receipt;
changing any decision-bearing field conflicts before mutation.

Ordinary run and resume input messages cannot contain `PeerContentPart`, even
when they exactly match an accepted occurrence. Only authenticated peer delivery
may insert these parts; durable transcript reconstruction preserves them as
historical data, not as permission for another caller-authored insertion.

Before queue admission, the registered export policy must authenticate the
producing occurrence and its current source/export authority through
`acquire_peer_append`; it yields typed evidence binding the exact source export
receipt, producer record, source session/incarnation, content commitment and
audience. The policy holds its revocation guard through durable queue
admission. A caller-supplied payload or commitment is not evidence of
production. The SessionStore transaction independently verifies
the exact target participant/session binding.

Queue acceptance, transcript append, and provider exposure are separate
outcomes. A pending or excluded receipt is never treated as an answer or
consent. Providers must render peer content as explicitly attributed external
data; providers and stores that do not advertise the peer-content capability
fail closed.

Named provider exposure is recorded separately by the runtime with
`PeerContentExposureRequest`. The
exposure operation must identify the append key, model attempt, provider and
capability version; replaying the complete request returns the durable receipt,
while any changed field conflicts. Settlement does not itself launch a model
or authorize a new append.
Runtime exposure identifiers are bounded, domain-separated commitments of the
append key/operation, model attempt, provider and capability version. Full native
request comparison remains authoritative; the digest grants no permission.
Authoritative `exposed` receipts are minted only by the runtime model-step
owner after provider serialization; the public facade rejects caller-authored
positive or negative exposure claims through `CayuApp.expose_peer_content`.

Peer content does not create recipient sessions, wake a model, grant tools,
transfer budgets/resources, or replace human/approval gates.

## Current authorization, attempts, and interruption

The registered append policy receives the complete request, including interest,
withdrawal authority, deadline and replacement identity. Public readback uses
`SessionExportPolicy.acquire_peer_read`; its default denies. A revoked read
returns the historical delivery status with `disclosure="withheld"` and no
occurrence payload. It never rewrites an append as a failed delivery.

Generic session-message inspection and action results retain peer lifecycle
metadata but mark the payload unreadable. Queue credentials are not source
disclosure credentials; use the authenticated peer read API for content.

Native admission invokes the runtime's synchronous provider qualification against
the transaction-owned target before queue insertion, including when target
creation wins a race. The recorded fallback plan's providers must also qualify.
An unresolved creation may retain a pending attempt, but cannot enter the queue
without this check. Native replay does not re-admit already accepted work.
New admission also honors the ordinary queue's interruption and pending
completion-finalization fences. Peer operation keys share the target queue's
idempotency namespace with ordinary steering; conflicting use is rejected,
never overwritten.

Each consumer incarnation can own at most
`PEER_CONTENT_MAX_OUTSTANDING_PER_CONSUMER = 64` pending or queued peer
deliveries in aggregate. Native transactions enforce this ceiling, combining
both representations and counting the current key only once. Exclusion or
terminal queue settlement frees capacity; retry does not consume another slot.
Exact target deletion durably discharges capacity for its successful appends
without removing their receipts. Missing queue rows alone are not settlement
evidence, and unresolved pending attempts still require exclusion/reconciliation.
Individual payload ceilings therefore also bound outstanding retained content.
This is peer admission capacity, not a new quota on ordinary user messages.

Every delivery attempt requires `deadline_at_ms`. The native transaction owner
compares it with store-owned time; equality is expired and records a definitive
`delivery_deadline_expired` exclusion. An already committed append remains an
append after its admission deadline. Same-operation deadline changes conflict.

A new operation can replace only the exact excluded predecessor named by
`replaces_operation_key`, with a strictly increasing attempt generation (1–64).
The occurrence, successful-append key and wake policy remain unchanged. Pending
attempts fence replacement; successful appends cannot be replaced. Native attempt
history retains exact old receipts, so delayed retries cannot regain ownership or
erase the successful successor. Public `read_peer_content(..., expected=request)`
compares the complete expected operation, including excluded predecessors.
Pending servicing keeps the original caller tuple immutable. The receiving
owner passes a separate refreshed transcript cursor to the native transaction;
only an exact already-pending operation may use that fence. Replaying the
original request after servicing or restart returns the committed outcome.
Exclusion and replacement serialize with append
under the existing native transaction owner; the destination creation fence is unchanged.

Provider adapters report `record_peer_serialization(request)` only after building
the approved payload. This records exposure at the serialization boundary, not
remote acceptance or model completion. The runtime-only observer is excluded from
request JSON and remains bound to the exact request projection. All six built-in
adapters report it, including OpenAI background start. Missing runtime authority
fails closed for peer serialization.

Remote token counting has no peer-disclosure receiver and is unavailable for
peer-bearing context. The runtime uses its existing estimation fallback instead;
the built-in remote counting adapters also reject direct peer-bearing requests
before payload serialization or network access.

Guard cleanup, cancellation, timeout and stream abandonment never imply either
exposure outcome. Before positive serialization evidence, the exact exposure
receipt stays pending; afterward its durable exposed receipt survives interruption.
An acknowledgement failure is reconciled by exact exposure readback, not provider
redispatch. Definitive receiver denial before serialization records not-exposed.

An exact retained delivery may be discharged under the registered
`SessionExportPolicy.acquire_peer_exclusion` guard even when content disclosure
has been revoked. The default denies; the policy must authenticate current
withdrawal/cleanup authority for the full request, retained receipt and reason.
This guard cannot create a fresh obligation or grant content access. If append
already won, exclusion returns truthful appended status with its payload withheld.
Exclusion before any retained operation still requires producing authorization.

Checkpoint compaction retains historical peer parts separately as typed messages.
Compactors receive content-free markers, never the peer payload; peer-bearing
prompt-cache compaction uses bounded projection rather than the raw cached prefix.
Reconstruction derives retained peer parts from the durable transcript. Subsequent
provider requests still need the ordinary current model-attempt disclosure guard.
Retained peer input counts toward context-size limits and is not discarded to
make a summary fit. Compaction does not renew or bypass revoked disclosure.

## Registered export runtime disclosure seam

The registered session-export receiver currently authenticates runtime exports
through `SessionExportPolicy.acquire_runtime(...)`. That entrance requires an
authenticated `SessionExportAccessContext` plus a `SessionExportRuntimeOrigin`
derived from a live `ToolContext`. It is intentionally denied by the default
policy, and the runtime origin is not a caller-shaped receipt.

That is not a suitable handoff for this issue's model-attempt exposure: a peer
occurrence is produced by a completed assistant turn, while provider exposure
occurs in a named model attempt. The model-attempt receiver seam below is the
small shared-contract extension for those inputs. Passing a fabricated access
context or reusing a historical append receipt would bypass session-export revocation
semantics, so the implementation remains fail-closed here.

The shared-contract extension is represented by the registered
`PeerContentExposureReceiver.acquire_peer_exposures(context, items=...)`
seam. The receiver must recheck source/export/audience/exposure authorization
and hold revocation through the provider-attempt boundary; it must not accept
a caller-supplied receipt or a fabricated export context. The model-step owner
now mints a distinct
`PeerModelAttemptOrigin`, loads the exact append receipt, acquires this guard
before provider dispatch, and serializes only the bounded
`PeerContentPayload` yielded by the receiver. Exposure evidence is recorded
against the same model attempt. A missing receiver, malformed origin, changed
target incarnation, or revoked authorization fails closed.
The attempt owner resolves the authorized projection before request measurements
and durable dispatch fingerprinting. The projected messages pass through runtime
secret redaction and validation before becoming the serialization snapshot; the
receiver guard remains held through provider serialization and attempt cleanup.

When no separate receiver object is registered, Cayu uses a thin adapter to the
registered `SessionExportPolicy.acquire_peer_exposures` hook. Each typed
`PeerContentExposureItem` contains the runtime origin (including exact append
key, target incarnation, provider and attempt identity), occurrence, and audience.
The hook authenticates every item under one shared revocation guard and yields
an ordered tuple of bounded payloads. It must not recursively acquire a
non-reentrant single-occurrence lock. The base implementation delegates exactly
one item to `acquire_peer_exposure` and rejects larger batches; multi-occurrence
receivers must explicitly implement the batch hook. Neither default grants
access or creates a second policy.

Generic HTTP transcript readback has no source-export disclosure authority.
It returns only `{"type": "peer_content", "disclosure": "withheld"}` for peer
parts, even before revocation; payloads and private append/creation identities
are not serialized. Authorized payload readback uses `CayuApp.read_peer_content`
and its current registered export-owner authorization.

The qualification path demonstrates the source side with the same owner: it
creates a durable `SessionExportReceipt` through `CayuApp.export_session`,
reads the exact export through `CayuApp.read_session_export`, and passes that
receipt identity into peer admission. The registered owner verifies the
receipt's source session/incarnation, audience and exported payload commitment
before yielding `PeerContentAppendAuthorization`. This is an application
qualification receiver, not automatic producer publication or fan-out.

Ordinary assistant responses containing private provider state can use explicit
`SessionExportRequest(source_selection="assistant_visible_text_v1", ...)`.
The existing export owner supplies only approved visible text to its registered
projector while retaining exact source validation internally. Export authorization
and peer append/read/exposure guards remain separate; this selection never grants
disclosure permission. Private provider state and thinking cannot become peer payloads.

## Destination creation fence

A peer append key names either an existing session/incarnation or the complete
`SessionCreationTarget` supplied by the shared creation owner. Future keys
must leave both session identity fields absent; they never guess an incarnation.
The stable key retains that creation target after creation. An appended receipt
separately records the resolved session ID and store-minted incarnation.

Memory resolves the shared decision while holding the SessionStore mutation lock.
SQLite reads it within the peer write transaction. PostgreSQL acquires the shared
creation-operation lock before peer-operation and destination-row locks, then
reads it on the same cursor. Every append revalidates the live binding against
the resolved incarnation. Missing or conflicting creation evidence is rejected;
an exact pending creation retains pending delivery.

Delivery withdrawal changes only the peer receipt, not the creation decision.
It can settle a pending delivery as excluded. Creation may still finish, but
that exact delivery cannot append afterward. If append already won, exclusion
returns its appended receipt rather than relabeling the outcome.

The trusted receiving owner can discover pending requests with
`SessionStore.list_pending_peer_content(after_operation_key=..., limit=...)`.
Pages are ordered by operation key and limited to 64; the last returned key
continues the scan. This index is independent of creation-settlement discovery.
The host calls `CayuApp.service_pending_peer_content(session_id, context=...)`
to resolve these requests after creation, including after store reconstruction.
Each attempt passes through fresh registered export authorization held through
native admission. Ordinary runtime queue servicing never retries a historical
request without that authorization, and servicing does not launch a model.
Terminal target decisions remain owned by the destination creation fence and survive public-ID reuse.

Tests in `test_peer_creation_fence.py` exercise public FRESH/FORK creation
against peer append/exclusion on Memory, SQLite and PostgreSQL. Cancellation
and acknowledgement-loss tests retain the exact pending request and reopen
persistent stores before reconciliation. This does not authorize recipient
execution or introduce a second creation fence.
