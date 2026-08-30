# Memory foundation contracts

This document records Cayu's v5.1 long-term-memory foundations, immutable
knowledge revisions, and the bounded cross-source recall layer built on them.
Context composition, curation, admission, and exposure evidence remain separate
layers. Automatic recall now publishes its exact retrieval/admission receipt and
tracks every provider-facing use through the durable exposure lifecycle.

## Knowledge and memory are different layers

**Knowledge** is durable canonical semantic material: entries, chunks, source
identity, lifecycle, immutable revisions, and revision-bound source evidence.
**Memory** is the larger recall system that can retrieve permitted
knowledge, transcript episodes, artifact-derived documents, and other typed
sources, fuse them, select a bounded context contribution, and record exposure.

The current `KnowledgeStore` is therefore one memory source. `SessionStore`
transcript search is another. The WRRF types in `cayu.retrieval` are
source-neutral and do not turn transcripts or artifacts into knowledge.

## Durable agent work context and recall checkpoints

`AgentWorkContext` is the application's immutable, revisioned description of
what one stable task is currently about: its goal, scope identities, optional
workflow position, and bounded entity, artifact, repository-path, code-symbol,
and planned-action identities. It is application/runtime-owned state. It is not
prompt text, a model-authored plan, a `TaskStore` replacement, or another
workflow engine. Cayu never makes it provider-visible by itself.

`AgentWorkContextStore` deliberately remains a narrow independent boundary.
Its in-memory, SQLite, and PostgreSQL implementations retain exact historical
revisions and one indexed current pointer. Creation and append use
compare-and-swap; a stale concurrent writer cannot overwrite the winner.
Publication operation IDs provide exact replay receipts. Publishing
semantically identical content records a no-change receipt without inventing a
new revision. The API exposes exact current, revision, receipt, and checkpoint
reads only; it has no unbounded task-list operation. Storage revision 69 adds
only empty authoritative tables and does not infer, copy, or backfill context
from tasks, sessions, transcripts, or knowledge.

The PostgreSQL implementation accepts caller-owned pools only when they retain
the built-in transactional connection and cursor behavior. Autocommit pools,
behavior-changing callbacks, subclasses, and configuration drift fail before
mutation authority is acquired, preserving atomic history/head/receipt writes
and transaction-scoped advisory locks.

`AgentRecallCheckpoint` is a separate compare-and-swap record for one exact
agent, task, knowledge namespace, and access-policy fingerprint. It binds the
work-context revision and hash, the captured knowledge-change and
semantic-index-readiness high-water marks, and the last safely processed
sequence within each captured frontier. Its meaning is intentionally narrow:
the named processor completed that bounded freshness range against that exact
work context. It is not evidence of relevance, attention, provider exposure,
notification consumption, task completion, or side-effect authority.

For an unchanged work context, a checkpoint may advance monotonically through
the future delta-processing lane, but neither processed sequence may exceed
the high-water mark captured from its owning durable source, and captured
high-water marks never regress for the same checkpoint key. Advancement binds
only the task's current work-context revision and hash; PostgreSQL shares that
task fence among checkpoint writers and excludes concurrent context
publication while the binding is committed. A changed work-context hash
requires a full-index processing result before the checkpoint can move; an old
delta cursor is never silently inherited in a way that would make older
knowledge ineligible. Changing the access-policy fingerprint creates a distinct
checkpoint key. The compare-and-swap revision and processed sequences establish
ordering; `updated_at` is attributable event time and is not used as concurrency
authority because distributed worker clocks may skew.
Checkpoint advancement happens only after processing, so a crash or
cancellation beforehand leaves the prior frontier available for an idempotent
retry.

`AgentRecallProcessor` is the provider-neutral coordinator over those facts.
Given one exact agent, durable work context, namespace, access scope, prior
checkpoint, and `RecallSituation`, it selects one of three explicit paths:

- no checkpoint or a changed work context runs full-index knowledge recall;
- an unchanged context with newer source frontiers runs bounded delta recall;
- an unchanged context with no newer frontier returns an explicit no-work
  result without running retrieval.

The request requires an exact single-namespace `KnowledgeAccessScope` matching
the checkpoint namespace. A multi-namespace or privileged scope would couple
independent namespace frontiers and make unrelated changes trigger empty delta
work; callers must derive the narrow principal-authorized scope first.

Full-index recall is also evaluated inside the captured store frontier. Current
revisions materialized after its knowledge high-water mark, semantic projections
published after its readiness high-water mark, and relation lineage published
after the knowledge frontier are excluded from that result and remain eligible
for the next delta. Attached lineage resolves endpoint revision, lifecycle
status, and currentness at that same knowledge frontier while still requiring
live endpoint authorization, so a concurrent revision can neither leak future
metadata nor bypass a later access revocation. The processor does not
approximate this boundary with wall clock time or move the high-water read after
retrieval, either of which could cross or skip a concurrent commit.

Delta recall derives exact current revision references from only the bounded
change/readiness pages it processed. `KnowledgeStore.search_revisions()`
enforces that identity set together with the captured knowledge and readiness
frontiers inside in-memory, SQLite, and PostgreSQL ranking, including their
embedding variants. It never performs an unrestricted global top-k search and
filters afterward, so a relevant changed revision cannot be hidden by
higher-ranked unchanged records. The paired frontier also excludes semantic
readiness and attached relations published after capture. Its knowledge bound
uses the revision's latest materialization event, so deleting and recreating an
entry with the same ID and revision number cannot impersonate the captured
generation. Stale, retired, deleted, expired, or unauthorized referenced
revisions are omitted by the store's currentness and access boundary. A
relation-only change evaluates its canonical subject revision; a later endpoint
revision remains independently eligible through its own knowledge change.

Every processing result carries the exact work-context identity, captured and
safely processed frontiers, source events, deduplicated eligible revisions,
bounded recall diagnostics, and an optional `AgentRecallCheckpoint` proposal.
The processor never persists the proposal. A caller that will deliver the
materialized result must use `stage_recall_delivery()` so the exact result and
checkpoint compare-and-swap commit together; advancing that checkpoint first
would reopen the crash window this boundary closes. A truncated event page advances only
through its returned ordered prefix. If a retired entry makes an old readiness
event disappear from the current access view, the checkpoint's already proven
high-water mark remains the monotonic floor. A semantic timeout/failure leaves
the readiness cursor retryable even when the lexical knowledge cursor can
safely advance. A failed full-index semantic lane proposes no checkpoint, so a
new or changed work-context basis must retry the complete scan. A delta may
propose only lexical progress when an unfinished readiness frontier still
guarantees another attempt; if no such frontier remains, it likewise withholds
the proposal. Supplying the previously returned `AgentRecallFrontier` for an
unchanged-context delta keeps later observable source events outside that
retry. Exact currentness is deliberately reevaluated: if a referenced revision
became stale, it is omitted and its newer revision remains eligible through the
later change event. Durable replay of an already materialized delivery belongs
to the subsequent staging/delivery layer.

Operational failures from either required freshness read are normalized to an
`AgentRecallProcessingError` with a bounded change/readiness failure code and
the backend exception retained as its cause. Cancellation still propagates
unchanged, and no failure path constructs a checkpoint proposal.

This primitive does not wake agents, inject provider context, claim
`ContextExposure`, acknowledge notifications, or change the public knowledge
tools. Those remain separate scheduling, composition, exposure, and delivery
responsibilities.

### Atomic staged recall delivery

`AgentRecallDelivery` is the bounded, immutable handoff from checkpoint-aware
processing to a downstream delivery worker. It retains the exact serialized
`AgentRecallProcessingResult`, its fingerprint, the exact checkpoint and
checkpoint fingerprint, full-index or delta classification, work-context and
access authority, and staging attribution. Loading it materializes that frozen
result directly; it never re-runs retrieval or follows a newer, retired,
unauthorized, or deleted knowledge revision.

`AgentWorkContextStore.stage_recall_delivery()` commits the delivery and its
proposed checkpoint in one critical section or database transaction. The
delivery carries the explicit expected checkpoint revision. Exact replay returns
the existing record, while a reused delivery ID, operation ID, checkpoint
revision, stale checkpoint, or stale work context fails with a typed conflict.
If any checkpoint, delivery, or state write fails or is cancelled, neither the
stage nor checkpoint is visible.

The queue is scoped by the exact agent, task, namespace, and access-policy key.
`claim_recall_delivery()` returns only the oldest unacknowledged stage for that
key and uses an indexed, hard one-record lookup. A claim contains a worker ID,
attempt, state revision, and bounded lease. Renewal, explicit release for retry,
lease-expiry takeover, and acknowledgement all require the exact current claim;
stale workers cannot release or acknowledge a newer attempt. Claim and release
identities remain reserved after release or takeover so retries cannot silently
acquire a second identity. An exact retry converges only while its original
attempt remains retained, and it can return a claimed state only while that
lease remains live. Expired claimed or renewed replays fail with an expired
conflict; after takeover, claim and release replays fail with a typed superseded
conflict instead of returning the newer worker's claim authority.

Lease time is owned by the store clock. A new stage cannot claim a future
staging time, and release or acknowledgement evidence cannot claim a future
transition time. Caller attribution therefore cannot move the delivery clock
forward, extend a configured lease, or pin the oldest pending stage beyond its
bounded lease.

An acknowledgement records only that a durable downstream boundary accepted the
stage. Its typed reference may point to a `RecallReceipt`, `ContextExposure`, or
application handoff, but it does not create that evidence or claim provider
visibility, model attention, notification consumption, or task completion.
Provider composition and exposure transitions remain owned by their existing
runtime contracts.

Storage revision 71 is a clean prerelease break. It installs only empty
delivery, state, claim, and release tables beside the checkpoint authority. It
does not infer or backfill deliveries from existing checkpoints, knowledge,
sessions, transcripts, receipts, or exposures, and there is no legacy read,
dual-write, or compatibility path.

### Idle recall subscriptions

Active runtimes continue to call `AgentRecallProcessor` inline. An idle task can
instead publish an immutable `AgentRecallSubscription` that binds one exact
agent, task, namespace, access-policy fingerprint, current work-context
revision/hash, admission policy, bounded query/facets, priority, minimum
interval, expiry, and active/paused/cancelled status. Updates are immutable
revisions guarded by compare-and-swap. Every declared facet must already belong
to the bound work context, so a subscription can narrow that task's retrieval
input but cannot silently acquire unrelated scope.

Each subscription owns an independent checkpoint stream derived from its
subscription ID and exact retrieval-shaping input. Two subscriptions over the
same agent, task, namespace, and access policy therefore observe the same
change frontier without consuming one another's progress. Schedule- or
policy-only revisions retain the cursor, while changing the query or facets
starts a fresh full-index stream so newly matching older knowledge is not
skipped. A scheduler must pass the claimed subscription's checkpoint key and
stream ID through processing and commit unchanged.

Facets are exact indexed filters, not extra search prose. Add the corresponding
`agent_recall_facet_aspect(field_name, value)` tokens to a knowledge entry's
`aspects` when publishing it. Values within one facet category are ORed, while
different categories are ANDed. A subscription without `query` performs a
bounded metadata-only lookup over those exact groups; unrelated text cannot
produce a match.

No background service is required. An existing application scheduler can claim
the oldest due subscription for the agent/task/namespace/access authority of a
checkpoint key, build its single-namespace situation with
`subscription.recall_situation()`, and then load and process the claimed
subscription's exact stream key. Claims use
store-clock leases, exact replay, renewal, release, expiry takeover, and stale
runner fencing. Paused, cancelled, expired, stale-context, unauthorized, or
already-waiting subscriptions are not due. Priority breaks ties only after the
oldest due time; it does not bypass scope or timing authority.

`commit_recall_subscription_evaluation()` has two successful material outcomes:

- weak or empty admitted material commits the successor checkpoint and an
  immutable silent evaluation receipt, without a delivery or wake;
- relevant non-empty material atomically commits the successor checkpoint,
  exact staged `AgentRecallDelivery`, immutable evaluation receipt, and one
  pending `AgentRecallSubscriptionWake`.

Failure or cancellation publishes none of those records. A pending wake
coalesces another evaluation for that subscription, while later knowledge and
index-readiness frontiers remain eligible after the wake is accepted. The
committed delivery is authoritative; wake recovery never re-runs retrieval to
reconstruct its payload.

The wake has its own scheduler claim/release/acknowledgement lifecycle. Wake
acknowledgement means only that the application scheduler durably accepted a
request to revisit the task. It makes the already staged delivery available to
the normal delivery lifecycle, but does not claim or acknowledge that delivery,
mutate a `TaskStore` record, dispatch a provider call, create a `RecallReceipt`
or `ContextExposure`, assert model attention, or consume a notification. The
runtime must later claim and acknowledge the delivery against its actual
downstream evidence boundary.

Storage revision 73 is a clean prerelease break because processing results now
bind their exact retrieval-shaping input and every subscription has an isolated
checkpoint stream. Every staged delivery must project that processing-result
schema version into a constrained column, so a process running the older
contract cannot insert an unreadable result after migration. Revision 73 is for
fresh schemas only: databases with the older checkpoint shape are rejected and
must be recreated. It installs empty subscription revision/head/publication,
runner claim/release/state, evaluation, and scheduler-wake claim/release/state
tables with bounded indexes. It does not infer or backfill records from tasks,
contexts, checkpoints, deliveries, knowledge, sessions, receipts, exposures,
or notifications, and it adds no compatibility or dual-write path.

The nearby context concepts have different lifetimes and owners:

- `RecallSituation` is ephemeral caller input for one retrieval boundary. Its
  optional `work_context` text may be derived from a durable
  `AgentWorkContext`, but the two are not interchangeable.
- A provider-facing memory focus or future memory delta is a bounded context
  composition outcome. The runtime context composer alone decides its content
  and placement; the durable work-context store does not inject either.
- `TaskStore` and application workflow state retain scheduling, lifecycle, and
  authority. Work-context workflow fields are descriptive foreign identities,
  not control-plane commands.
- `RecallReceipt` records retrieval and admission, while `ContextExposure`
  records what reached a provider boundary. Neither fact can be inferred from
  a recall checkpoint.
- A future notification consumer must keep its own delivery/consumption
  evidence. Advancing a checkpoint does not acknowledge a notification.

The checked [agent work-context performance baseline](../benchmarks/memory/agent-work-context-performance-v1.json)
measures current-runtime zero-record store construction, indexed current reads,
CAS revision appends, CAS checkpoint advances, and incremental SQLite storage
without provider calls. Its runner and fixed regression ceilings are documented
with the other hermetic memory benchmarks.

The checked [checkpoint-recall performance baseline](../benchmarks/memory/checkpoint-recall-performance-v1.json)
measures full-index, one-revision delta, maximum 250-reference delta (249
knowledge changes plus one ready index projection), and no-work processing over
500 existing records with 50 samples per in-memory and SQLite backend. It makes
no provider calls. PostgreSQL and pgvector use behavioral integration tests,
including a query-plan regression proving that frontier-filtered semantic recall
retains the partial HNSW index; exact scanning remains the bounded-revision,
negative-filter, oversized-vector, and filtered-underfill fallback.

The checked [staged-recall delivery performance baseline](../benchmarks/memory/recall-delivery-performance-v1.json)
measures 50 atomic stage/checkpoint commits followed by deterministic claims and
acknowledgements, plus the indexed no-pending path after all 50 records are
terminal. It records fixed p50 and p95 ceilings for in-memory and SQLite stores,
makes no provider calls, and keeps PostgreSQL performance outside the hermetic
credential-free matrix while exercising the same behavior in integration tests.

The checked [idle recall-subscription performance baseline](../benchmarks/memory/recall-subscription-performance-v1.json)
measures 50 indexed zero-due claims, silent evaluation commits, atomic relevant
wake publications, scheduler claims, and scheduler acknowledgements for both
in-memory and SQLite stores. It makes no provider calls and applies fixed p50
and p95 ceilings; PostgreSQL exercises the same lifecycle through shared
conformance, concurrency, migration, reopen, and cancellation tests.

## Bounded cross-source recall

`RecallEngine` runs registered `RecallSource` adapters concurrently under
per-source and overall deadlines, validates their independently ranked lanes,
and fuses them with a caller-versioned `RetrievalFusionStrategy`. The built-in
`KnowledgeRecallSource` contributes separate lexical and semantic lanes;
`TranscriptRecallSource` contributes a narrative transcript lane. Every source,
channel, candidate list, record representation, and combined result has an
explicit count or byte ceiling.

The lexical knowledge lane is authoritative for the built-in knowledge source.
Its optional semantic lane runs concurrently under its own deadline; an
unsupported, failed, or timed-out semantic lookup is recorded as partial lane
coverage without discarding successful lexical evidence. A source-level lexical
failure still follows the source's required/optional policy.

`RecallEngineConfig.max_result_bytes` bounds the canonical serialized
`RecallResult`, including diagnostics and continuations rather than only the
candidate text. If the metadata-only result cannot fit, recall fails closed.
Sources declare which of their channels accept continuation cursors; supplying
a cursor for a non-pageable channel is rejected instead of silently ignored.
Pageable sources provide the exact frontier after every ranked hit. When fusion
or the result-byte ceiling omits candidates, recall advances a channel only
through its contiguous ranked prefix whose candidates are actually returned.
An omitted candidate is therefore eligible on the next page rather than being
silently skipped.

Callers construct an immutable `RecallSituation` with the current query,
optional bounded recent conversation or work context, an explicit
`KnowledgeAccessScope`, a knowledge namespace, and the exact session IDs that
may be searched. Knowledge filters execute inside `KnowledgeStore`. Transcript
filters execute inside `SessionStore`; missing and inaccessible-by-omission
session IDs are indistinguishable to the search result. Recall never discovers
tenant/session authority or widens the caller's scope.

`RecallResult` contains deterministic fused candidates, exact canonical
identity and revision, a bounded representation, content hash, source-specific
locator, fusion diagnostics, and per-source coverage state. Unsupported or
temporarily omitted optional lanes are not described as complete. A required
source failure fails the recall request; an optional source can be represented
as unavailable without inventing empty complete coverage.

`KnowledgeRecallSource(..., lineage_limit=N)` can additionally attach a
`KnowledgeLineageResult` to the highest-ranked bounded set of exact knowledge
records. `lineage_limit=0` is the default, so applications that do not request
lineage pay no inspection cost. Enrichment never changes a lexical or semantic
rank. It exposes typed exact-revision references: `supersedes`,
`superseded_by`, `derived_from`, `derivation_source_for`, or `contradicts`.
An unresolved current contradiction identifies the other active revision as an
explicit alternative; it never labels either side as the winner. Archived
predecessor text is not recalled or copied into the lineage projection.
Before attachment, the recall adapter revalidates the store page against the
exact bounded query and requires the recalled revision to remain current and
active in the lineage snapshot. An intervening advance fails the source closed
instead of returning a candidate and explanation from conflicting lifecycle
snapshots.
Recall caps enrichment at 100 records, 100 links per record, 64,000 bytes per
lineage page, and a 1 MB configured aggregate reservation. Repeated chunk
records for the same exact revision share one store inspection but each consume a record
slot and only selected records receive the projection. Lower constructor bounds
are recommended for interactive paths; the defaults select at most 10 records
and reserve 8,192 bytes per selected record.

Every fusion implementation has an immutable strategy identity which must match
the caller's versioned fusion configuration and the returned diagnostics. A
custom strategy may rank differently, but cannot claim the built-in WRRF
identity or alter source evidence, channel coverage, or provenance.

Transcript search indexes only `TextPart` content from user and assistant
messages. Thinking, tool calls/results, provider state, and system messages are
not searchable or returned. A canonical case-folded document uses collision-free
hex identities for ordinary terms and fixed-size SHA-256 identities for long
terms to preserve Python word boundaries across the in-memory, SQLite FTS5, and
PostgreSQL GIN-backed implementations. Phrase adjacency and distinct-term
coverage outrank bounded term frequency. SQLite and PostgreSQL intersect opaque
session terms inside their full-text indexes, while memory addresses only the
selected session posting lists. All three use the same explicit scope,
relevance order, pagination, byte, and scan contract. Exceeding the scan
ceiling yields no fabricated partial ranking or continuation; coverage is
reported incomplete so a caller can deliberately raise the bound and retry.
The index version includes the runtime Unicode database used by Python's word
matching and case folding. Durable SQLite and PostgreSQL stores persist that
tokenizer identity and fail startup on a mismatch; operators must use a clean
revision-46 database rather than mixing index writers with different Unicode
semantics. Cayu does not rewrite existing transcript rows or repair this
mismatch.

SQLite session-store reads return caller cancellation and recall deadlines
promptly. Because Python cannot stop an already-running SQLite worker thread
safely, the physical read remains fenced behind its connection lock until it
settles; later reads and shutdown cannot reuse that connection in the meantime.

Recall is retrieval only. `AutomaticRecallPolicy` is a separate calibrated
admission primitive that classifies the fused head as strong memory focus,
reference-only offers, or silent candidates under explicit count and byte
bounds. `AutomaticRecallContextPolicy` runs that boundary once for a real user
interaction, freezes the redacted provider-neutral contribution through retries,
tool rounds, repair, compaction, and recovery, and expires it at the next real
user message. It never mutates the durable transcript.

This first context manager does not claim semantic awareness of everything
already stated in the provider-visible context and does not reposition material
around a guessed lost-in-the-middle region. Those remain future composition and
exposure-evidence decisions. The credential-free
[cross-source example](../examples/cross_source_recall.py) shows retrieval and
admission as explicit separate steps.

## Recall receipts and context-exposure evidence

`RecallReceipt` is an immutable, bounded record of one already-computed recall
and admission operation. It binds the situation, engine/configuration/policy,
access-scope fingerprint, source frontier, honest source coverage, exhaustive
outcome counts, and every admitted or offered representation by canonical
identity, revision, representation ID, content hash, locator, reason, and
fusion rank. It does not store query text, recalled text, prompt text, or
provider payloads. An unavailable or failed source is explicitly incomplete;
it cannot masquerade as an empty complete search.
Locators are a closed union for exact knowledge-entry revisions, knowledge
chunks, and transcript messages. Custom sources retain only a domain-separated
keyed fingerprint of their canonical locator, never the arbitrary locator
object itself. This keeps replay identity exact without turning the evidence
store into a path for credentials, queries, or source-specific private fields.
Private situation, access-scope, configuration, frontier, composition, profile,
policy, tool, and request material is bound through `KeyedEvidenceFingerprint`:
a domain-separated HMAC-SHA-256 digest plus a non-secret key ID. Raw SHA-256 of
low-entropy private material is rejected as a receipt/exposure field because it
would create an offline guessing oracle; HMAC keys never enter the record.

A receipt proves what retrieval inspected and selected. It does **not** prove
that a provider saw the material. `ContextExposure` records that separate fact
as a compare-and-swap lifecycle for one exact model and provider attempt:

- `planned`: context composition selected exact contributors and receipt IDs;
- `prepared`: the provider-neutral request was prepared;
- `dispatch_started`: durable dispatch intent was committed before network I/O;
- `acknowledged`: provider or recovery evidence establishes acknowledgement;
- `completed`: provider or recovery evidence establishes completion; and
- `failed`, `cancelled`, or `indeterminate`: conclusive or explicitly ambiguous
  terminal evidence prevents a guessed success.

Only `acknowledged` or `completed` is positive provider-exposure evidence.
`planned`, `prepared`, and `dispatch_started` must never be counted as “the
model saw this.” A provider request ID is retained only when the adapter can
expose one safely; acknowledgement can still be proven by a bounded internal
evidence reference when no such ID is available.

`RecallItemExposure` links each exact recalled representation in the planned
composition back to its immutable receipt item. Stores reject altered hashes,
locators, identities, duplicate receipt items, cross-interaction links, and
mixed receipt/exposure key identities, or reused model/provider attempt identities.
A receipt records the model step where
recall ran; structured-output repair and other later model steps in the same
interaction may link that frozen receipt through their own distinct exposures.
Lifecycle transitions carry an exact
expected state and revision. Concurrent writers therefore produce one winner
and a typed conflict instead of a lost update; replaying the same transition ID
returns the current durable record.

The `SessionStore` evidence surface provides exact create/load, bounded
session-scoped keyset pages, item lookup, and fenced transition methods.
In-memory, SQLite, and PostgreSQL implementations share one conformance suite.
Every record and page has item-count and serialized-byte bounds, cursors are
bound to their query scope, and page reads use bounded lookahead rather than a
full remaining-row count. Deleting the owning session cascades its receipts,
exposures, and item links. Custom stores advertise the complete surface with
`supports_recall_evidence = True`.

`AutomaticRecallContextPolicy` builds its receipt from the same frozen
`RecallResult` used for admission; it does not rerun retrieval or reconstruct a
frontier later. Receipt persistence succeeds before the automatic-recall
checkpoint can authorize context composition. The checkpoint binds the receipt
ID and exact durable receipt-document digest to the exact rendered manifest
with a purpose-separated HMAC, so substitution fails closed at the provider
boundary without persisting the key.
After context selection,
compaction, pressure handling, tool/structured-output preparation, and provider
request construction have settled, Cayu fingerprints that exact final
`ModelRequest`, creates the attempt's planned/prepared exposure and item links,
and includes the exposure identity in the durable model-completion stage.
Recall items removed completely by later context selection receive no item link;
an altered automatic-memory envelope fails closed instead of being misreported.
`dispatch_started` is committed inside the final pre-network fence sequence,
before the budget dispatch fence. The exact model-stage dispatch receipt remains
the last local durable fence before provider-controlled code. If any required
receipt, exposure, item-link, or transition write fails, the provider is not
called and budget reservations have not been marked dispatched. When optional
provider-backed input counting is enabled, a memory-bearing request crosses this
same fence before the counter receives it and the later model call reuses the
exact prepared dispatch. Recovery closes
an exposure as failed when that final model-stage receipt is absent; a
receipt-bearing synchronous attempt remains conservatively ambiguous.

The first normalized non-error provider response establishes acknowledgement. A
normalized completion records `completed`; an error-only response records `failed`
without claiming positive exposure, while an error after output preserves the
earlier acknowledgement in its history. Typed authentication
rejections and context-overflow exceptions end that attempt as `failed` even
when the adapter raises them before yielding a frame; cancellation and transport
paths record cancellation only when adapter evidence proves it and otherwise
finish as `indeterminate`. Generic retries get distinct model,
provider, and exposure attempt identities while retaining the same composition
fingerprint. Context-overflow recovery rebuilds and fingerprints the replacement
request. Durable background-operation recovery advances the original exposure
with recovery acknowledgement/completion evidence. A local connection or worker
loss after the durable operation acknowledgement leaves that exposure open for
recovery; explicit unavailable recovery closes it as indeterminate. An unrecoverable
synchronous dispatch becomes indeterminate and requires the runtime-owned
`CayuApp.recover_model_completion_stage(...)` operation.

Automatic recall therefore requires a recall-evidence-capable `SessionStore`
and a keyed `RequestFootprintConfig` (`fingerprint_key_id` plus
`fingerprint_key`). Cayu derives a purpose-separated in-memory HMAC key for
memory evidence; the configured secret and derived key never enter checkpoints,
events, receipts, or exposures. Missing evidence capability or key material
fails context construction before recall or provider dispatch. `off` mode does
not require either capability.

Storage revision 51 only creates empty evidence tables and indexes. It does not
scan runtime history, synthesize receipts for past recall, or carry a legacy
reader/writer path. Existing sessions remain unchanged; evidence exists only
for operations that explicitly publish it through this contract.

## Bounded public memory attribution

`runtime_evidence(...)` projects receipts, context exposures, item links, and exposure
lifecycle into the versioned `cayu.memory_attribution.v1` contract. Consumers do not
scan raw events or the private store. `trajectory_from_session(...)` promotes the same
typed section through a read-only path; it does not run application, provider, tool,
environment, hook, or recovery behavior.

Projection completeness and provider exposure are deliberately separate. The section
reports `complete`, `truncated`, `unavailable`, `redacted`, or `contradictory`, while
each exposure retains its own lifecycle, including `indeterminate`. A missing row that
was bounded away is never reported as complete empty success. Count, source-byte, and
projection-byte bounds are global to the report or trajectory rather than multiplied by
session count, and `*_count_at_least` fields state what is known without an unbounded
count query.

Portable receipt, exposure, interaction, and item identities are session-scoped,
domain-separated HMAC aliases. The projection never emits memory or prompt text,
queries, transcript/provider bodies, credentials, embeddings, arbitrary metadata or
source names, raw locators, raw content hashes, or hidden reasoning. If alias key
authority is unavailable while rows exist, the result is explicitly `redacted` and the
private rows are not released. A broken receipt/exposure/item relationship is
`contradictory` and likewise releases no projected records.

The hermetic performance runner separates document preparation, evidence persistence,
steady-state SQLite storage, empty-report projection, memory-bearing projection, and
serialized projection size. It makes no provider calls:

```bash
PYTHONPATH=src python scripts/run_memory_evidence_performance.py --check
```

The checked 50-pair baseline is
[`benchmarks/memory/memory-evidence-performance-v1.json`](../benchmarks/memory/memory-evidence-performance-v1.json).
The zero-record `runtime_evidence(...)` run is a same-process current-runtime control,
not a historical pre-feature measurement. It exposes the always-on cost paid by a
session with no receipt or exposure rows; the populated-minus-control measurement then
isolates the incremental cost of projecting 50 pairs. Regression checks require
zero-record p95 at or below 5 ms in memory and 10 ms on SQLite, preparation p95 at or
below 10 ms, in-memory and SQLite persistence p95 at or below 15 ms and 25 ms
respectively, p95 incremental projection overhead at or below 10 ms per pair,
projected size at or below 6,000 bytes per pair, and steady-state SQLite storage at or
below 32 KiB per pair. The normal test suite validates the checked artifact and every
regression lane. Run the hermetic command above to measure current code before accepting
performance-sensitive changes.
The absolute ceilings include headroom for shared-runner scheduling variance while still
bounding the complete 50-pair workload. Latency is environment-sensitive; the artifact
records its Python/platform identity, p50/p95 observations, zero-record control, and
absolute overhead without mixing provider latency into the result.

## Storage-enforced access

Every built-in knowledge operation requires a `KnowledgeAccessScope`. A store
may bind one scope at construction for a single trusted principal, or a caller
must pass `access_scope=` on each operation. Omission never means global access;
trusted maintenance uses the explicit `KnowledgeAccessScope.privileged()`.

The application—not Cayu—maps authentication, users, organizations, and RBAC to
generic constraints:

- allowed namespaces;
- required labels;
- allowed visibility classes;
- optional allowed source types and source identities;
- allowed lifecycle states; and
- expired-material eligibility.

Query filters can narrow those constraints but cannot widen them. Stores apply
the scope in typed ID reads, chunk reads, list/search candidate selection,
writes, lifecycle changes, deletion, publication receipts, review/indexing, and
embedding jobs. Inaccessible ID reads return no entry/chunks so they do not
become an existence oracle; an attempted inaccessible mutation raises
`KnowledgeAccessDenied`. Chunk identities are global storage identities: every
chunk-writing path authorizes an existing chunk's owning entry before reporting
occupancy. A foreign-scope collision therefore raises `KnowledgeAccessDenied`,
while an authorized collision raises the backend-independent
`KnowledgeChunkConflict` without leaving a partial entry, chunk set, or receipt.

An exact historical entry or chunk read must satisfy the scope twice: against
the requested immutable snapshot and against the logical entry's current
revision. Archiving, deleting, expiring, relabeling, or otherwise restricting a
current revision therefore revokes ordinary access to its history; only an
explicit scope that can read both states can audit it. Mutation authorization is
slightly different by design. A principal that can mutate the current revision
may append an `archived` or `deleted` successor without gaining read access to
that retirement state. Promotion or reactivation into any other state still
requires the destination state and every other destination attribute to satisfy
the scope.

```python
from cayu import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeQuery,
    KnowledgeVisibility,
)

scope = KnowledgeAccessScope.for_namespace(
    "acme-support",
    required_labels={"project": "billing"},
    allowed_visibilities=[KnowledgeVisibility.PROJECT],
    allowed_source_types=["document"],
)
store = InMemoryKnowledgeStore(access_scope=scope)
result = await store.search(
    KnowledgeQuery(text="refund approval", namespace="acme-support")
)
```

Runtime environments and `CayuApp` defensively copy the application scope and
carry it to model tools and automatic knowledge context. Administrative review
routes require the app's explicit scope as well.

Publication receipts retain an immutable authorization projection, so an exact
idempotency replay remains available after hard deletion without depending on a
current entry or leaking a different namespace. Database revision 41 is a clean
break for that projection; existing populated receipt tables are not guessed or
backfilled.

## Revision-bound evidence and atomic changes

`KnowledgeEvidence` records why one exact knowledge revision exists. Evidence
is immutable, has a global ID, belongs to an exact entry revision, and may bind
to one exact chunk from that revision. Its source identity is deliberately
generic: a source type plus an ID or URI, a source revision or hash, a durable
locator, an `origin` or `supporting` role, and a `live`, `detached`, or
`retained` disposition. Locator and metadata objects are bounded durable JSON;
they do not become provider-specific columns.

Create, append, and owned publication accept `evidence=` and commit the entry,
chunks, evidence, and metadata-only `KnowledgeChange` atomically. Lifecycle-only
successors (status transitions and tombstones) inherit evidence under new
deterministic evidence IDs and rebind chunk evidence by chunk index. A caller
that materially changes content must supply evidence explicitly; omission means
the new revision has no evidence. Exact publication replay writes nothing, and
the publication request digest covers evidence as well as the entry and chunks.
The historical revision-42-to-43 transition preserved exact empty-evidence
receipt replay. Revision 60 supersedes that upgrade posture: a populated
pre-60 knowledge schema is deliberately not migrated or interpreted.

`read_evidence(...)` applies the same current-plus-historical authorization rule
as entry and chunk reads and returns a record/byte-bounded result. Evidence IDs
are global storage identities. An authorized collision raises
`KnowledgeEvidenceConflict`; a foreign-scope collision raises
`KnowledgeAccessDenied` without revealing its owner.

Every successful canonical mutation publishes one ordered `KnowledgeChange` in
the same transaction: create, content revision append, lifecycle transition,
tombstone, hard delete, or expiration. Changes contain identity, revision,
kind, operation ID, sequence, and commit time only—never knowledge text,
evidence locators, or source payloads. Authorization uses immutable before/after
audiences captured with the mutation. A scope that loses access receives the
change so it can remove stale derived state, while a scope that gains access can
materialize the successor. Creation has only an after audience; hard deletion
and expiration have only a before audience. Expiration eligibility is frozen at
publication, and removal signals remain visible to scopes that could have
materialized the entry before expiry, so delayed consumers do not silently lose
cleanup work. For revision-42 entries without a publication audience, revision
43's durable migration timestamp supplies that visibility baseline without
fabricating historical changes.

`read_changes(...)` returns bounded sequence pages with at most
`MAX_KNOWLEDGE_CHANGE_LIMIT` records and an accessible
high-water mark that always comes from committed store state, never from the
caller cursor. A cursor beyond the store's current sequence is rejected; a
cursor may legitimately exceed one scope's accessible high-water, in which case
continuation does not move backwards. After a full scan,
`initialize_change_consumer(...)` binds a
new consumer cursor to the high-water captured before that scan; the operation
is idempotent but cannot reset an active or already-started consumer.
Scope-bound consumers then use `claim_change`, `acknowledge_change`, and
`release_change` for fenced, leased, at-least-once delivery. A consumer ID is
permanently bound to the canonical digest of its access scope; concurrent
workers cannot acknowledge one another's claims, expired or stale claims fail
closed, and exact acknowledgement replays remain idempotent after later cursor
progress. Lease eligibility uses a store-owned clock; PostgreSQL uses its
database clock in production so worker clock skew cannot extend or revive a
claim. SQLite/PostgreSQL cursor and acknowledgement state survives restart.
This change stream is canonical mutation publication, not derived-index
readiness.

## Revision-bound cross-entry relations

Four kinds of lineage remain deliberately separate:

- consecutive revisions of one logical `KnowledgeEntry` already have implicit
  same-entry lineage through their revision numbers;
- `KnowledgeEvidence` explains which source material supports one exact
  revision;
- `KnowledgeRelation` records a reviewed semantic statement between exact
  revisions of two different logical entries; and
- recall receipts and `ContextExposure` record retrieval, selection, and actual
  provider-facing use. A relation does not mean that the model saw either entry.

`KnowledgeRevisionRef` is the exact `(entry_id, revision)` endpoint. The closed
`KnowledgeRelationKind` vocabulary is intentionally small: a replacement
`supersedes` its predecessor, a derived record `derived_from` its source, and
`contradicts` is symmetric. Contradictions are stored in one canonical endpoint
order, so concurrent opposite-order publications converge on one semantic
identity. Relations cannot connect two revisions of the same logical entry;
ordinary revision order already represents that case.

```python
from cayu import (
    KnowledgeRelation,
    KnowledgeRelationKind,
    KnowledgeRelationQuery,
    KnowledgeRevisionRef,
)

relation = KnowledgeRelation(
    id="reviewed-replacement-2026-08",
    subject=KnowledgeRevisionRef(entry_id="refund-policy-new", revision=2),
    object=KnowledgeRevisionRef(entry_id="refund-policy-old", revision=4),
    kind=KnowledgeRelationKind.SUPERSEDES,
    created_by="policy-reviewer",
    policy_id="knowledge-maintenance-v1",
)
receipt = await store.publish_relations(
    [relation],
    operation_id="reviewed-replacement-operation",
)
page = await store.read_relations(
    KnowledgeRelationQuery(reference=relation.subject, limit=100)
)
```

`publish_relations(...)` copies and canonicalizes at most 100 records, requires
both exact endpoints and both current logical entries to be authorized, and
commits the relations, one metadata-only `relation_published` change per
relation, and one immutable operation receipt atomically. Exact operation replay
returns the original receipt without writing again. Reusing an operation ID,
relation ID, or semantic tuple for different material fails closed. Relation
changes carry only the subject revision, relation ID, operation ID, sequence,
and commit time; their four immutable endpoint authorities must all authorize
an outbox reader through the exact and publication-time current state of each
endpoint.

`read_relations(...)` is a count- and byte-bounded incoming, outgoing, or
both-direction exact-revision lookup. Stable `(created_at, id)` pagination binds
its cursor to the reference, direction, kind filter, and access scope. Endpoint
authorization is performed in the backend query before hydration. Advancing an
entry never retargets an existing relation. Hard deletion or expiry removes
relations whose exact endpoints no longer exist, while immutable publication
receipts and already-published metadata-only changes remain reconciliation
evidence and never recreate the relation on replay.

`inspect_lineage(...)` is the safer read surface for recall and operator
explanations. It projects only relation identity, semantic role, the exact and
current counterpart references, lifecycle status, current/stale state, the
unresolved-contradiction marker, and creation time. It never returns knowledge
text, relation metadata, actor identity, policy payload, or evidence locators.
Every relation, currentness, counterpart-lifecycle, and unresolved filter is
bound into the stable cursor with the access scope.

Reviewed archival is special only at this reference boundary. An active exact
predecessor may be named after its logical current revision becomes `archived`,
which lets a selected replacement explain what it superseded. Ordinary entry,
chunk, evidence, and raw relation reads remain subject to their existing exact
plus current authorization and therefore do not release archived content. A
deleted, expired, relabeled, foreign-source, or foreign-namespace endpoint is
not widened. The in-memory, SQLite, and PostgreSQL implementations evaluate the
projection in one bounded store snapshot.

```python
from cayu import KnowledgeLineageQuery, KnowledgeRevisionRef

lineage = await store.inspect_lineage(
    KnowledgeLineageQuery(
        reference=KnowledgeRevisionRef(entry_id="refund-policy-new", revision=2),
        limit=20,
    )
)
for link in lineage.links if lineage is not None else ():
    print(link.role, link.counterpart, link.currentness)
```

A relation is not truth, similarity, approval, current-state activation,
ranking weight, or context-placement authority. This slice performs no graph
traversal. Lineage-aware recall reports reviewed relations but cannot create,
approve, activate, archive, rerank, or place knowledge. `ContextExposure`
continues to attribute actual provider-facing use from the unchanged exact
recall locator; a lineage reference is not exposure evidence.

Breaking storage revision 60 installs this final prerelease relation contract.
Fresh databases and completely empty earlier knowledge schemas initialize
directly. Any populated pre-60 canonical, evidence, receipt, outbox, readiness,
embedding, or relation table makes migration fail before DDL. There is no
backfill, metadata fallback, dual write, or legacy relation interpretation.

The hermetic relation performance checker makes no provider calls:

```bash
PYTHONPATH=src python scripts/run_knowledge_relation_performance.py --check
```

Its checked report returns 50 matching relations and safe lineage links from a
store containing 5,000 additional unrelated relations, so both endpoint lookups
must remain index-bound. The
result is recorded in
[`benchmarks/memory/knowledge-relation-performance-v1.json`](../benchmarks/memory/knowledge-relation-performance-v1.json).
The zero-relation lane is a current-runtime control with the same canonical
entries, not a historical binary comparison.

## Atomic reviewed knowledge maintenance

Revision-bound relations describe lineage but do not change lifecycle by
themselves. `KnowledgeMaintenanceProposal` and `KnowledgeMaintenanceDecision`
provide the separate reviewed authority needed to activate one pending
replacement, archive exact superseded predecessors, and publish the approved
relations as one atomic operation.

A proposal is an immutable, bounded plan. It binds:

- one exact pending replacement revision;
- up to 50 exact current source revisions, each with exactly one `supersedes`,
  `derived_from`, or `contradicts` disposition;
- the relations that bind the replacement's deterministic active successor
  revision (as the subject for directed dispositions, or either endpoint for a
  symmetric contradiction);
- the complete access scope and policy identity used for review; and
- bounded rationale, evidence summary, metadata, proposer identity, timestamp,
  and a canonical fingerprint.

The proposal is supplied by application policy or a future candidate-discovery
component. The store does not search for duplicates, invoke a curator, infer
supersession from similarity, or let model output authorize the mutation. A
decision binds the exact proposal fingerprint and requires a non-model reviewer
identity, safe reason, timestamp, and immutable operation ID.

```python
proposal = KnowledgeMaintenanceProposal(
    id="refund-policy-maintenance-2026-08",
    replacement=KnowledgeRevisionRef(entry_id="refund-policy-new", revision=1),
    sources=[KnowledgeRevisionRef(entry_id="refund-policy-old", revision=3)],
    relations=[
        KnowledgeRelation(
            id="refund-policy-supersession-2026-08",
            subject=KnowledgeRevisionRef(entry_id="refund-policy-new", revision=2),
            object=KnowledgeRevisionRef(entry_id="refund-policy-old", revision=3),
            kind=KnowledgeRelationKind.SUPERSEDES,
            policy_id="knowledge-maintenance-v1",
        )
    ],
    access_scope=review_scope,
    policy_id="knowledge-maintenance-v1",
    rationale="The pending revision replaces the reviewed policy.",
    evidence_summary="The application verified the signed policy artifact.",
)
decision = KnowledgeMaintenanceDecision(
    operation_id="refund-policy-maintenance-operation-2026-08",
    proposal_id=proposal.id,
    proposal_fingerprint=proposal.fingerprint,
    kind=KnowledgeMaintenanceDecisionKind.APPROVE,
    reviewer_type=KnowledgeActorType.USER,
    reviewer="policy-owner",
    reason="The exact proposal and evidence were reviewed.",
)
receipt = await review_workflow.decide_maintenance(proposal, decision)
```

Inside one store transaction, approval reauthorizes every exact reference,
checks that the replacement is still pending and every source is still active
at the reviewed revision, appends the active replacement revision, appends
archived revisions only for `supersedes` sources, publishes every relation,
appends the metadata-only lifecycle and relation changes, and persists the
proposal, decision, and receipt. Any stale revision, authorization denial,
identity conflict, failure, or cancellation leaves all of those records
unchanged. Exact replay returns the original receipt with `replayed=True`.
Maintenance history and exact replay remain authorized by the immutable
pre-transition entries that the reviewer inspected. Archiving a source outside
the review scope's readable statuses therefore neither hides that history nor
grants the reviewer read access to the archived revision.

`derived_from` and `contradicts` sources remain active. In particular,
contradiction records an unresolved reviewed conflict without silently choosing
a winner. Rejection persists immutable review history but does not activate,
archive, relate, or delete any entry; it leaves entry lifecycle unchanged. Any
additional lifecycle authority depends on how the application created and owns
that pending content.

Normal active recall cannot see the pending replacement before approval. After
approval it sees the replacement only after the predecessor lifecycle and
lineage are committed with it. This storage operation does not rank relations,
run graph traversal, or decide what retrieved memory enters model context;
recall policy and `ContextExposure` remain separate later boundaries.

Breaking storage revision 63 installs the final reviewed-maintenance record.
Fresh databases and completely empty earlier knowledge schemas initialize
directly. A populated pre-63 knowledge schema fails before any DDL or data
change and must be explicitly replaced. There is no backfill, legacy proposal
interpretation, compatibility wrapper, or dual-write path.

The hermetic performance gate runs with no provider calls:

```bash
PYTHONPATH=src python scripts/run_knowledge_maintenance_performance.py --check
```

Its checked workload applies 20 decisions with 20 sources each and compares
them with a current-runtime zero-decision control containing the same 420
entries. It measures preparation, atomic application, exact replay, receipt
loading, and incremental SQLite storage. Results are recorded in
[`benchmarks/memory/knowledge-maintenance-performance-v1.json`](../benchmarks/memory/knowledge-maintenance-performance-v1.json).

## Deterministic maintenance candidate routing

`KnowledgeMaintenanceRouter` is the read-only boundary between explicit
application-produced hints and the later consolidation planner. It does not scan the
whole corpus, discover its own authority, call a model, create a pending replacement,
or change knowledge. The application or a separately owned scheduler submits a bounded
`KnowledgeMaintenanceRoutingRequest` containing exact revision signals and an explicit
`KnowledgeAccessScope`, namespace, label subset, and policy identity.

Signals have a closed meaning:

- `exact_reference` nominates one exact revision directly;
- `duplicate_hint` nominates two revisions for comparison while preserving any raw
  similarity score as diagnostics only;
- `expiry` verifies an entry's exact expiration time against the supplied cutoff;
- `low_usage` verifies that creation, latest revision, and last-use time are all older
  than the supplied cutoff; and
- `contradiction` binds and verifies one exact accessible revision-bound `contradicts`
  relation identity between the two exact revisions.

None of these signals proves equivalence, truth, authority, or supersession. They only
justify spending a later bounded evaluation budget on the candidate set.

```python
from datetime import UTC, datetime

from cayu import (
    KnowledgeMaintenanceCandidateSignal,
    KnowledgeMaintenanceRouter,
    KnowledgeMaintenanceRoutingRequest,
    KnowledgeMaintenanceSignalKind,
    KnowledgeRevisionRef,
)

now = datetime.now(UTC)
request = KnowledgeMaintenanceRoutingRequest(
    id="refund-maintenance-routing-2026-08",
    policy_id="knowledge-maintenance-v1",
    namespace="acme-support",
    labels={"project": "billing"},
    access_scope=review_scope,
    signals=(
        KnowledgeMaintenanceCandidateSignal(
            id="refund-policy-duplicate-hint",
            kind=KnowledgeMaintenanceSignalKind.DUPLICATE_HINT,
            references=(
                KnowledgeRevisionRef(entry_id="refund-policy-a", revision=3),
                KnowledgeRevisionRef(entry_id="refund-policy-b", revision=2),
            ),
            producer_id="billing-policy-index",
            producer_version="2026-08",
            reason_code="bounded_lexical_duplicate_hint",
            observed_at=now,
            raw_score=8.25,
            score_kind="bm25-diagnostic-v1",
        ),
    ),
    created_at=now,
)
result = await KnowledgeMaintenanceRouter(store).route(request)
```

Before any store read, the router rejects requests that exceed its signal or unique
candidate-read ceilings. It loads only current revisions through storage-enforced
authorization and passes a per-read byte ceiling to `KnowledgeStore.get_entry(...)`.
Stores persist the canonical payload size with every revision, so they can refuse an
oversized authorized entry before fetching or copying its content. The router also
reserves concurrent reads against `max_candidate_load_bytes`; this bounds aggregate
materialized input independently from the smaller final-payload budget. Entries refused
by either byte boundary receive a content-free `candidate_bytes` omission. Inaccessible
entries remain indistinguishable from missing entries and never disclose their size.

Contradiction verification reserves every relation page from the request-wide
`max_relation_load_bytes` budget before launching the read. Page reservations share the
same deterministic signal order as routing, are capped by `max_concurrency`, and reclaim
only the unused portion of a validated page. This prevents individually valid page and
concurrency settings from multiplying into unbounded simultaneous materialization. When
the remaining budget cannot admit one valid relation page, every pending scan receives a
content-free `relation_coverage_incomplete` omission instead of a guessed decision.

The router rejects stale, cross-scope, and non-active candidates and applies configured
signal priority plus stable timestamp/identity tie-breaking. Pair signals are admitted
or omitted as a unit. Final admission uses incremental exact canonical-byte accounting,
adding only the current signal/reference deltas instead of rebuilding and reserializing
the entire trial payload for every signal. The planner-facing payload is bounded by exact
candidate count and canonical serialized bytes; timeout and caller cancellation cancel
in-flight reads rather than returning a guessed partial result.

`KnowledgeMaintenanceRoutingResult` exhaustively partitions every submitted signal
into routed signals or content-free omission diagnostics. `truncated=True` means a
candidate budget or bounded relation scan prevented complete routing; unavailable,
stale, or condition-not-met signals are fully evaluated rejections instead. The result
binds canonical request and configuration fingerprints and defensively copied current
entries. It also reports exact relation payload bytes consumed against the configured
request-wide ceiling. Routing performs no publication, relation write, lifecycle
transition, or proposal persistence.

Breaking storage revision 65 installs persisted canonical entry sizes and the bounded
entry-read contract. Fresh databases and completely empty earlier knowledge schemas
initialize directly. A populated pre-65 knowledge schema fails before DDL or data change
and must be explicitly replaced. There is no size backfill, legacy read fallback,
compatibility wrapper, or dual-write path.

The hermetic performance gate uses the full supported 50 exact candidates across
in-memory and SQLite stores, plus a zero-candidate control that performs no store reads.
It makes no model or provider calls and verifies that routing leaves every canonical
revision unchanged:

```bash
PYTHONPATH=src python scripts/run_knowledge_maintenance_routing_performance.py --check
```

Results are recorded in
[`benchmarks/memory/knowledge-maintenance-routing-performance-v1.json`](../benchmarks/memory/knowledge-maintenance-routing-performance-v1.json).

## Bounded maintenance planning and independent evaluation

`KnowledgeMaintenancePlanningWorkflow` is the read-only semantic boundary after
candidate routing. An application injects a `KnowledgeMaintenancePlanner` and a
separate `KnowledgeMaintenancePlanEvaluator`; Cayu rejects using the same object or
logical component identity for both roles. The workflow does not choose a provider,
supply a universal prompt, persist a proposal, create a pending replacement, publish a
relation, or change lifecycle state. Even an accepted planning result is an evaluated
draft without write or review authority.

The planner receives a defensively copied `KnowledgeMaintenancePlannerInput` containing
a minimized `KnowledgeMaintenancePlanningSnapshot`, the planning-configuration
fingerprint, allowed replacement kinds, and a stage budget. The snapshot carries only
authorized routed candidates and signals, requested namespace/labels and policy, exact
binding fingerprints, and candidate payload accounting. It deliberately excludes the
broader access-scope configuration, omitted signals, router limits, and unrelated source
identities. A strict `KnowledgeMaintenancePlanDraft` binds:

- every exact routed source revision;
- the routing-request, routing-result, planning-configuration, and policy fingerprints;
- replacement text, title, kind, aspects, and impact targets, but no model-controlled
  namespace, visibility, status, labels, source authority, or write identity;
- one typed relation disposition per source using an unpersisted replacement endpoint;
- bounded replacement claims mapped to exact source revisions and referenced by relation
  dispositions; and
- bounded rationale and evidence-summary text.

Planner output is untrusted even after schema validation. Cayu deterministically rejects
foreign or omitted sources, incomplete or repeated relation coverage, invalid directed
relation orientation, missing relation evidence, evidence outside the routed snapshot,
disallowed replacement kinds, and mismatched policy or configuration bindings. Only a
structurally valid plan reaches the separately injected evaluator. That evaluator owns
the application-specific semantic checks that deterministic code cannot infer from prose:
unsupported synthesis, information loss, contradiction handling, retention constraints,
policy compatibility, and prompt-injected source content. Its result is another strict,
revision-bound structure containing framework-closed, kind-bound codes and exact
references rather than evaluator-selected strings or copied source/replacement prose.
Before disclosing any candidate to either component, the workflow also proves that routed
signals and omissions exactly partition the supplied request and that each routed signal
still equals its original request value. A self-asserted request fingerprint therefore
cannot substitute unrelated authorized knowledge into a planning attempt.

Source entries are storage-reauthorized and compared with the routed immutable snapshot
before planner invocation, after planning, and after evaluation. A revision advance,
disappearance, or scope loss therefore prevents acceptance; a change during planning
also avoids spending the evaluator budget on stale work. Pending-proposal publication
still compares the same exact revisions atomically because a read-only accepted result
cannot reserve the corpus. A source that actually advances produces a deterministic stale
rejection. By contrast, a timeout or storage failure while checking currentness remains a
distinct revalidation-failure outcome, including whether it happened after planning or
evaluation, so retry policy never has to infer infrastructure failure from a semantic
rejection.

```python
from cayu import (
    KnowledgeMaintenancePlanningConfig,
    KnowledgeMaintenancePlanningOutcome,
    KnowledgeMaintenancePlanningWorkflow,
)

workflow = KnowledgeMaintenancePlanningWorkflow(
    store,
    planner=application_planner,
    evaluator=independent_evaluator,
    config=KnowledgeMaintenancePlanningConfig(
        planner_id="billing-consolidator",
        planner_version="2026-08",
        evaluator_id="billing-consolidation-reviewer",
        evaluator_version="2026-08",
        planner_model_ids=("provider/planning-model",),
        evaluator_model_ids=("provider/review-model",),
        allowed_replacement_kinds=("fact", "procedure"),
    ),
)
planning = await workflow.plan(request, result)
if planning.outcome is KnowledgeMaintenancePlanningOutcome.ACCEPTED:
    # The draft is eligible for explicit pending publication, never activation.
    queue_for_pending_review_publication(request, result, planning)
```

Configuration separately bounds planner input, plan output, evaluator input and output,
source revalidation bytes and concurrency, claim count and size, replacement text, stage
timeouts, model calls, and provider cost. The planner receives its claim-count,
claim-size, and replacement-size ceilings in a planner-specific budget before invocation,
so an integration never has to infer hidden structured-output limits from a failed call.
Cost uses integer micro-US dollars to avoid floating-point accounting ambiguity. Each
stage also receives an application-owned allowlist of provider model identities and
reports measured `KnowledgeMaintenanceInferenceUsage`. Unknown component-reported model
identities make the component output invalid and are not reflected into the result;
authorized over-budget usage is preserved while the plan fails closed. Caller cancellation
propagates, even if an injected component catches its own cancellation: planner and
evaluator calls run in separately owned tasks, late output is never accepted, and unsettled
cancelled work remains observed until it finishes. Component exception text is never copied into
diagnostics, truncated routing never calls either component, and zero-candidate routing
performs no store read or component call. Every result records the application-configured
planner and evaluator identities and versions; a completed evaluation is cross-bound to
that evaluator attribution. Final accepted/rejected codes are stamped by the workflow,
not selected by the evaluator.

The planning workflow revalidates the public routing result into its own snapshot contract
and rejects more than the hard 50-source bound before any storage read or component
disclosure. It does not trust caller-supplied router limit fields to establish that bound.

The hermetic performance gate uses the full 50-source bound across in-memory and SQLite
stores. Its deterministic planner and evaluator make no provider or model calls, report
zero cost, perform the required three currentness reads per source, and verify that no
knowledge revision changes. The separate zero-candidate control performs no store read or
component call:

```bash
PYTHONPATH=src python scripts/run_knowledge_maintenance_planning_performance.py --check
```

Results are recorded in
[`benchmarks/memory/knowledge-maintenance-planning-performance-v1.json`](../benchmarks/memory/knowledge-maintenance-planning-performance-v1.json).

## Atomic pending maintenance proposals

`KnowledgeMaintenanceProposalPublisher` is the explicit write boundary after an
accepted planning result. It defensively revalidates the exact routing request, routing
result, accepted plan, source coverage, evidence mappings, relation orientation, policy,
and source security boundary before asking the store to write. Rejected, incomplete, or
stale planning results cannot reach storage through this workflow.

Publication creates one immutable review artifact in a single store transaction:

- revision 1 of a deterministic pending replacement, including its complete chunk set;
- one live `KnowledgeEvidence` pointer and content hash for every exact source revision;
- the exact `KnowledgeMaintenanceProposal` consumed by the reviewed-decision workflow;
- an attempt-independent `KnowledgeMaintenanceAcceptedPlan` containing the evaluated
  plan and its routing, configuration, planner, and evaluator bindings;
- one metadata-only `CREATED` change in the canonical knowledge outbox; and
- an immutable operation receipt that binds every component by SHA-256.

It does not activate the replacement, archive a source, or publish a relation. Those
effects remain exclusively owned by `KnowledgeReviewWorkflow.decide_maintenance(...)`
and the store's atomic reviewed-decision transaction. A decision for a durably published
proposal must equal the stored proposal exactly; changing its rationale, evidence,
relations, sources, or any other fingerprinted field is rejected.

Publication also owns the replacement's lifecycle. While review is pending, no
other entry operation can activate, archive, mutate, delete, or prune it. Rejection
keeps the replacement pending and unlocks only explicit forward retirement:
pending to archived or deleted, and archived to deleted. The rejected replacement
cannot be reactivated or content-mutated, and its exact audit revision cannot be
hard-deleted or pruned. These fences apply to replacements created by this durable
publisher; directly constructed maintenance proposals do not acquire publication
ownership merely by being reviewed.

Approval still requires every source to be the active current revision. Rejection
does not: the store authorizes an exact published rejection from the immutable
publication snapshot, so a reviewer can close and retire a proposal after sources
advance or their expired revisions are pruned. This does not restore or mutate any
source and does not weaken approval's currentness checks.

All sources must have identical namespace, complete label map, and visibility. The
pending replacement inherits that exact boundary, while the application supplies an
explicit review scope that can access current sources and pending knowledge. Publication
rechecks that every source is still the active current revision and verifies evidence
hashes inside the same transaction. An advance after evaluation therefore creates no
replacement, proposal, evidence, chunk, or outbox residue.

```python
from cayu import (
    KnowledgeMaintenanceProposalPublisher,
    KnowledgeMaintenanceProposalPublisherConfig,
)

publisher = KnowledgeMaintenanceProposalPublisher(
    store,
    access_scope=maintenance_review_scope,
    config=KnowledgeMaintenanceProposalPublisherConfig(
        publisher_id="billing-maintenance-publisher",
        publisher_version="2026-08",
    ),
)

publication = await publisher.publish(request, result, planning)
# publication.proposal is pending and must still receive an external review decision.
```

Proposal, replacement, relation, evidence, and operation identities derive from the
accepted semantic plan, review scope, and publisher configuration. Attempt timestamps
and provider usage do not participate, so equivalent accepted attempts converge on the
same bytes and identity. A retry after a lost response or caller cancellation loads the
existing artifact and marks the returned receipt as replayed; different material reusing
an operation or proposal identity fails closed. Loading the artifact after a decision
retains the exact pending revision for audit while reporting that the proposal is already
decided.

In-memory, SQLite, and PostgreSQL stores implement the same publication and review-handoff
contract. Breaking storage revision 67 adds only the new proposal-publication record.
Existing knowledge revisions are retained, but no proposal, accepted plan, or receipt is
inferred or backfilled from them. Pre-67 workers cannot share the migrated store; there is
no dual-write path, legacy proposal interpretation, or compatibility wrapper.

## Explicit reviewed knowledge curation

`KnowledgeCurator` is the provider-neutral, explicitly invoked path from application
evidence to human-reviewable knowledge. It does not inspect sessions by itself, start a
worker, choose a model, or write active knowledge. The application decides when a run or
domain operation is complete, extracts bounded `LearningSignal` values, and calls
`curate(...)` with a `LearningBatch`.

The lifecycle keeps five concepts separate:

- raw evidence is the application-owned transcript, artifact, repository revision, or
  domain record;
- a learning signal is a bounded observation that points to exact evidence but is not
  itself knowledge;
- a candidate is a proposed reusable fact, procedure, or pattern from a caller-supplied
  `KnowledgeCandidateGenerator`;
- pending knowledge is an accepted candidate committed for review with immutable,
  revision-bound `KnowledgeEvidence`; and
- active knowledge is a reviewed revision that normal list, search, recall, and context
  injection may use.

A separately supplied `LearningEvaluator` must explicitly accept each candidate. An
optional application policy can reject or transform content before evaluation; when it
is present, the configuration requires its stable identity. Generator and evaluator
identities are always explicit. Core supplies no prompts, model IDs, taxonomy, ambient
scheduler, semantic-merging heuristic, or universal secret detector. Applications must
validate source authorization and enforce their own domain-specific rejection or
redaction rules.

Accepted candidates always enter `KnowledgeStatus.PENDING`. The curator uses the same
atomic revision, chunks, evidence, publication-receipt, change-outbox, and derived-index
readiness contracts as other knowledge writes. Deterministic scoped proposal identities
make exact retries and concurrent calls converge across processes. A retry of an already
pending, active, archived, or deleted proposal reports that durable state; it does not
create a replacement revision or rerun an evaluator after the exact proposal is known to
be durable.

Activation receipts retain the full bounded request for exact replay and audit, so access
requires both the immutable publication-time scope and the logical entry's current scope.
Expiration pruning atomically replaces the current boundary with a content-free final-revision
retirement authority; it never infers pruning from a receipt's own expiry. A retained receipt
requires its publication scope, that final retirement scope, and explicit expired-material
authority. The marker prevents reuse of the logical entry ID until hard deletion erases it and
all activation receipts. Hard deletion can perform that erasure after pruning and returns
`None` because the canonical entry was already removed; the content-free publication receipt
remains solely to preserve operation-id occupancy. Database reads authorize the receipt against
current-entry or retirement authority in one stable snapshot. The final authority has a
1,048,576-byte canonical limit, so every governed publication and later successor is rejected
before mutation if its final access scope could not be preserved during expiration pruning.

`LearningBatchResult` reports batch, signal, and candidate outcomes with stable codes.
Generator failure invalidates the whole batch before any write. Evaluator, policy, and
store failures are isolated per candidate where the durable outcome can be established
safely, and raw component exception text is not copied into the public result.

The default configuration accepts at most 50 signals (16 KiB each and 256 KiB for the
batch), 20 candidates (64 KiB each and 512 KiB together), 32 KiB of candidate text, and
4 KiB of title text. A signal may carry 20 source references; one candidate's complete
provenance, including all referenced signals, may carry 100. References, JSON metadata,
and evaluator notes are each capped at 16 KiB by default. Configuration can lower these
bounds or raise them only to the contract ceilings. Candidate evaluation is limited to
four concurrent calls by default, and unfinished publication tasks retained across caller
cancellation are bounded separately. Timeouts, chunk sizes, and chunk counts are also
explicit configuration rather than unbounded component behavior.

```python
async with KnowledgeCurator(
    knowledge_store,
    candidate_generator=my_generator,
    evaluator=my_evaluator,
    config=KnowledgeCuratorConfig(
        candidate_generator_identity="acme.generator.v1",
        evaluator_identity="acme.evaluator.v1",
        namespace="acme",
        labels={"tenant": "acme"},
    ),
) as curator:
    result = await curator.curate(LearningBatch(id="build-42", signals=(signal,)))

reviewer = KnowledgeReviewWorkflow(
    knowledge_store,
    namespace="acme",
    labels={"tenant": "acme"},
)
pending = await reviewer.list_pending()
approved = await reviewer.approve(pending.entries[0].entry.id)
```

`KnowledgeCurator` and `RememberKnowledgeTool` share the same bounded retained-publication
lifecycle. Caller timeout or cancellation leaves the exact dispatched publication owned so an
in-process retry joins it rather than racing a second write. Directly constructed components
should be used as async context managers or closed with `await component.aclose(timeout_s=...)`.
A `CayuApp` owns this lifecycle for registered tools; server shutdown seals publication before
draining it for `knowledge_publication_shutdown_grace_seconds`. The same deadline covers
receipt-reconciliation reads already retained by a publication, so a mounted Cayu application
does not leave cooperative store tasks behind in a host event loop that remains alive.

Grace expiry requests cancellation from the local store awaiter but does not claim that a remote
transaction failed or was rolled back. A later process uses the same operation ID and durable
publication receipt to reconcile a commit whose acknowledgement was lost. Custom in-process store
adapters must let lifecycle cancellation unwind their local coroutine and must not perform
unbounded blocking work on the event-loop/default-executor shutdown path. Python cannot forcibly
stop arbitrary extension code that suppresses every cancellation; deployments still need their
normal process-supervisor hard shutdown deadline for a broken adapter.

The existing `remember_knowledge` tool remains the explicit foreground write primitive
for an agent or application. The curator is a higher-level evidence-to-proposal workflow;
it does not replace that tool, retrieval fusion, `ContextExposure`, or the runtime's
separate decision about what active memory enters model context. See the credential-free
[`reviewed_knowledge_curator.py`](../examples/reviewed_knowledge_curator.py) example for a
completed-session-to-pending-to-approved-to-later-recall path.

## End-to-end reviewed-maintenance evaluation

The provider-free `run_knowledge_maintenance_evaluation(...)` contract connects the
maintenance router, planner/evaluator boundary, pending-proposal publisher, explicit
review transaction, active recall, and exact historical lineage in one reusable corpus
runner. Its public corpus contains six reference outcomes: duplicate merge,
authoritative supersession, unresolved contradiction, stale proposal, reviewer
rejection, and historical lineage after approval. In-memory and SQLite results are
checked into the repository; PostgreSQL executes the same correctness corpus in its
store parity suite.

The runner measures exact routed-set precision and recall, replacement claim retention
with its exact source mappings, exact source-revision evidence retained after review,
unsafe acceptance, exact lifecycle preservation, lineage correctness, and end-to-end
latency. Historical correctness requires both the archived exact-revision read and its
typed replacement lineage. It requires an empty store so pre-existing records cannot
hide leakage or change the expected lifecycle. Every case has a separate namespace and
application-owned access scope. Corpus validation applies the executable 50-source
maintenance bound and the recall primitive's 8,192-byte query bound, and recall inspects
that complete permitted lineage set. Receipt, evidence, lifecycle, and lineage checks bind
the full returned contracts to the proposal-derived entry revisions, relation identities,
and requested result queries; a matching logical entry ID is not sufficient.

The fixture planner and evaluator are separate components with explicit identities, but
both have a hard zero-model-call and zero-cost budget. This intentionally separates the
framework question—whether Cayu safely carries a known semantic decision through all
boundaries—from the provider-dependent question of whether a particular model produces
good semantic decisions. The evaluation does not discover signals, judge real-world
truth, change `ContextExposure`, or decide what memory enters agent context. Private
production-shaped corpora use the same bounded schema with
`origin="external_private"` and remain outside the public repository.

Run the public backend matrix from the repository root:

```bash
PYTHONPATH=src python scripts/run_knowledge_maintenance_evaluation.py
```

## Derived-index identity and readiness

Every comparable durable embedding uses one immutable
`KnowledgeEmbeddingIdentity`. The identity binds the logical entry and exact
revision, optional exact chunk, projection type and the SHA-256 of the exact
projected UTF-8 text, embedding model and dimensions, preprocessing version,
generator and generator version, and index-representation version. The
projection hash is computed by Cayu rather than trusted from optional chunk
metadata. A content hash by itself is never a safe reuse key. Changing any
component creates a different projection space; an old row may remain auditable
but cannot be returned as a hit for the new identity.

`KnowledgeIndexReadiness` is a separate append-only publication sequence. A
new identity starts `pending`; the same attempt may then become `ready` or
`failed`. A retry publishes a new `pending` attempt through compare-and-swap
against the latest readiness sequence. Exact operation replay is idempotent,
while stale sequences, old attempts, and operation-ID reuse fail with
`KnowledgeIndexReadinessConflict`. Readiness publication is allowed only while
the named entry revision and optional chunk still match current canonical
state, so a slow worker cannot make an obsolete projection ready.

`read_index_readiness(...)` returns bounded, authorized event pages through a
store-owned high-water mark. `load_index_readiness(...)` resolves the latest
state for one exact identity. These are optional, non-abstract extension hooks:
lexical-only custom stores remain valid, while stores advertising semantic
search must implement equivalent identity, fencing, and readiness semantics.
Canonical knowledge commits do not claim readiness, and readiness publication
never creates or changes a canonical revision.

The built-in embedding stores consume that boundary explicitly with bounded
`process_embedding_changes(consumer_id, worker_id, limit=..., record_limit=...)`
calls. `limit` bounds canonical changes and `record_limit` independently bounds
the total chunk projections written, failed, or removed by one call. When one
change exceeds that budget, its claim is released without advancing the cursor;
the next call deterministically skips exact ready identities, retries failed
identities, and continues with the next chunk or stale-vector cleanup page.
Canonical writes first commit an outbox change
without calling an embedding provider. A worker claims the change, publishes
`pending`, commits the exact-identity vector, publishes `ready`, and only then
acknowledges the change. A crash after the vector commit leaves it invisible
behind `pending`; replay repairs `ready` without paying for the vector again. A
crash after `ready` replays without a duplicate visible projection. Provider
failures publish a non-sensitive `failed` state and can be retried through an
explicit bounded backfill.

`store_embedding_projections(...)` is the public persistence boundary for
already-computed vectors. Each submitted `KnowledgeEmbeddingProjection` binds
the vector to its complete identity plus the exact pending readiness sequence
and attempt. The store accepts it only if that identity is still current and
authorized and that pending attempt has not been superseded; stale records are
omitted from the typed write result. The caller then publishes `ready`
separately. Replaying the same attempt and vector is idempotent. Reusing that
attempt marker with a different vector raises
`KnowledgeEmbeddingProjectionConflict` and leaves the whole request unchanged;
the vector becomes replaceable only after a newer pending attempt wins the
readiness compare-and-swap. This allows an external projection service to
compute vectors without duplicating private store logic, while the built-in
workers use the same fenced path. `backfill_embeddings(...)` returns the same
portable bounded result shape for the in-memory and PostgreSQL embedding stores.
When more eligible records remain, `next_cursor` is an opaque keyset
continuation bound to the exact query, access scope, projection configuration,
and refresh mode. Pass it back as `cursor=...` to advance a large same-identity
refresh without revisiting the first page.

Semantic and hybrid reads never mutate the index. Each result carries
`KnowledgeIndexCoverage` for the exact eligible chunk set and complete
comparable projection space: model, dimensions, preprocessing, generator, and
index representation; ready vectors, pending or missing projections, failed
attempts, and the greatest matching readiness sequence. `complete` is true only
when every eligible record is ready.
Keyword lanes remain available during partial semantic coverage, but callers
can distinguish that result from a complete semantic search.

PostgreSQL materializes the complete readiness identity as typed columns for
bounded coverage and rebuild scans. Each vector row also records the exact
pending readiness sequence and attempt that accepted it, so a stale batch
cannot masquerade as the current write. Projection attempts are append-only
within an identity. Current search resolves the newest accepted vector no later
than the current READY event, while frontier search resolves the newest accepted
vector no later than the READY event at its captured sequence. A newer refresh
therefore cannot replace or hide the projection selected by an older frontier,
including across a crash between vector commit and READY publication. HNSW
indexes are partial to one complete compatible projection space, with separate
current-projection and historical-frontier graphs so retained attempts cannot
degrade ordinary current-search candidate budgets. Vectors from different
models, generators, preprocessing versions, dimensions, or index representations
never share an ANN graph. Schema validation rejects a cross-space HNSW index.
Projection publication locks the corresponding readiness-current rows before
acceptance and activation. New attempts enter as historical rows, and a partial
unique index enforces at most one current projection per complete identity even
under concurrent refresh and readiness publication.

Breaking schema revision 44 historically added the readiness event log and
current pointer while preserving revision-43 canonical state. Pre-identity
vector rows were deliberately discarded as rebuildable derived data. Current
revision-60 startup does not use that preservation path for a populated
pre-60 knowledge store.

Evidence prefixes and multi-entry expiration changes use the same scalar
identity ordering in every built-in backend. SQLite makes its binary collation
explicit and PostgreSQL uses the portable `C` collation, so bounded reads and
bulk cleanup do not change with the database locale.

Canonical entry IDs are limited to `MAX_KNOWLEDGE_ENTRY_ID_BYTES` UTF-8 bytes
and canonical chunk IDs to `MAX_KNOWLEDGE_CHUNK_ID_BYTES`. The same limits apply
to every receipt, evidence, and change reference, keeping identity behavior and
indexed storage portable across the in-memory, SQLite, and PostgreSQL backends.

Breaking schema revision 43 historically installed the evidence, change, and
consumer tables and the exact chunk-owner key used by evidence foreign keys.
That transition did not fabricate evidence or historical change events. It is
not a compatibility promise across the later revision-60 clean break.

## Revision-schema reset policy

The revision-first knowledge schema at database revision 42 is a deliberate
breaking contract:

- fresh databases and completely empty legacy knowledge tables may initialize;
- populated mutable-knowledge tables require an operator-approved database
  replacement/reset;
- startup and migration must not infer revision 1, silently delete knowledge,
  shadow-write, fall back to mutable reads, or expose a revision-optional API.

`require_empty_knowledge_revision_transition(...)` is the content-free decision
primitive used by each backend migration. The backend supplies the complete
required table set and population evidence. An incomplete inspection fails; any
populated table raises `KnowledgeRevisionResetRequired` before schema, data, or
migration-ledger mutation. The empty transition installs immutable entry
revisions, revision-bound chunks, and a current-revision pointer atomically.
Normal startup validates the complete authoritative revision layout, including
table columns and types, keys and revision bounds, foreign keys, the current
revision view, FTS structures, and required indexes. A recorded revision-42
database with a missing or conflicting object fails before serving traffic.
Revision 60 applies the same fail-before-DDL rule to every populated pre-relation
knowledge schema. Empty old schemas are replaced atomically with the relation-aware
outbox; populated schemas require an explicit application-owned replacement.

## Deterministic weighted rank fusion

`WeightedReciprocalRankFusion` is the versioned reference strategy. Each enabled
channel supplies a bounded ranked list with exact typed canonical identity,
representation, content hash, index version, truncation/continuation evidence,
raw diagnostic score, explanations, and normalized deterministic features.

For candidate `c`:

```text
score(c) = Σ channel_weight / (rrf_k + best_rank_in_channel)
           + Σ feature_weight × normalized_feature_value
```

Raw BM25, cosine, and provider scores remain diagnostics and never enter this
formula. Duplicate representations collapse to one canonical revision and one
contribution per channel while retaining every match. Channels and arithmetic
are evaluated in sorted order. Ties resolve by score, best rank, channel count,
then `(record_type, record_id, revision)`.

The configuration records a caller-owned version, strategy version, weights,
`rrf_k`, feature adjustments, per-channel candidate ceiling, fused-head limit,
and a canonical SHA-256 fingerprint. Construction rejects numeric values that
cannot enter that canonical durable representation, so every accepted
configuration is immediately fingerprintable. Weight maps remain deeply
immutable after construction, and updated model copies pass through the same
validation before receiving a new fingerprint. The result records channel/index
coverage, candidate/omission counts, truncation reasons, and continuation
availability. Inputs that omit a configured channel, add an unconfigured channel,
exceed a budget, duplicate a rank, or disagree on canonical feature values fail
closed.

A replacement `RetrievalFusionStrategy` must preserve these invariants even if
its ranking method differs: hard filtering occurs before its inputs, all work is
bounded, canonical revision identity is retained, output is deterministic for
recorded inputs/configuration, diagnostics reproduce the decision, and raw
payload text from rejected candidates is not required in model-facing output.

## Reproducible baselines

[`benchmarks/memory`](../benchmarks/memory/README.md) contains public hermetic
corpora and checked in-memory/SQLite results for both knowledge-only retrieval
and cross-source recall. The runners require no model or network calls. They
measure retrieval quality, false injection/results, stale results,
authorization leaks, source/locator correctness, honest partial coverage,
candidate/truncation counts, byte/token overhead, latency, multilingual
queries, duplicate provenance, and short follow-ups.

The same bounded corpus schema accepts `origin="external_private"`, trajectory
identity, and turn index. Production-shaped long trajectories stay outside the
public repository and use the same runner locally; reports identify the corpus
revision, backend, search mode, embedding/reranker identity, and configuration.
