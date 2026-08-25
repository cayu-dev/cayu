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
below 32 KiB per pair. A dedicated no-coverage CI lane executes the current workload,
while the normal test suite validates the checked artifact and every regression lane;
later code therefore cannot retain stale green numbers or distort timing with coverage.
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

A relation is not truth, similarity, approval, current-state activation,
ranking weight, or context-placement authority. This slice performs no graph
traversal and does not automatically archive a predecessor. Reviewed atomic
supersession and relation-aware recall are separate later policies.

Breaking storage revision 60 installs this final prerelease relation contract.
Fresh databases and completely empty earlier knowledge schemas initialize
directly. Any populated pre-60 canonical, evidence, receipt, outbox, readiness,
embedding, or relation table makes migration fail before DDL. There is no
backfill, metadata fallback, dual write, or legacy relation interpretation.

The hermetic relation performance gate runs in CI with no provider calls:

```bash
PYTHONPATH=src python scripts/run_knowledge_relation_performance.py --check
```

Its checked report returns 50 matching relations from a store containing 5,000
additional unrelated relations, so endpoint lookup must remain index-bound. The
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
archive, relate, or delete any entry. Rejected pending content therefore remains
pending unless the caller performs a separate lifecycle decision.

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
cannot masquerade as the current write. HNSW indexes are partial to one complete
compatible projection space; vectors from different models, generators,
preprocessing versions, dimensions, or index representations never share an
ANN graph. Schema validation rejects a cross-space HNSW index.

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
