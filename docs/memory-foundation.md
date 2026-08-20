# Memory foundation contracts

This document records the Phase 0 contracts for Cayu's v5.1 long-term-memory
work and the immutable knowledge-revision core now built on them. Cross-source
recall, context composition, curation, and automatic governance remain separate
layers.

## Knowledge and memory are different layers

**Knowledge** is durable canonical semantic material: entries, chunks, source
identity, lifecycle, immutable revisions, and revision-bound source evidence.
**Memory** is the larger recall system that can retrieve permitted
knowledge, transcript episodes, artifact-derived documents, and other typed
sources, fuse them, select a bounded context contribution, and record exposure.

The current `KnowledgeStore` is therefore one memory source. The WRRF
types in `cayu.retrieval` are source-neutral and do not turn transcripts or
artifacts into knowledge.

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
Receipts preserved from revision 42 remain exactly replayable after migration;
their entry-and-chunks-only digest is accepted only for an empty-evidence retry.

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
readiness; index workers add their readiness protocol in the following slice.

Evidence prefixes and multi-entry expiration changes use the same scalar
identity ordering in every built-in backend. SQLite makes its binary collation
explicit and PostgreSQL uses the portable `C` collation, so bounded reads and
bulk cleanup do not change with the database locale.

Canonical entry IDs are limited to `MAX_KNOWLEDGE_ENTRY_ID_BYTES` UTF-8 bytes
and canonical chunk IDs to `MAX_KNOWLEDGE_CHUNK_ID_BYTES`. The same limits apply
to every receipt, evidence, and change reference, keeping identity behavior and
indexed storage portable across the in-memory, SQLite, and PostgreSQL backends.

Breaking schema revision 43 installs the evidence, change, and consumer tables
and the exact chunk-owner key used by evidence foreign keys. The DDL preserves
revision-42 entries, IDs, revisions, chunks, and receipts that satisfy the
portable identity bounds; out-of-contract identities are rejected before any
revision-43 DDL is applied. The migration deliberately does not fabricate
evidence or historical change events for mutations that happened before
revision 43.

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

## Reproducible baseline

[`benchmarks/memory`](../benchmarks/memory/README.md) contains the public
hermetic corpus, checked in-memory/SQLite keyword results, and the command that
reproduces them without model or network calls. It measures retrieval quality,
false injection, stale results, authorization leaks (including ID/chunk reads),
source/citation correctness, candidate/truncation counts, byte/token overhead,
latency, and English/Spanish/French slices.

The same bounded corpus schema accepts `origin="external_private"`, trajectory
identity, and turn index. Production-shaped long trajectories stay outside the
public repository and use the same runner locally; reports identify the corpus
revision, backend, search mode, embedding/reranker identity, and configuration.
