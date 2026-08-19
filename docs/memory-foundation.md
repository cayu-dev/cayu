# Memory foundation contracts

This document records the Phase 0 contracts for Cayu's v5.1 long-term-memory
work and the immutable knowledge-revision core now built on them. Cross-source
recall, context composition, curation, evidence, and automatic governance remain
separate layers.

## Knowledge and memory are different layers

**Knowledge** is durable canonical semantic material: entries, chunks, source
identity, lifecycle, and immutable revisions, with evidence added by a later
slice. **Memory** is the larger recall system that can retrieve permitted
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
