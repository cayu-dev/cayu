# Release notes

## v0.1.0rc5

This is the final code candidate for `v0.1.0`. If its release evidence passes,
the final release should differ only in version and release metadata.

### Upgrade from v0.1.0rc4

Pin the complete application to `cayu==0.1.0rc5`, refresh its lockfile, and run
its own tests. Upgrade independently deployed Cayu servers, dashboards, and
generated clients together: the server contract advances from version 4 to
version 6, and the public application manifest and generator plan advance to
schema version 7.

The storage schema advances from revision 23 to revision 29. Revision 26 is a
deliberate prerelease boundary: migration rejects a populated pre-26 Cayu
session database instead of attempting to rewrite its durable interaction
history. Stop all older workers, take an application-consistent backup, and
recreate populated prerelease Cayu session databases before starting rc5.
Empty stores migrate normally. Run `cayu storage status` and
`cayu storage migrate` against every explicitly configured SQLite or PostgreSQL
session store and budget ledger, then confirm revision 29 with no pending
migrations. Do not run mixed rc4/rc5 workers.

After deployment, verify `cayu version`, run `cayu check --json`, execute the
application's test suite, and exercise a durable session through process
restart, approval or deferred input, and recovery. Persistent multi-worker
deployments must use one consistent public-authority alias keyring.

### What this candidate validates

- Durable interaction admission, replay, approval, model-budget settlement,
  and terminal recovery now retain bounded, attributable evidence across
  crashes and retries.
- `cayu serve` and `cayu worker` provide stable entrypoints for configured
  projects, while bounded topology, workflow, usage, and event projections
  support control-plane inspection without exposing private authority.
- Oversized tool results can be externalized after redaction, workspace
  mutations support revision-checked reads and writes, and remote allocation
  fails closed where exact recovery is unavailable.
- The bundled model catalog and pricing snapshot have been refreshed against
  current provider information.

The complete intended `v0.1.0` contract remains documented in the unreleased
section below. This rc5 section is the curated public GitHub release note and
the focused upgrade guide for the candidate.

## v0.1.0rc4

This release candidate is an upgrade-testing checkpoint for applications and
coding agents consuming Cayu as a normal Python package. It is not the final
`v0.1.0` support declaration.

### Upgrade from v0.1.0rc3

Pin the complete application to `cayu==0.1.0rc4` (including whichever extras it
uses), refresh its lockfile, and run its own tests. Do not upgrade only a
globally installed `cayu` executable while leaving the application environment
on rc3. Run every installation and migration command with Python 3.11 or newer;
an unqualified system `python3` may be older than Cayu supports.

The storage schema advances from revision 21 to revision 23, with breaking
boundaries at revisions 22 and 23. For every SQLite or PostgreSQL database used
by a session store or separately configured budget ledger:

1. stop all rc3 and older Cayu workers;
2. take an application-consistent backup;
3. use the rc4 executable to run `cayu storage status`, followed by
   `cayu storage migrate` against the explicit `--sqlite PATH` or
   `--postgres DSN` target;
4. confirm `cayu storage status` reports revision 23 with no pending
   migrations; and
5. deploy only rc4 workers.

Do not use a mixed rc3/rc4 rolling deployment. After revision 23 is committed,
an application-only rollback to rc3 is unsupported; restore the application
and its pre-upgrade database backup together.

The server contract advances from version 2 to version 4. Upgrade independently
deployed Cayu servers, generated clients, and dashboards together. The rc4
dashboard intentionally rejects an older server contract instead of guessing.

After deployment, verify `cayu version`, run `cayu check --json`, execute the
application's test suite, and exercise at least one durable session through
process restart. Coding agents should treat failed storage status, migration,
contract, or project tests as a blocked upgrade rather than editing around the
guard.

## Unreleased

### Captured sessions can launch bounded fresh trials from the Control Plane

Simple captured sessions with safely reconstructable input can now move directly
from evidence review to a fresh one-trial run. The review sheet keeps preview
side-effect free, exposes the published target ceilings and optional runtime and
cost contractions, saves the reviewed captured result, and queues fresh work
through the existing durable eval worker. Generated project targets default to
one trial and concurrency one and reuse normal provider, tool, environment,
approval, and operator policy; broader scale or changed authority still requires
an explicit application-owned target. Runtime and estimated-cost ceilings apply
to each trial independently, while trial count and concurrency bound aggregate
run scale.

Authenticated HTTP provenance and all execution contractions are now part of the
immutable run admission record. Restarted workers reconstruct the same bounded
runtime invocation instead of falling back to unattributed SDK authority. Cost
ceilings use only a server-owned `PriceBook` that currently prices the resolved
target model; the catalog publishes its compatible currencies and admission
rejects missing model pricing or a currency mismatch before writing. The browser
cannot supply pricing or executable configuration. Ambiguous dashboard retries
retain the original idempotency key, and the fresh launch uses the same
run/result and reporting contracts as every other eval.

Storage revision 50 is a breaking boundary that adds the durable eval-run
invocation projection. Stop revision-49 workers, back up each SQLite or
PostgreSQL store, run `cayu storage migrate`, and confirm revision 50 before
starting current workers. Mixed revision-49 and revision-50 eval workers are
unsupported.

### OpenAI subscription retries are bounded with migration-explicit authority

OpenAI subscription HTTP and SSE errors now preserve bounded typed retry
classification. Known terminal failures remain terminal, known transient
failures follow the caller's ordinary retry policy, and genuinely unclassified
provider failures use the stricter `RetryPolicy.max_unknown_attempts` ceiling,
which defaults to at most two total attempts. Durable `model.error`,
`model.retry`, and `model.attempt_discarded` evidence keeps the caller's general
`max_attempts` separate from the classification-specific
`effective_max_attempts` and records `reason="unknown_provider"` for unknown
failures through terminal exhaustion.

This semantic addition deliberately advances model-finalization profile
material from v1 to v2 and the built-in `ModelCompactor` and
`PromptCacheCompactor` materials from version 1 to version 2. Existing
schema-v3 sessions can therefore report `finalization` and, where applicable,
`context_compaction` drift. Start a new session or explicitly adopt the new
profile through an application policy and an authority-authorized adoption
intent; the default policy rejects the drift before work. Pre-change serialized
adoption requests have a different fingerprint and must be resubmitted under
the new contract. No storage migration is required.

### Captured sessions become durable evaluations from the Control Plane

Completed and failed retained sessions now expose an **Evaluate** workflow that
previews public-safe evidence, creates assertions from observed facts, scores
without rerunning the application, and atomically saves an immutable captured
result with its expectation corpus. The Evals Results catalog exposes captured
and fresh results together, with actor-attributed compare-and-swap baseline
approval. Captured-only corpora contain no invented replay input and fresh-run
admission explains that runnable input or a scenario must be authored first.

Storage revision 48 is a breaking boundary that permits the case catalog to
represent captured-only cases with zero runnable messages. Stop revision-47
workers, back up each SQLite or PostgreSQL store, run `cayu storage migrate`,
and confirm revision 48 before starting current workers. Mixed revision-47 and
revision-48 workers are unsupported.

### Canonical tool catalogues provide shared dynamic-tool identity

Every admitted agent now owns one immutable `ToolCatalogSnapshot` derived from
its registered tools. Frozen `ToolDescriptor` records bind canonical native or
MCP identity, the exact callable schema and behavior declarations, and
sanitized provenance into deterministic descriptor versions. Snapshot
revisions are independent of registration order and JSON object-key order,
while provider requests and exposure policies keep their existing registration
order. MCP descriptors reuse authoritative manifest/contract evidence and
retain only a fixed-size fingerprint of the original source tool name.

The catalogue revision is bound into direct-tool execution profiles,
schema-v2 exposure evidence, and compact model-completion recovery authority.
Reconstruction accepts an equivalent catalogue and rejects changed or corrupt
catalogue authority before recovered tool execution. Tool implementation
identity remains independently governed by the existing
`tool_implementations` execution-profile component. Memory, SQLite,
PostgreSQL, and session export/import share the same durable evidence shape.
The outer execution-profile schema remains v3 because its record and component
set are unchanged; pre-catalogue v3 profiles conflict on the versioned
`direct_tools` component and are not translated.
Public builders require explicit grant material for nonempty compact catalogues
because canonical MCP ids cannot recover model-visible registered names.

`call_tool`, `search_tools`, and the structured-output submission name are now
one reserved framework namespace and collide at registration instead of being
shadowed. This slice adds no gateway, discovery, grant, routing, or
provider-native dynamic-tool behavior, and it does not change default provider
payloads. Persisted prerelease exposure authority without a catalogue revision
is rejected rather than supported through a compatibility path.

### OpenAI Responses background work survives worker loss

`OpenAIProvider(background=True)` now starts stored, streamed Responses API
background operations and durably binds their response ID and accepted sequence
cursor to Cayu's provider-operation recovery contract. Recovery retrieves
offline completions or resumes after the last accepted OpenAI event without
duplicating transcript output, terminal events, usage, or cost settlement.
Cancellation targets the same response and preserves completion when it wins a
race. The default OpenAI path and every unsupported provider remain synchronous.

OpenAI does not document exact key-only recovery for a lost create
acknowledgement, so Cayu reports generic `ambiguous_submission` evidence instead
of retrying or heuristically searching for a response. Background storage,
retention, ZDR, latency, account, model, region, and project-policy tradeoffs are
documented in `cayu guide providers` and must be accepted explicitly by enabling
the provider option.

### Bounded cross-source recall preserves exact evidence and authority

`RecallEngine` now coordinates independently bounded knowledge and transcript
sources, validates exact canonical records, and applies deterministic,
caller-versioned weighted reciprocal-rank fusion. `RecallSituation` requires an
explicit knowledge access scope and explicit transcript session IDs;
`RecallResult` reports exact revision/locator evidence, source coverage,
truncation, and fusion diagnostics. Recall is intentionally retrieval-only and
does not automatically expose candidates to a model context.

Session stores add bounded narrative transcript search with identical
in-memory, SQLite, and PostgreSQL behavior. Only user/assistant `TextPart`
content is indexed; thinking, tools, provider state, system messages, and
non-text parts are excluded. A portable case-folded document uses collision-free
hex identities for ordinary terms and fixed-size SHA-256 identities for long
terms, giving Python, SQLite FTS5, and PostgreSQL GIN identical long-token,
Unicode, and token-boundary semantics. Phrase and distinct-term coverage
outrank bounded repetition. Results are relevance-ranked
before page truncation, and an exceeded scan ceiling fails closed without
exposing an arbitrary partial ranking. Opaque session terms constrain candidate
work inside each backend index.
Explicit session scoping, score-bound keyset cursors, scan ceilings, and byte
ceilings prevent global or unbounded transcript search.

Recall continuations now advance only through the contiguous per-channel prefix
present in the returned fused result, so fusion-head and result-byte clipping do
not skip omitted transcript hits. Optional semantic knowledge lookup has its own
deadline and no longer discards successful lexical evidence when it fails.
Custom fusion implementations must publish their own configuration-matched
strategy identity instead of claiming WRRF provenance.

Storage revision 46 is a deliberate breaking boundary because session-store
binaries now promise the indexed search capability. It does not backfill or
reinterpret any pre-revision-46 transcript row. Migration fails before changing
the schema when the transcript table is populated; recreate that Cayu database
before starting this build. Empty stores may advance to revision 46 and fresh
stores are created directly with the final projection and indexes. Cayu carries
no transcript compatibility or projection-repair path.

The transcript index version records Python's Unicode tokenizer database, and
durable stores persist the same identity. A mismatch fails startup and requires
a clean database; it is never repaired or migrated in place. Cancelled SQLite
transcript reads return to the caller promptly while the physical worker
remains connection-fenced until it settles.

The public hermetic cross-source corpus freezes backend-parity identities and
measures recall, false results, stale revisions, authorization leaks, locator
correctness, honest partial coverage, candidate/byte overhead, multilingual
queries, duplicate provenance, and short follow-ups without provider calls.

## v0.3.0

`v0.3.0` hardens Cayu's durable runtime contracts while extending the
reproducible evidence, knowledge, workspace, task, browser, and evaluation
foundations introduced since `v0.2.1`.

### Upgrade from v0.2.1

Stop all `v0.2.1` and older workers, take application-consistent backups, and
upgrade independently deployed servers, dashboards, generated clients, and
workers together. The server contract advances from version 10 to version 16,
and the public application manifest and generator plan advance from schema 7
to schema 9.

The storage schema advances from revision 36 to revision 50. Follow the
revision-specific migration boundaries below: revisions 39 through 50 contain
breaking durable contracts, and populated legacy knowledge or task stores may
require the explicitly documented rebuild or drain procedure. Run `cayu storage
status` followed by `cayu storage migrate` against every configured SQLite or
PostgreSQL store, and confirm revision 50 with no pending migrations before
starting `v0.3.0` workers. Mixed-version deployment and application-only
rollback across these boundaries are unsupported.

Revision 47 adds the origin-aware immutable Evals result index,
actor-attributed baseline pointers, and idempotent baseline mutation audit. It
indexes existing fresh results without copying or fabricating execution
documents. Every fresh-result writer must maintain that index atomically, so
revision-46 workers must be stopped before migration and cannot share the
migrated database.

### Project serving assembles the durable Evals foundation

`cayu serve` now derives normalized project identity from `[project].name`,
release identity from `CAYU_RELEASE_ID` or the public application-manifest
fingerprint, and a durable Evals store from the project's existing SQLite or
PostgreSQL session-store declaration. Explicit loopback `--dev` may create the
project-local `data/cayu.db` default; production never invents storage. All
public identity crosses the application's workload-secret redaction boundary,
and Cayu owns and closes the assembled store.

Generated maintained-service factories carry an opaque
`ProjectControlPlaneContext` into the server assembler. Existing factories
remain source-compatible and receive an actionable `cayu check` warning plus a
conservative, idempotent `cayu generate service-context` migration. Explicit
`EvalsConfig` remains authoritative and is never field-merged with automatic
state. Generated project serving now publishes a bounded, stable target for
every registered agent while retaining executable authority only in process.

### Control Plane Evals now publishes operation-level readiness

The Evals navigation and direct route now remain discoverable even when a
deployment has not assembled the Evals catalog. The page renders the server's
independent readiness for captured evaluation, catalog reads and writes,
captured-result persistence, scenario conversion, fresh launches,
cancellation, comparison, and reports. Unready pages do not probe absent Evals
endpoints, and planned framework work is distinguished from a genuine
deployment or runtime limitation.

The control-plane contract advances from version 13 to version 14 with the new
required `capabilities.evals_readiness` projection. Its closed state and reason
codes are discovery metadata rather than authorization: authentication,
mutation policy, and operation preconditions remain authoritative at the
underlying routes. Independently deployed servers, generated clients, and
dashboards must be upgraded together. This first delivery slice does not yet
assemble Evals storage or execution targets automatically and adds no durable
writes or workers.

### Tool exposure now governs frozen model-step request profiles

Cayu now provides immutable `RegisteredToolCapability` summaries, bounded
`ToolExposurePolicyRequest` and `ToolExposureDecision` records, deterministic
expose-all and static named-profile policies, and `resolve_tool_exposure(...)`.
Resolution uses detached policy input, accepts names only, rejects policy
mutation and unknown or out-of-ceiling selections, restores canonical
registration order, and binds the resolved profile to exact schema and
definition fingerprints without exposing live tool or environment objects.
Capability summaries are derived once at agent registration and reused by
execution-profile resolution.

`CayuApp.register_agent(..., tool_exposure_policy=...)` now applies that
contract end to end. Cayu resolves one registration-ordered exposure snapshot
before context pressure and official token counting, sends exactly that subset
through OpenAI, Anthropic, Chat Completions, Bedrock, and Vertex requests, and
reuses the same snapshot for generic retries and context-overflow recovery.
The default remains expose-all, and the runtime-owned structured-output tool is
preserved independently of application exposure.

A provider call for a registered tool absent from the frozen request is blocked
before `ToolPolicy`, approval, hooks, secret resolution, environment access, or
tool execution. Cayu emits typed `not_exposed_in_request` evidence without
arguments and appends a provider-valid error result. Compact snapshot authority
survives ordinary tool-round recovery and approval or user-input interruption,
while exposed calls continue through every existing authorization and execution
control.

Ordinary same-session resumes and queued dispatches now seed the next policy
request from the latest runtime-attested durable exposure profile. This keeps
phase selection and `profile_changed` evidence continuous across process
reconstruction; malformed or caller-authored lookalike evidence fails closed.

Tool exposure now emits the typed, content-minimized `tool.exposure.recorded`
evidence reserved by the public contract: profile and resolved fingerprint,
registered/ceiling/exposed counts, provider/model/step identity, and profile
transition state, with no separate tool-name list, tool definitions, arguments,
or policy reasoning. Application-selected profile ids are public and must be
stable non-secret labels.
Conversational request footprints advance to schema version 3 and bind the same
exposure summary to their keyed tool-manifest and cache-prefix identities.
`LLMJudge` now runs under a durable zero-tool capability ceiling even when its
registered agent has tools; adversarial candidate content cannot expose or
execute them. The paired tool-exposure economics fixture reports requests,
retries, cache categories, provider usage, quality, and cost without claiming a
universal winning strategy.

### Bounded fork groups are durable public runtime operations

`CayuApp.run_fork_group(...)` now freezes one terminal source checkpoint and
execution profile, runs 2-16 caller-named sibling sessions under bounded
parallelism and one causal budget, applies registered deterministic gates, and
admits only bounded allow-listed evidence to a tool-free evaluator. Lifecycle,
branch evidence, one validated selection, failures, and exact-request replay
are durable across in-memory, SQLite, and PostgreSQL session stores. Version 1
fails the group if any sibling or gate fails while preserving every branch for
inspection.
Atomic source digests prevent same-epoch source drift while siblings are being
created. Revision-fenced lifecycle publication prevents stale coordinators from
replacing a terminal result, reconciles lost commit acknowledgements, and sends
nonterminal child sessions through ordinary Cayu recovery after process loss.
One renewable store-backed execution claim prevents applications sharing a
store from concurrently resuming the same group. The tool-free evaluator's
exact execution profile is frozen before durable admission, and extension
failures are recorded only through bounded, secret-redacted diagnostics.

### Workspace mutation attribution is explicit and fail-closed

Workspace revision deltas no longer imply per-tool causality. Mutation receipts
now distinguish exclusive tool attribution, concurrent ambiguity, and
external/unknown changes; exact attribution requires stable resource identity
plus matching adapter-provided writer-isolation evidence at both ends of the
window. Overlapping in-process windows taint every participant, edits between
windows remain separate gap evidence, and direct workspace operations are
reconciled against independently observed endpoints with bounded content-free
projections. Private-argument and dynamic multi-call quarantine uses a fixed
projection with no direct-operation hashes, metadata, counts, or gap evidence,
cannot claim exact attribution, and clears the process-local gap baseline so a
later receipt cannot resurrect quarantined evidence. Built-in bindings default
to unknown isolation.

Terminal binding finalization records an unattributed delta from the last
durable after-window observation, and session forks explicitly report shared or
unproven workspace lineage without claiming an isolated derived revision. The
new evidence is additive JSON in existing events and round-trips through all
built-in session stores; no storage migration is required.

### Derived knowledge indexes publish exact identity and readiness

`KnowledgeEmbeddingIdentity` now binds every comparable derived embedding to
its exact entry revision, optional chunk, projection content, embedding space,
preprocessing, generator, and index representation. Independent
`KnowledgeIndexReadiness` events use compare-and-swap sequence fencing,
attempt fencing, idempotent operation replay, bounded authorized high-water
pages, and durable restart behavior across in-memory, SQLite, and PostgreSQL.
Lexical-only custom stores keep optional extension hooks instead of pretending
to support embeddings.

`InMemoryEmbeddingKnowledgeStore` and `PostgresEmbeddingKnowledgeStore` now keep
canonical publication provider-free and consume committed changes through
bounded `process_embedding_changes(...)` workers with independent change and
projection-record budgets plus deterministic continuation. The crash-safe order is
pending readiness, vector commit, ready readiness, then outbox acknowledgement.
Semantic/hybrid search is read-only, rejects non-ready or incompatible rows,
and returns machine-readable `KnowledgeIndexCoverage`; explicit bounded
backfill retries failed or missing projections.

Breaking storage revision 44 preserves canonical revision-43 knowledge and
adds the readiness event/current tables. PostgreSQL drops pre-identity
`cayu_knowledge_embeddings` rows during migration because vectors are derived
and cannot be safely assigned the missing identity. Stop older workers, migrate
once, and rebuild semantic projections; Cayu does not add a legacy read path or
fabricate readiness.

### Knowledge-store conformance is adversarial and explicit

The shared `KnowledgeStore` suite now registers each in-memory, SQLite, and
PostgreSQL backend with explicit lifecycle, durability, and optional-capability
claims. Reusable scenarios cover compare-and-swap revisions, owned immutable
results, authorization, atomic failure, exact change publication and page
metadata, portable ordering, lifecycle guards, projection readiness, and
embedding-space isolation. Ten deliberately broken adapters prove that each
scenario detects its intended defect instead of merely replaying happy paths.

Structured knowledge query terms and phrases that normalize to no searchable
tokens now fail at `KnowledgeQuery` construction consistently across backends.
PostgreSQL embedding startup also validates the revision-bound table's primary
key, checks, composite cascading foreign key, and required ready B-tree indexes;
a same-named but structurally incompatible index no longer survives `CREATE`,
`MIGRATE`, or `VALIDATE` startup. These checks harden the rebuildable revision-44
derived table and require no new storage revision.

### Knowledge revisions carry evidence and publish atomic changes

Knowledge create, append, and owned-publication operations now accept immutable
`KnowledgeEvidence` bound to an exact entry revision and optional exact chunk.
Lifecycle-only successors inherit and rebind evidence; content-changing
successors require callers to supply it explicitly. Evidence participates in
publication idempotency, bounded authorized reads, defensive copying, and
global collision protection across the in-memory, SQLite, and PostgreSQL
backends. Revision-42 publication receipts remain replayable after migration
when the retry carries no evidence; evidence-bearing requests never use the
older entry-and-chunks-only digest.

Every successful canonical knowledge mutation now emits one metadata-only,
before/after-audience-bound `KnowledgeChange` in the same transaction. A scope
that loses access still receives the removal/update signal needed to clear stale
derived state; expiration cannot make already-published work disappear. Bounded
change pages accept at most `MAX_KNOWLEDGE_CHANGE_LIMIT` records, expose an
honest store-owned accessible high-water mark, and reject
cursors beyond the current store sequence, while scope-bound leased consumers
can initialize from a full-scan baseline and provide fenced at-least-once
delivery with durable SQLite/PostgreSQL cursors. Failed writes, exact
publication replays, authorization denials, and rolled-back transactions emit
no change.

Change-consumer leases now use store-authoritative time (the database clock on
PostgreSQL), and durable acknowledgement receipts keep exact retries
idempotent even after the consumer advances again.

Evidence prefixes and multi-entry expiration changes now use portable scalar
identity ordering across the in-memory, SQLite, and PostgreSQL backends, so
bounded results and cleanup publication do not depend on database locale or
insertion order.

Owned PostgreSQL publication now acquires the global change-sequence lock only
after entry, chunk, and evidence payloads are written. Unrelated canonical
writes can proceed while a large payload is staged without weakening atomic
receipt and outbox publication.

Entry IDs and chunk IDs now have explicit portable UTF-8 byte limits shared by
canonical models, referenced evidence/receipts/changes, and every built-in
backend. Revision-43 migration rejects out-of-contract revision-42 identities
before applying DDL, avoiding backend-specific index failures. The revision-43
migration timestamp also preserves cleanup delivery for migrated entries that
expire after the outbox baseline without widening access to entries that were
already expired.

Breaking schema revision 43 adds evidence, ordered-change, and consumer-state
tables without rewriting revision-42 knowledge or fabricating historical
events. Stop pre-43 knowledge writers, run `cayu storage migrate`, and confirm
revision 43 before starting this build. Mixed revision-42/revision-43 knowledge
writers are unsupported.

### Model execution is attributable to complete immutable profiles

Execution-profile schema version 3 now binds provider-adapter behavior,
provider-visible request controls, context selection, knowledge injection,
compaction, live-state projection, application and invocation budgets,
structured output, and finalization semantics in addition to the existing
runtime, model, instruction, tool, policy, hook, environment, and effect
authority. Model dispatch, retries, recovery, explicit compaction, budget
evidence, structured-output repair, and model completion retain the frozen
governing profile instead of resolving mutable registrations again.

Governed request footprints advance to schema version 2 and reference that
profile without replacing their separate concrete-request evidence role.
Paired cost-quality comparison reports and bounded runtime-evidence reports
also advance to schema version 2 so attempt-level cost and runtime evidence can
identify the governing profile. These records remain content-free: they do not
retain raw prompts, knowledge, credentials, secrets, or unbounded schemas.

### OpenAI providers support native hosted web search

Applications can grant one registered agent immutable, typed
`OpenAIWebSearch` authority separately from Cayu-executed tools. Both the API-key
and experimental subscription providers send the native Responses API tool,
retain completed search actions, complete returned source lists, and URL
citations, and publish bounded provider-owned lifecycle evidence. Search calls
and unknown outcomes are accounted independently from tokens; current published
OpenAI call pricing is represented with source provenance, while unavailable
pricing remains explicitly unpriced. Strict budgets reject hosted search when a
hard per-response call ceiling cannot be established.

The public application manifest and generator plan advance from schema version
8 to version 9 because hosted-tool authority participates in registration and
execution identity. Regenerate committed manifests and generated plans; older
schema documents are not interpreted as granting hosted search. The server
contract advances from version 15 to version 16 because usage and cost
projections add hosted-search resource fields. Upgrade independently deployed
servers, packaged dashboards, and generated clients together. This change does
not require a storage migration. Paired cost-quality reports advance from schema
version 2 to version 3 so Compound and other comparison consumers retain hosted
search calls, unknown outcomes, and their separately priced resource cost.

### Fork and workspace branch authority fails closed

`CURRENT_CHILD` session forks now require an attributable application authority
decision even when their structural profile fingerprints are equal. Provider
changes always project and preflight portable history, including same-model
transitions. Local workspace branches now reject source/private per-directory
path-semantics mismatches and revalidate those semantics before publication.

### Knowledge updates are immutable, revision-addressed publications

Knowledge entries now retain immutable revisions and a transactional current
revision pointer. Every chunk belongs to an exact entry revision; compare-and-
swap publication permits only the successor of the caller's expected current
revision, and concurrent stale writers receive a typed conflict. Current and
historical reads are explicit, lifecycle changes append revisions, current-only
search and list operations cannot surface stale material, and hard deletion
removes the complete revision history. In-memory, SQLite, and PostgreSQL stores
share the same revision, access-control, publication-receipt, and defensive-copy
contract. Runtime context candidates and knowledge-facing APIs carry the exact
revision they expose.

Historical entry and chunk reads now require authorization for both the exact
snapshot and the logical entry's current revision, so an older active revision
cannot bypass a later tombstone or access restriction. Authorized principals can
still retire current material to `archived` or `deleted` without being granted
read access to the retired state; promotion and reactivation continue to require
destination access. Revision exhaustion is rejected before mutation with the
same error across all built-in backends.

The server contract advances from version 11 to version 12 because knowledge
entry and chunk representations now require their exact revision. Upgrade
independently deployed servers, packaged dashboards, and generated clients
together.

Breaking schema revision 42 replaces the mutable knowledge layout. Fresh stores
and stores whose legacy knowledge tables are all empty migrate normally. A
populated pre-42 knowledge store is refused before Cayu changes its schema,
data, or migration ledger: stop older writers, take an application-consistent
backup, and explicitly replace/reset that database rather than fabricating
revision history. Run `cayu storage status` followed by `cayu storage migrate`,
then confirm revision 42 before starting current workers. Mixed pre-42 and
revision-42 knowledge writers are unsupported. Revision-42 startup also verifies
the complete knowledge table, constraint, current-view, search-structure, and
required-index contract and refuses a damaged or conflicting schema before
serving reads.

### New runs use one atomic model target

`RunRequest.target` now accepts an optional exact
`ModelTarget(provider_name=..., model=...)`. It replaces the independent
`RunRequest.provider_name` and `RunRequest.model` fields, and the server run
body likewise replaces `model` with `target`. Update Python callers and
generated server clients together; the removed fields are rejected rather than
silently combined. When `target` is absent, Cayu continues to use the agent's
model and its configured provider or normal model-pattern/default routing.
The server contract advances from version 10 to version 11 for this breaking
request-body change.

### Local workspaces support bounded speculative branches

An exact `LocalWorkspace` can create an isolated branch from a complete observed
workspace revision. Branches expose the ordinary workspace API, deterministic
content-free net changes, explicit rollback, and conflict-checked publication.
Source snapshots, overlays, evidence, lifetime, and active branch count are
bounded; unsafe paths, symlinks, special files, stale baselines, and publication
conflicts fail closed. Other workspace backends remain explicitly unsupported.

Local branches can now opt into durable recovery by supplying a
`WorkspaceBranchStore`, such as the runtime's `SessionWorkspaceBranchStore`
adapter for `SessionStore`, together with stable branch, idempotency, run-epoch,
and binding authority. Custom session stores must explicitly support owned,
off-thread guarded commits before the adapter accepts them. Creation,
publication intent/progress, commit, rollback intent, rollback, conflict,
expiry, failure, and ambiguity survive process replacement across the bundled
in-memory, SQLite, and PostgreSQL stores. Recovery finishes only source paths
that still match an exact recorded before state, recognizes already-applied
paths, and reports durable ambiguity instead of guessing when external content
appears. Commit and rollback become terminal in the store before cleanup, and
stale run owners are fenced at the store boundary while a live binding resolver
holds an exact generation claim through mutation and terminal settlement;
binding replacement is rejected while that claim remains active. Custom claims
are transferable to later settlement tasks, and a failed release remains owned
and retryable instead of silently reopening the binding generation.
Publication keys remain permanently bound to their first bounded change-set
attempt, and provably pre-mutation inspection failures settle as replayable
failed results.

### Queued dispatch is bound to durable execution profiles

Queued tasks now retain the target session instance, source and required
execution-profile fingerprints, and deterministic terminal handoff identity.
Workers validate that authority before provider, tool, verifier, environment,
or other governed work begins, and restart reconciliation settles the exact
session/task outcome without redispatching completed work.

Breaking schema revision 40 installs the queued-dispatch handoff indexes and
marks the new profiled dispatch envelope as the only supported queued payload.
Before migrating a shared session or task store, stop every revision-39-or-
older dispatch producer and worker, take an application-consistent backup, and
drain or explicitly settle every legacy task type configured on any
`TaskStoreDispatcher`. This includes the default `cayu.dispatch` type and every
application-defined custom task type. After all older processes are quiescent,
run `cayu storage status` followed by `cayu storage migrate`, then confirm
revision 40 before starting current workers. Mixed revision-39/revision-40
operation and application-only rollback across this boundary are unsupported.

### Knowledge access is explicit and retrieval foundations are reproducible

Every built-in knowledge store operation now requires a principal-derived
`KnowledgeAccessScope`, supplied per operation or bound to the store. The scope
is enforced inside memory, SQLite, and PostgreSQL reads, searches, mutations,
chunk access, embedding paths, and publication-receipt replay. Runtime context
injection, knowledge tools, indexing, and review carry the same scope; callers
cannot widen a store-bound scope. Trusted maintenance must opt into the explicit
privileged scope. Global chunk-identity occupancy is authorized before it is
reported: foreign-scope collisions fail with `KnowledgeAccessDenied`, authorized
collisions use `KnowledgeChunkConflict`, and PostgreSQL serializes entry, chunk,
and publication identities in one server-side ordered lock batch. The remember
tool preserves those deterministic outcomes as access denial or publication
conflict instead of misreporting them as an outcome-ambiguous write, but only
after an authoritative receipt lookup proves publication absent. An unavailable
receipt lookup preserves the ambiguous outcome of post-commit extension work.

Breaking schema revision 41 stores the immutable authorization projection beside
each knowledge publication receipt so exact replay remains safe after an entry is
hard-deleted. Stop pre-41 workers and take an application-consistent backup before
migrating. Existing receipt authorization is deliberately not inferred or
backfilled: databases with populated pre-41 knowledge publication receipts must be
recreated. Empty receipt tables migrate normally. Run `cayu storage status`
followed by `cayu storage migrate`, then confirm revision 41 before starting
current workers. Mixed pre-41 and current knowledge writers are unsupported.

The memory foundation also adds a deterministic weighted reciprocal-rank fusion
primitive with bounded channel diagnostics, an explicit revision-reset/refusal
contract, and a versioned hermetic retrieval corpus and baseline runner. These are
foundation primitives; automatic curation and context placement remain separate
future layers. WRRF configuration construction now rejects values outside its
canonical durable fingerprint domain rather than accepting a configuration that
can fail only after fusion work. Its weight maps are deeply immutable, and
`model_copy(update=...)` revalidates updated values before returning a new
configuration.

### Deterministic completion verifiers publish independently owned decisions

Applications can register a side-effect-free deterministic completion verifier
under the exact durable verifier identity carried by a work contract, then ask
`CayuApp.verify_completion_proposal(...)` to evaluate a persisted proposal. The
runtime resolves and invokes the adapter with bounded immutable context, binds
its outcome to the live durable claim and frozen contract, and publishes the
decision without applying it to task or session state. Exact completed retries
reconcile from the store without requiring the process-local adapter, while
missing registrations, provider verifier kinds, malformed outcomes, conflicting
identity, and capacity exhaustion before claim mutation fail closed. A retained
cancellation-resistant adapter blocks overlap in the same app, while its live
renewed claim blocks other app instances. Runtime-minted execution-owner
generations prevent that live claim from dispatching the same verifier
concurrently elsewhere. A quiescent coordinator inherited through a process
fork mints a fresh owner generation before use; inherited active execution
state fails closed and requires rebuilding the app in that worker. The
execution timeout is bound into the exact claim,
and the owner renews its lease before dispatch and through execution,
cancellation-resistant draining, and publication. Process or renewal loss stops
that protection, so expiry permits at-least-once deterministic evaluation under
a fresh claim while fencing the stale owner from publication. Completed retries
continue to reconcile from convergent durable claim and decision indexes,
including revision-1 claims whose absent execution-owner field retains its
historical request digest.

This is the deterministic execution-to-decision slice of verified work. The
complete contract, task binding, attempt, proposal, verifier claim, decision,
and decision-application receipt lifecycle now persists with matching semantics
in `InMemoryTaskStore`, `SQLiteTaskStore`, and `PostgresTaskStore`. Session
execution authority is also durable: an ordinary admission and a contracted
task attachment race to one database-owned decision, and neither process
restart nor task terminalization weakens the winner.

Breaking storage revision 49 adds those task-store records and the task's
immutable contract reference. Stop all revision-48 and older task workers,
take an application-consistent backup, run `cayu storage migrate`, and confirm
revision 49 before starting current workers. Existing ordinary tasks migrate
with no contract binding. Mixed-version task workers and application-only
rollback are unsupported because older workers can complete contracted tasks
through an ordinary terminal entrance.

Every concrete `InMemoryTaskStore`, `SQLiteTaskStore`, and `PostgresTaskStore`
subclass must explicitly re-establish the documented
`verified_work_mutations_are_cancellation_quiescent` proof, including when its
public verified-work mutations are inherited unchanged. This prevents an
extension from inheriting settlement authority after replacing a readiness
hook, wrapper, or other dynamically resolved dependency.
Externally managed pools supplied to `PostgresTaskStore` must likewise retain
the exact built-in `AsyncConnectionPool` and psycopg async connection classes;
custom pool or connection subclasses, pool callbacks, callable configuration,
and custom row/cursor behavior cannot establish the built-in mutation
settlement proof and are rejected before database dispatch.
PostgreSQL worker-lease attachment, heartbeat, release, retry settlement, and
terminalization now sample PostgreSQL's current clock after the authoritative
row-lock wait. Reclamation uses one PostgreSQL clock value in its single
`FOR UPDATE SKIP LOCKED` statement, so locked rows are skipped and newly expired
leases become eligible on the next poll. Process-clock skew and a transaction
timestamp captured before contention can no longer renew, settle, or reclaim the
wrong lease. If caller cancellation encounters rollback, abort, or unwind
failures, the public `CancelledError` remains authoritative and carries one
bounded, redacted cause with that ordered settlement evidence through the runtime
API.

`CayuApp.apply_completion_decision(...)` now owns the public transition from a
durable verifier decision to task state. It validates the immutable authority
chain, including a decision-bound canonical digest of the complete final durable
verification claim (execution-owner generation, execution timeout, request
digest, attempt number, and final lease timestamps included), applies the
bounded exact request through the cancellation-quiescent store boundary, and
requires the atomic application receipt before returning the applied `Task`.
Exact retries replay the receipt snapshot, including after later task progress,
while checking that its immutable task fields and invocation provenance still
agree with the durable task authority. Ordinary acknowledgement loss reconciles
from that receipt; cancellation and process-control remain authoritative and
require a later exact retry. Accepted results remain application-owned and must
be reconstructed from a durable result reference whose digest matches the
accepted proposal.

This slice still does not add provider-backed judges, automatic result
resolution, or an automatic continuation worker.

### Durable tasks retain immutable invocation provenance

Tasks now carry a Cayu-minted invocation identity and their immediate execution
source through claims, worker replacement, scheduling, session attachment, and
terminal settlement. Server-created work also retains its verified product
subject, so replacement workers can reconstruct the exact task provenance
without trusting caller-authored identifiers.

Breaking schema revision 39 applies this immutable origin contract to tasks and
durable dispatch. Stop pre-39 workers and take an application-consistent backup
before migrating. Populated prerelease task stores cannot be assigned truthful
origins and must be recreated; empty task stores migrate normally. Run
`cayu storage status` followed by `cayu storage migrate`; migration passes
through revision 39 before the breaking revision-40 queued-dispatch boundary
described above. Confirm revision 40 before starting current workers. Generated
product-operation stores also add a required originating-subject column;
recreate populated prerelease product databases rather than fabricating that
authority.

### Worker task terminalization survives lost acknowledgements

Worker-claimed task completion and failure now commit a deterministic terminal
receipt atomically with task state. Exact retries return the original terminal
task after lease clearance, while changed worker, kind, payload, or digest
intent fails closed. `CayuApp` and `run_task_worker(...)` use a bounded,
observable retry helper for acknowledgement-ambiguous store failures; claim
loss, deterministic operating-system failures, validation, conflicts, and
cancellation are not retried. Success and uncertainty expose elapsed time and
applied backoff, and worker/dispatcher loops preserve an independently won
terminal outcome without masking same-key changed intent. This is durable task
state idempotency, not an exactly-once guarantee for external effects.

Additive schema revision 38 installs the receipt table in SQLite and PostgreSQL.
Revision-37 binaries remain compatible with the new table, but new built-in
task stores require revision 38. Current workers also require the breaking
revision-39 task-provenance boundary and the breaking revision-40 queued-dispatch
boundary described above; migrate the shared database through revision 40 before
deploying them.

### SQLite knowledge updates no longer scan the global FTS corpus

SQLite knowledge chunks now share a stable integer identity with their FTS5 rows.
Updating, replacing, pruning, or deleting one entry addresses only that entry's
chunks, avoiding corpus-sized work while the shared SQLite writer lock is held.
Search behavior and BM25 ranking are unchanged.

The knowledge migration advances the storage schema from revision 36 to
breaking revision 37 before the additive revision-38 task receipt migration and
breaking revision-39 task-provenance boundary, followed by the breaking
revision-40 queued-dispatch boundary. Stop older workers, drain every configured
legacy dispatcher task type as described above, take an application-consistent
backup, run `cayu storage status` followed by `cayu storage migrate`, and confirm
revision 40 before restarting current workers. The SQLite migration rebuilds
legacy knowledge FTS rows in one transaction. After an ambiguous interruption,
run `cayu storage status`: retry when revision 36 remains, or continue through
the later migrations when the complete revision 37 commit is already visible.
PostgreSQL records revision 37 without changing its knowledge schema, then adds
the shared terminalization receipt table at revision 38 before continuing
through revisions 39 and 40.

## v0.2.1

`v0.2.1` gives durable Cayu sessions an explicit execution identity and hardens
the boundaries that carry model, tool, workspace, knowledge, and MCP work across
retries, restarts, and operator-directed changes.

### Highlights

- Sessions persist a versioned execution profile covering their model target,
  provider configuration, tools, approval policy, environment, context policy,
  and other execution-critical inputs. Ordinary resume fails closed on drift;
  applications can explicitly inspect and authorize a compatible profile
  adoption at a safe boundary.
- Model targets can change through an atomic durable transition rather than
  mutating live agent configuration. The selected provider and model remain
  attributable through pending work, recovery, forked sessions, and restart.
- Every new session records immutable root-invocation provenance. Derived
  sessions preserve the same root while recording their immediate execution
  source, and the protected server derives authenticated provenance instead of
  accepting client-authored identity claims.
- Stdio and Streamable HTTP MCP transports now enforce validated per-message,
  aggregate-response, idle-timeout, and absolute-deadline limits. Ambiguous
  timeout, cancellation, and peer-failure paths fence or terminate uncertain
  shared sessions before reuse.
- Model-authored knowledge publication is operation-owned and receipt-backed
  across the built-in stores. Acknowledgement loss reconciles against immutable
  evidence instead of compensating by deleting a shared deterministic entry.
- Active `SyncBinding` generations reserve both source and target workspace
  identities before provisioning, copy, and sync-back work. Bounded workspace
  reads, runner listings, attachment limits, S3 deletion, provider cleanup,
  reasoning-state replay, child-session identity, virtual-egress authority, and
  internal event namespaces also fail closed at their public boundaries.
- `cayu cloud` validates application slugs, distinguishes local and production
  contexts, reports bounded deployment diagnostics, and waits for Agent service
  health before declaring a deployment ready.

### Upgrade from v0.2.0

Python 3.11 or newer is required. Stop all `v0.2.0` workers and take an
application-consistent backup of every configured SQLite or PostgreSQL store
before upgrading. Do not run mixed `v0.2.0` and `v0.2.1` processes against the
same stores.

The storage schema advances from revision 34 to revision 36. Revision 35 adds
operation-owned knowledge-publication receipts and is a mixed-writer boundary.
Revision 36 requires immutable invocation provenance on every session. Because
existing populated `v0.2.0` session stores never recorded that provenance, Cayu
cannot truthfully infer it: archive any evidence that must be retained, then
recreate each database containing session rows. Do not edit the database or
fabricate invocation identities to bypass this guard. Empty databases and
databases without session rows migrate normally. Run `cayu storage status` and
`cayu storage migrate` against every explicitly configured session store,
budget ledger, eval store, task store, and knowledge store, then confirm
revision 36 with no pending migrations before starting `v0.2.1` workers.

The server contract advances from version 9 to version 10. Upgrade independently
deployed servers, packaged dashboards, and generated clients together. Portable
trajectory documents advance from schema version 2 to version 3; regenerate
version-2 exports from their authoritative source rather than assigning invented
invocation provenance during loading.

### Verification

Install `cayu==0.2.1` in a clean environment and verify `cayu version`,
`cayu cloud --help`, and `cayu check --json`. Use fresh stores for a
current-contract smoke test, then exercise a durable session through restart,
an explicit model or execution-profile transition, durable knowledge
publication, one bounded MCP call, and the packaged `/cayu/` dashboard.

## v0.2.0

`v0.2.0` makes Cayu's durable runtime directly operable as a production agent
system: completed sessions can become portable eval corpora, delayed tasks stay
store-gated until their durable availability time, the packaged dashboard owns
the authenticated eval workflow, and the reserved `cayu cloud` command now
supports deploying and operating Cayu Cloud applications.

### Highlights

- Runtime-native evals can capture bounded terminal session evidence, promote
  it into reviewable portable corpora, execute those corpora through the same
  trusted local core, publish durable results, and compare compatible runs.
- The packaged `/cayu/` dashboard adds corpus management, durable eval-run
  control, result inspection, comparisons, CI export, and delayed-task
  visibility against the versioned server contract.
- Tasks accept an optional UTC availability time. The store remains
  authoritative for eligibility, future work cannot be claimed early, and
  concurrent workers retain the existing lease and acknowledgement-loss
  guarantees.
- `cayu cloud` provides authenticated login, deployment, environment and secret
  management, service inspection, rollback, and bounded operational evidence
  without changing Cayu's root Python exports or durable runtime schemas.
- Public boundary objects, provider traffic, tool results, vault values,
  operational evidence, and eval records now take owned portable snapshots and
  reject malformed, non-finite, or otherwise unsafe input before it can become
  durable or externally dispatched.

### Hardening since v0.2.0rc1

- Interaction transitions, terminal recovery, gated-loop replay, and public
  operation settlement now require positive, lifecycle-scoped durable evidence
  and recover safely after acknowledgement loss or worker replacement.
- Runner preflight validates commands and environment removals before secret
  resolution, while the worked Modal runner applies the same ownership and
  hostile-input boundary before SDK dispatch.
- Tool output suppression, duplicate interaction model names, malformed webhook
  signatures, and out-of-domain eval comparison scores now fail explicitly and
  safely.
- Provider, approval, budget, usage, vault, and eval evidence is detached from
  caller-owned mutable inputs, and HTTP/retry metadata is validated before use.

### Upgrade from v0.1.0

Python 3.11 or newer is required. Stop all `v0.1.0` workers, take an
application-consistent backup, and upgrade independently deployed Cayu servers,
dashboards, generated clients, and workers together. Do not run mixed `v0.1.0`
and `v0.2.0` processes against the same stores.

The storage schema advances from revision 29 to revision 34. Run
`cayu storage status` followed by `cayu storage migrate` against every
explicitly configured SQLite or PostgreSQL session store, budget ledger, eval
store, and task store, then confirm revision 34 with no pending migrations
before starting `v0.2.0` workers. Revision 34 includes the durable eval catalog,
run lifecycle, and delayed task availability contracts.

### Verification

Install `cayu==0.2.0` in a clean environment and verify `cayu version`,
`cayu cloud --help`, and `cayu check --json`. Use fresh stores for a clean
current-contract smoke test, then exercise a representative durable session,
eval run, delayed task, and `/cayu/` dashboard journey before production
rollout. This release does not claim mixed-version operation or compatibility
with durable stores created by earlier prereleases.

## v0.2.0rc1

This is the first release candidate for `v0.2.0`. It freezes the current
runtime, evaluation, and delayed-task contracts for clean-install and
fresh-store validation. Bug fixes discovered during candidate testing may ship
in `v0.2.0rc2`; otherwise the final release should change only version and
release metadata.

### What this candidate validates

- Completed production sessions can become bounded, reviewable eval
  trajectories and portable corpora, then run through the same trusted local
  execution core used by runtime-native evals.
- The packaged dashboard supports the complete authenticated eval workflow:
  corpus management, durable run control, result inspection, compatible
  comparisons, and dashboard-to-CI export.
- Provider cursor recovery, interruption handling, lossless trial outcomes,
  evidence provenance, and cost-aware comparisons remain explicit across
  retries, restarts, and unavailable evidence.
- Durable tasks can be scheduled with an optional UTC availability time. The
  store remains authoritative for eligibility, concurrent claimers cannot take
  future work early, and the dashboard reports durable configuration without
  trusting the browser clock.

### Candidate verification

Install `cayu==0.2.0rc1` into a clean Python 3.11-or-newer environment and use
fresh SQLite and PostgreSQL stores. Fresh stores initialize at schema revision
34, which includes durable eval state and delayed task availability. Verify
`cayu version`, run `cayu check --json`, execute the current-contract test suite,
and exercise a representative durable session, eval run, and delayed task. This
candidate does not claim mixed-version operation or compatibility with durable
stores created by earlier prereleases.

## v0.2.0.dev0

### Upgrade from v0.1.0

The storage schema advances from revision 29 to revision 33. Revision 30 is an
additive PostgreSQL index rebuild that keeps direct-child session traversal in
the same bytewise identifier order as SQLite, memory, and Python validation.
Revision 31 records runtime ownership of fresh-input markers and raises the
compatibility floor so older workers cannot expose that private marker as
ordinary event payload. Revisions 32 and 33 add the durable eval catalog, run
lifecycle tables, and their target-leading query indexes.
Stop all `v0.1.0` workers, take an application-consistent backup, and run
`cayu storage status` followed by `cayu storage migrate` against every
explicitly configured SQLite or PostgreSQL session store, budget ledger, and
eval store. Confirm revision 33 with no pending migrations before starting
post-release workers, and do not run mixed `v0.1.0` and development-version
workers.

### Completed production sessions can become eval trajectories

`trajectory_from_session(...)` now reconstructs a completed or failed durable
session tree as the same serializable `Trajectory` used by fresh runtime-native
evals. It is a read-only historical-evaluation boundary: it invokes no provider,
tool, environment, hook, recovery, or store mutation. Descendants are admitted
from their durable start or fork sequence relative to each parent's terminal
sequence, so pre-terminal background work is retained and later forks are not.
Every admitted node must independently satisfy the exact terminal-evidence
contract, one retained-evidence budget applies to the complete admitted tree,
and closure changes, contradictory lineage, or a configured tree-depth overflow
fail with stable typed errors. Node construction validates each captured record
once and validates the completed tree once, so deep accepted trees do not incur
quadratic subtree copying.
Retained start and fork origins must carry the runtime-owned direct parent, and
fork source identity must agree with that parent, including when a child session
is promoted directly as the selected root. Matching caller-authored lineage
text is discarded at built-in store ingestion and cannot satisfy promotion.
Admission reads a dedicated minimal lineage projection containing only
pre-hydration-bounded structural identity and payload-free origin identities.
Session display fields and event JSON are never selected, so a large field or
event on a later excluded child cannot amplify capture memory or introduce
backend-specific JSON transport eligibility. The caller's session limit applies
only to retained evidence; excluded candidates use a separate hard-bounded
minimal lineage walk. PostgreSQL schema revision 30 rebuilds the existing
direct-child traversal index with bytewise identifier collation, keeping Unicode
keyset order identical across PostgreSQL, SQLite, and memory without sacrificing
the bounded index scan.

Workspace and artifact probes now retain capture provenance. Missing historical,
uncaptured, operationally unavailable, or truncated evidence that cannot decide
an assertion produces `unavailable` with no score; only a successfully observed
missing file or empty artifact scope is negative evidence. A partial artifact
listing may still prove a positive match from the records it did retain.

The new `ToolsCalledInOrder` assertion checks the exact model-requested tool
sequence from the durable transcript, independent of parallel scheduler event
order. The trajectory capture operation deliberately does not read current
workspace or artifact state, re-execute the application, or publish a
control-plane route. The reviewable corpus-promotion layer described below
builds on this immutable evidence boundary without changing its meaning.

### Terminal session evidence has one bounded snapshot boundary

The in-memory, SQLite, and PostgreSQL session stores can now load a completed or
failed session's exact event prefix through its matching terminal event together
with the attributed transcript, pending publication marker, run/lifecycle
boundaries, and exact complete-result byte accounting. The optional operation
rejects active, interrupted, incomplete, contradictory, or oversized evidence
with stable typed errors and never presents truncation as a complete capture.
SQL stores preflight bounded counts and conservative stored lengths before
hydrating payloads; PostgreSQL bounds the full raw JSONB representation,
including whitespace-heavy JSON string content, under a distinct transport
policy that can reject JSONB-expanded scientific-notation values without
changing the canonical limits applied to accepted evidence;
custom stores must explicitly opt in only when they provide the same guarantees.
`trajectory_from_session(...)` now consumes this storage boundary for
production-session trajectory capture. Revision 31 stores an explicit SQL proof
bit for runtime-attested fresh-input markers and raises the compatibility floor to
31. Pre-31 readers do not know that the marker is private and would expose it as
ordinary event payload, so mixed-version operation and app-only rollback are
rejected. Rows written before migration receive a false proof bit and cannot gain
runtime authority from payload text alone. The authenticated, stateless dashboard
workflow described below consumes this boundary without adding draft persistence
or eval execution.

### Production sessions produce reviewable portable eval candidates

`build_promotion_candidate(...)` projects an eligible terminal trajectory into
one bounded, redacted, identity-free candidate containing caller input, public
assertion evidence, a default editable regression case, and the source app,
release, evidence-policy, and optional pricing fingerprints. The runtime marker
binds the exact caller-input transcript range and its canonical message digest;
missing, altered, or caller-authored attribution fails closed before promotion.
Approval continuations, resumes, later input, structured output, non-user roles,
and non-text input remain explicit unsupported cases rather than being silently
rewritten into a different replay contract.

Candidates can be rescored against the captured evidence, converted to the
existing portable corpus model, and exported as deterministic UTF-8 JSON.
Authenticated servers can opt into two stateless control-plane routes and the
packaged dashboard workflow: an operator can edit the suite, case, captured
input, and every portable assertion; preview the exact draft against freshly
reconstructed evidence; and download the same candidate as canonical corpus
JSON. Editing invalidates the preview, and a changed source snapshot or server
policy requires a fresh candidate. The workflow does not persist drafts,
publish corpora, execute evals, or invoke providers, tools, environments, or
hooks.

### Portable eval corpora run through one trusted local execution core

Downloaded corpus JSON can now run immediately through the Python SDK or
`cayu eval run --corpus PATH`. A corpus-mode `EvalPlan` binds one validated
`CorpusTarget`: the local `CayuApp`, message-free request base, bounded trusted
bootstrap, application release, evidence policy, optional PriceBook, and hard
execution ceilings. Corpus data supplies only bounded user text, allowlisted
assertions, and trial settings. Target, policy, applicable pricing, and limits
are checked before provider dispatch, and fresh trials reuse the existing eval
runner and assertion bridge rather than a parallel evaluator.
Public target-key and application-release identity must survive the target app's
workload-secret redaction boundary unchanged. Selected-suite cost assertions
share one compile-time pricing binding instead of repeatedly validating and
fingerprinting the trusted PriceBook.

Successful execution returns a versioned `CorpusExecutionResult` containing the
sanitized `PublishedEvalRun` and the fresh release plus bounded public
AppManifest. Deterministic bounded JSON and standalone HTML retain only the
assertion projection's app-redacted output preview (at most 16 KiB per trial and
2 MiB per run), its evidence/truncation state and digest, and stable Cayu-owned
trial reason codes. They expose no raw output, omitted preview suffix, trajectory,
session ID, provider payload, exception text, credential, or executable target
state. AppManifest changes during execution reject publication. Typed
compatibility checks require an equal evaluation contract while intentionally
allowing different target releases and manifests.

`compare_corpus_execution_results(...)` now returns the complete immutable
comparison graph: typed compatibility reasons, bounded baseline/current
summaries, per-case deltas, and canonical status/score regressions. Deterministic
JSON and standalone HTML render that same graph. Incomparable results never
manufacture regression rows. `cayu eval report` and `cayu eval compare`
auto-detect direct and corpus result documents, and corpus-mode CI distinguishes
evaluated failure (`1`) from an unavailable decision (`2`). Release CI executes
the complete hermetic dashboard-to-local journey from the built wheel, while
`examples/evals_release_acceptance_live.py` provides the credential-gated
OpenAI/Anthropic application proof.

The CLI also provides `cayu eval validate`, `cayu eval inspect`, and atomic
`cayu eval merge`. Equal definitions deduplicate; a same-ID content conflict
rejects unless replacement is explicit. The merged document is fully validated
before replacing its destination. Corpus and `CorpusExecutionResult` schemas
start at version 1. Their nested `PublishedEvalRun` advances to version 2 because
trial output evidence and expanded diagnostic codes are required; version 1
published runs are unsupported and are not guessed or migrated.

### Eval results preserve every trial and explicit evidence gaps

Repeated eval cases now retain an ordered `EvalTrialResult` for every concrete
session instead of overwriting case evidence with the last trial. Each trial
keeps its own output, assertion outcomes, snapshot-derived usage and cost,
duration, diagnostics, completeness, and optional trajectory; case and run
aggregates are validated projections of those retained results. Fresh evals
drain runtime hooks and then read the same bounded terminal snapshot described
above for completed and failed sessions. Direct Python evals that end
interrupted remain supported through a narrower runner-owned path: every root
event's emitted durable sequence and type must match inside the store's same
bounded snapshot; interrupted descendants additionally prove their direct
parent in that fresh run tree. This does not make historical interrupted
sessions eligible for promotion. Incomplete, unsupported, contradictory, or
oversized evidence is `unavailable`, while execution/evaluation failures are
`error`. Neither outcome has a numeric score, and aggregate scoring never drops
it or coerces it to zero. Evidence completeness is independent from assertion
input availability, so an exact retained trajectory remains complete when, for
example, a cost assertion lacks pricing.

Third-party session stores that do not opt into
`supports_terminal_session_evidence` will report fresh eval trials as
`unavailable` after this upgrade rather than using non-atomic session, event,
and transcript reads. Store implementations must provide the exact bounded
operation before those evals can pass. Interrupted evals additionally require
the narrower `supports_runner_owned_interrupted_evidence` operation; the runner
does not fall back to generic unbounded event or transcript queries.

Persisted `EvalRun` baselines advance from schema version 3 to version 7: version
4 introduced the lossless result graph and explicit assertion outcomes, while
version 5 records conclusive workspace and artifact capture provenance in every
retained trajectory, and version 6 binds portable assertion results to the exact
definition revision evaluated. Version 7 can carry the portable corpus execution
contract a trusted executor fixes before provider dispatch, so a completed run
cannot be published under different case input, evidence, pricing, or suite settings.
Contracted saved runs now enforce the exact requested trial count, and complete
lossless and published trials require their exact aggregate usage so conclusive
usage observations cannot be detached from their source summary. Portable
evidence and published results preserve aggregate token totals above the
IEEE-754 safe-integer assertion ceiling as canonical decimal strings while
marking those token assertions unavailable. Rootless trajectories cannot acquire
conclusive model-step or usage evidence from detached fields, and publication
rejects cost metadata that contradicts its retained exact cost summary. Raw
published-result admission also fails closed on malformed branches before any
later oversized graph can be constructed.
Standalone trajectory documents advance from version 1 to version 2 for the same provenance
contract. Older prerelease documents must be regenerated; Cayu does not migrate
or guess at their meaning.

## v0.1.0

### Remote environment allocation fails closed without exact recovery

Custom remote `EnvironmentFactory` implementations can now declare a stable
provider and adapter-generation scope and use Cayu's durable allocation context
to persist intent before provider dispatch, reconstruct lost acknowledgements,
and atomically publish exact reconnect identity. A durable `REAPING` fence now
makes cleanup mutually exclusive with publication, so a losing concurrent
worker preserves an allocation another worker already published and cleanup
crashes resume idempotently. Pending allocation state is portable, bounded,
secret-free, fork-isolated, and conformance-tested across the in-memory,
SQLite, and PostgreSQL session stores.

The bundled Microsandbox, E2B, and Lambda MicroVM virtual-egress adapters do
not yet expose all primitives required for exact create-or-lookup recovery and
race-safe cleanup. New `CREATE` operations through those adapters now fail
before adapter setup or provider mutation instead of retaining a crash window.
Existing same-resource reconnect remains available where supported, and Docker
creation is unchanged because its allocation is process-local.
Custom `SandboxEgressAdapter` implementations must now explicitly classify
`process_external_allocation`; an undeclared classification fails new creation
before adapter preparation rather than assuming process-local behavior.

### Root checkpoints now have an explicit compatibility boundary

Runtime-owned checkpoint objects now carry `checkpoint_schema_version=1`.
Versionless prerelease checkpoints remain readable and are stamped on their
next normal write or JSONL export. Unsupported future or malformed roots fail
before governed work with the bounded public
`CheckpointCompatibilityError`; checkpoint contents are not rendered in the
error. In-memory, SQLite, and Postgres restart tests cover pause/resume and
future-version rejection. Future root-schema rollouts must deploy compatible
readers everywhere before enabling newer writers.

Custom session stores must forward the optional `checkpoint_root_guard` on
bounded pending-action and interruption-marker reads and invoke it before
interpreting nested checkpoint state.

### Configured projects can start server processes

`cayu serve` now discovers the nearest configured project, constructs its
process-scoped application once, wraps it with the supported `create_server`
contract, and starts uvicorn with explicit host and port values. Open access
requires `--dev`; protected deployments resolve a configured auth target.
Incomplete-session startup recovery remains off unless bounded statuses and an
inactivity threshold are both configured. Autoreload and multi-process server
workers remain out of scope.

### Configured projects can start worker processes

Projects can declare multiple `[tool.cayu.workers]` targets and run one
with `cayu worker <name>`. Entrypoints receive the fresh app and a cooperative
stop event, while the CLI owns SIGINT/SIGTERM handling and a bounded shutdown
grace period. The entrypoint still chooses `TaskStoreDispatcher.run_worker`,
`run_task_worker`, or another supported loop; the CLI does not infer task or
recovery policy.

### Oversized terminal tool results can be externalized before model context

`CayuApp(tool_result_projection_policy=...)` now provides an opt-in,
default-off projection boundary after tool-result redaction and hooks but before
terminal event and transcript publication. The built-in
`ArtifactExternalizingToolResultPolicy` keeps results at or below configured
byte/token-estimate thresholds unchanged. Above a threshold it stores the
redacted UTF-8 text through the active environment's `ArtifactStore` and gives
the model a bounded preview plus a typed reference whose generated
`ReadFileTool` instruction includes a policy-safe byte limit. Application
manifests fingerprint the built-in
policy's configured thresholds, preview bound, and token-estimation method.

Direct and MCP tools share the same boundary. Structured data, existing
artifact/file references, error disposition, and effect evidence are
preserved. Artifact identities are deterministic for a
session/tool-call/content combination, and local/S3 stores reuse exact
caller-supplied identities. Recovery republishes the original projected
terminal result without rerunning the tool. Store failure emits a bounded
explicit result rather than falling back to the oversized content. Events,
logs, and OpenTelemetry spans report only content-free sizes, identities,
hashes, estimation method, and outcome. Projection persistence has a bounded
deadline so a non-settling store cannot hold interruption open indefinitely;
completed projection evidence is published before concurrent cancellation is
redelivered. `LocalArtifactStore` stages complete deterministic artifacts before
atomic publication and repairs only matching legacy partial writes. The S3
store settles an in-flight threaded content upload and metadata commit before
redelivering cancellation, so timeout cannot leave an undiscoverable
content-only object; if metadata fails, it first removes content created by the
cancelled invocation.

The public application manifest and generator plan advance from schema version
6 to version 7. The runtime manifest now identifies the configured projection
policy, so enabling it changes the application fingerprint. This feature does
not change the storage schema or server contract.

### Interaction durability advances the server and storage contracts

First-interaction admission, queued delivery, and lifecycle closure now retain
stable interaction identity across acknowledgement loss and recovery. Session,
event, transcript, and checkpoint transitions that decide an interaction
outcome use store-atomic, idempotent publication. Setup failures retain a
durable incomplete-authority marker: source input remains inspectable, but the
session cannot be resumed, forked, or compacted without an authoritative system
prefix. Interaction-transition receipts content-bind their historical result
snapshots so acknowledgement-loss replay cannot be changed by later session state.

Storage revision 26 adds interaction attribution and a database-maintained,
gap-free per-session transcript ordinal. This prerelease contract is a clean
break: migration rejects a populated pre-26 Cayu session database instead of
rewriting its events or transcript. Recreate the Cayu database before starting
the new workers. Empty schemas migrate normally.

The server contract advances from version 4 to version 5. Mutation SSE
envelopes, API event records, and transcript records now require
`interaction_id`, and interaction-scoped endpoints are part of the published
OpenAPI surface. Regenerate clients and deploy independently hosted dashboards
with the matching server version.

### Runtime events now expose schema-aware public identities

The server contract advances from version 5 to version 6. REST event records and
SSE frames now expose stable `cayu_event_<sequence>` aliases instead of raw
durable event IDs; `Last-Event-ID` uses
`session_id:cayu_event_<sequence>`. The server accepts previously issued raw
markers and raw exact-event filters as read-only transition compatibility, but
never returns those private IDs. All new runtime event IDs reserve the `cayu_event_`
namespace, and ambiguous imported raw/public aliases fail with `409` rather
than selecting the wrong durable record. Regenerate clients and deploy the
matching dashboard with the server.

Actionable approval, user-input, and tool linkage uses field-scoped aliases
such as `cayu_event_<sequence>:input_id`. Resolution routes bind those aliases
back to private authority only within the exact request session; previously
issued raw linkage remains transitionally accepted but is no longer returned.

Secret-colliding session and interaction identities use versioned HMAC aliases
backed by a durable store index. Persistent deployments must provision the same
explicit alias keyring on every worker; key rotation retains old keys until their
issued aliases can be retired. Persistent stores durably select the active key,
backfill before cutover, and fence stale workers; retired active key IDs cannot be
reactivated. The reserved `cayu_authority_` namespace cannot be used for newly
caller-selected session IDs, including fork destinations.

Built-in event payload keys and fixed controls are now trusted only under their
exact owning `EventType`. New secret-bearing linkage authority fails before
persistence, while legacy and custom events remain observable through a
fail-closed public projection. Durable accounting, claims, watcher cursors,
retry, replay lineage, and genuine duplicate suppression continue to use the
original private event identity.

Mutation correlation IDs that contain configured workload secrets now fail
before dispatch. Custom event sinks receive only the canonical public event;
private deduplication and trace correlation remain restricted to exact built-in
sink adapters.

### Tool-approval resolution identities are mandatory

The same version-5 protected control-plane contract requires approval and
manual-recovery requests to carry the exact durable
`tool_round_id` and gating `tool_call_id` (and approval requests retain their
`approval_id`). These fields fence stale operators and retries from resolving a
newly paused round that happens to reuse a session or call identifier.
The atomic approval claim also stores a canonical digest of the requested
decision, reason, metadata, and resolver actor. A retry must match that digest
before environment setup or tool execution. The digest stays in private
coordination state rather than approval events, whose bounded audit metadata is
never replay authority. The final approval-close receipt retains the same
identity even when a run limit closes the round. Durable
resolution activity written before request digests existed is not reconstructed
from bounded or redacted audit events; those partial legacy resolutions remain
interrupted for explicit reconciliation.

Resume input retained behind an unresolved recovered tool round remains private
until the round's single grouped tool-result message is durably published.
Receipt replay and restart recovery then materialize that input idempotently
after the result, so publication validation, transcript ordering, and retry
identity remain consistent across a process stop.

Manual tool recovery remains a separate authority boundary. It may record an
externally verified terminal outcome, but it can close the approval
automatically only when every call in the round is already terminal and a
modern approval intent supplies the original digest. If a sibling remains,
recovery leaves the approval interrupted and the exact original approval
request must be retried before that sibling can execute.

Regenerate clients and upgrade independently deployed Cayu servers and
dashboards together. The dashboard intentionally rejects version 4 instead of
submitting an approval without the complete durable identity.

### Session topology reads are bounded and indexed

The protected server now exposes a session-focused topology read for control
planes. It returns a bounded ancestor path and batched, independently pageable
direct-child branches without loading transcripts, event histories, arbitrary
metadata, or output payloads. In-memory, SQLite, and PostgreSQL stores share the
same stable ordering, cursor, cycle, and node-limit contract; custom stores opt
in explicitly through `supports_session_topology`.

Schema revision 24 adds the composite parent/creation/id traversal index. The
revision is additive and keeps revision 23's compatibility floor, but current
session stores require `cayu storage migrate` before topology is advertised.
Built-in topology reads use bounded per-parent index seeks instead of ranking or
scanning complete child sets. Requests have independent identifier, cursor, and
256 KiB body ceilings; validation errors never echo rejected values, all
expected responses are private/no-store, every loaded graph cycle is rejected,
and structural identity fails closed if redaction would make it ambiguous.

The existing all-in-one causal-budget summary also fails clearly when its fixed
session, event-count, 4 MiB pre-hydration event-input, or response-size safety
ceilings are exceeded. Custom stores must implement the optional byte-bounded
event query primitive for that legacy summary; the server does not substitute
an unsafe full-payload read.

### Task topology linkage is bounded and optional

The protected session-topology read can now project tasks attached to explicitly
selected visible sessions and independently expand direct child-task branches.
Typed edges distinguish session ancestry, task ancestry, and durable
task-to-session links. Task data is deliberately limited to bounded identity,
display, lifecycle, assignment, and timestamp fields; inputs, results, errors,
metadata, status payloads, workers, and leases never cross this boundary.

Task linkage is an optional `TaskStore` capability. The response distinguishes
`not_configured`, `unsupported`, and `available`, so a custom or absent task
store does not disable the session graph. Session and task reads are separate
store snapshots and the response reports `cross_store_atomic=false` rather than
implying an atomic relationship that the storage contract does not provide.

Schema revision 27 adds `(session_id, created_at, id)` and
`(parent_task_id, created_at, id)` task indexes. It is additive and preserves
revision 26's rolling-compatibility floor, but current built-in task stores
require `cayu storage migrate` before topology reads are available. Each request
can select at most 50 visible session-to-task branches and 50 task-parent
branches; every branch has its own stable cursor, while one shared 500-task-node
and 4 MiB response ceiling prevents high-fan-out workflows from becoming an
unbounded control-plane read. The shared cap is allocated before stores hydrate
candidate rows, parent chains are validated under explicit depth/node bounds so
pagination cannot hide cycles, and PostgreSQL uses locale-independent task-ID
tie-breaking equivalent to SQLite and the in-memory store.

### Usage rollups can attribute bounded work to sessions

`POST /api/usage/rollup` now accepts an optional `session_group_limit`. This
adds at most 100 deterministically ordered per-session usage groups, current
lifecycle state, and an exact aggregate remainder for matching sessions omitted
from detail. With a price book, the response also includes per-session cost
summaries and an exact cost remainder. Shared workflow totals remain
authoritative, and either pricing projection fails conservatively to an
unevaluated state instead of publishing a partial cost as exact.
The complete serialized rollup has an authoritative 4 MiB ceiling, so repeated
session or pricing identities cannot amplify a bounded request into an
unbounded response; oversized results fail clearly with `413`. Usage request
bodies are separately limited to 3 MiB and invalid price books are never
reflected in validation responses. The sanitized response retains the existing
`HTTPValidationError` shape. Per-session identities are limited to 1,024 UTF-8
bytes and are rejected before SQL stores return them across the database
driver. Secret-bearing session authority fails closed instead of being redacted
into an ambiguous drill-down identity. Usage-request currency identities are
limited to 64 UTF-8 bytes before store work, preventing them from being
multiplied into an oversized in-memory per-session cost projection. Custom
session-store projections are bounded and canonically reconstructed before the
server publishes or prices them, so trusted Pydantic copies cannot bypass
nested field validation or projection invariants. The reconstructed result is
checked against the requested window, opt-in fields, group/input limits, and
exact shared/per-session pricing consistency. Exact session-pricing projections
also carry an internal commitment to each retained session's complete
price-relevant input identity. Post-construction mutation therefore cannot swap
cost between equal-usage sessions whose model, billing identity, or effective
pricing date differs.

The extension is opt-in. Existing usage-rollup requests keep the previous store
work and response semantics; SQLite and PostgreSQL execute the session and
session-aware pricing projections only when callers request them, inside the
same store-local read snapshot as the shared rollup.

### The dashboard can inspect bounded execution workflows

Session detail now links to an accessible, shareable Workflow view when the
configured session store advertises topology reads. The view combines the
bounded ancestor path, independently pageable child-session and task branches,
typed ownership links, lifecycle status, and one causal-budget usage/cost
rollup. It does not load event or transcript histories and does not fan out one
detail request per node. Native disclosure buttons, ordinary links, and a
nested list keep the complete interaction keyboard-readable without a canvas or
graph dependency.

Expansion and loaded-scope filters round-trip through bounded URL state, while
opaque continuation cursors remain transient. Manual refresh is always
available; automatic topology and usage refreshes are independently
single-flight, pause in hidden documents, and stop once every loaded node is
terminal. Missing task topology, dashboard pricing, or the complete Workflow
capability is reported explicitly instead of becoming an empty graph or a zero
cost. The packaged Chromium contract covers navigation, keyboard expansion,
all branch continuations, URL restoration, superseded-request cancellation,
visibility and terminal refresh behavior, capability denial, and absence of
history/per-node request fan-out.

### Usage and dashboard pricing advertise separate capabilities

The control-plane contract now reports usage aggregation independently from the
dashboard's configured pricing catalog. Deployments without a default price book
can still expose activity and token totals, while cost estimation remains
explicitly unavailable instead of hiding the complete Usage surface.

This advances the server contract from version 3 to version 4 because
`surfaces.usage` is a new required capability. Generated clients must be
regenerated from the current OpenAPI document. Independently deployed dashboards
and servers must be upgraded together; the dashboard rejects a version 3
response before resolving navigation or starting route-specific API requests.

### Coding workspace mutations are conditional and reviewable

Complete workspace reads now expose opaque file revisions, pageable byte
offsets, continuation metadata, and diagnostic SHA-256 digests. Every built-in
workspace implements atomic create-if-missing, conditional replacement, and
conditional deletion through the same cooperative resource/path lock.

The first-party coding surface adds exact multi-edit, conditional delete,
explicit create-versus-overwrite, and bounded Git status/summary/diff
inspection. Applications can require model-visible tests and repository-change
review without granting an unrestricted shell or relying on a racy
read/check/write sequence.

### Chat Completions usage dialects are explicit and subclass-safe

`ChatCompletionsProvider` now accepts an explicit `usage_dialect`, preserves
subclass class-attribute declarations, and continues to recognize Google's AI
Studio endpoint automatically. Explicit configuration wins over subclass and
endpoint-derived values, so Vertex AI and gateways can declare their accounting
semantics without unsafe hostname inference.

`UsageDialect.GEMINI` preserves hidden thinking included in provider totals and
bills it once as output. Ordinary OpenAI-compatible endpoints remain on
`UsageDialect.OPENAI`, where unexplained token-total mismatches fail closed.

### SyncBinding retains live target ownership until exact-owner cleanup

`SyncBinding` reservations no longer expire solely because a bind is old. Each
bind now owns every resolved target—fixed or factory-created—through an opaque
process-local generation keyed by the workspace's stable resource identity
until that exact generation finalizes successfully or is explicitly abandoned.
The registry is shared across `SyncBinding` instances in the process, so two
bindings that resolve to the same filesystem cannot both clear or copy it.
For direct binding callers, failed or cancelled finalization remains retryable,
and stale cleanup cannot release a newer owner. Cancellation received after a
workspace mutation is dispatched is propagated only after that mutation becomes
quiescent, preventing an old worker from changing files owned by a retry.

The prerelease `state_ttl_s` constructor option has been removed because elapsed
time cannot prove that a process-local binding stopped using its workspace.
Direct `SyncBinding` callers whose lifecycle will not invoke `finalize()` must
call `abandon()` with the matching `BoundWorkspace`; runtime-managed sessions
perform binding finalization through the environment lifecycle. If terminal
runtime finalization fails, Cayu first persists the failure evidence and then
abandons the exact process-local generation because no public retry handle
survives the terminal run. If that evidence cannot be confirmed durable, the
lifecycle retains the generation and its stable pending failure event; a later
ordinary environment setup drives a bounded cleanup settlement and must settle
that event before the target can be reused. A managed-egress wrapper explicitly
refuses abandonment until its adapter positively proves workspace mutations are
quiescent. Detachment alone is not proof: the wrapper escalates to a terminal
stop, suspension, or removal boundary when necessary before releasing
ownership. Managed-egress finalization fences new command dispatch, drains
already-admitted commands—including provider cleanup explicitly reported as
deferred—and then syncs while the runner-backed workspace remains readable.
Uncertain settlement remains fenced until explicit operator verification, and a
managed runner rejects overlapping stateful bindings before target mutation.
Synchronization commands remain settlement-accounted, while a target already
terminated by command cleanup is retired only after provider quiescence and
continues reporting its sync failure. An ordinary partial copy or deletion
failure retains that exact readable generation for retry instead of closing it
and falsely reporting a later success.
Independently detached guest processes remain outside managed dispatch
accounting. Interrupted Microsandbox
allocations stop without deleting their reconnect identity and restart that
same incarnation on continuation. Retained cleanup uses one owned task per
exact owner, caps concurrent settlements at 16, and bounds cleanup polling
during unrelated environment admission. Total active, in-flight, and retained
environment owners are capped per process by
`CayuApp(max_environment_lifecycle_owners=...)` (default 256); the explicit
cleanup drain and server shutdown path retain rather than cancel work that
outlives their bounded wait.

The public application manifest advances from schema version 5 to version 6 and
includes `runtime.max_environment_lifecycle_owners`, so deployment fingerprints
reflect this admission policy. Generator plans use the same version marker and
therefore also advance to version 6.

### Terminal recovery restores missing event evidence

Incomplete-session recovery now accepts every session status and repairs a
`completed`, `failed`, or `interrupted` session whose status commit survived
without its matching terminal event. Repair fences stale publishers, reuses a
stable event identity, reconciles lost append acknowledgements, preserves
pending interruption identity, and clears that marker only after matching
evidence is durable. Existing matching evidence is reused, while duplicate or
operation-bound status-conflicting terminal events fail closed.
Normal terminal publication also reconciles its preassigned event identity
before treating an ambiguous acknowledgement or side-effect delivery failure as
a run failure, so durable completion evidence cannot produce a second,
contradictory failure event.

The bounded inspection reads at most two lifecycle/terminal records and scopes
them to the latest start or resume. A terminal fork's `session.forked` event is
treated as a complete branch baseline rather than a run that needs a fabricated
terminal event. Resumed and continuation runs also claim a durable operation
identity atomically with their running status; their terminal events carry
`session_run_operation_id`, allowing repair to distinguish an old run's
terminal event even if the new lifecycle event never committed. The temporary
checkpoint marker is removed only after terminal evidence is durable, and
session deletion is rejected while that publication remains incomplete. A
later resume or continuation checks the bounded evidence even when an initial
run has no operation marker, repairs any missing prior boundary, and only then
atomically claims a new run. Recovery reconstructs missing approval, user-input,
and interrupted tool-round terminal payloads from their durable checkpoints
before reporting or repairing the pending action. When recovery fences an
abandoned resumed or continuation run, it atomically transfers that operation
marker to the recovery epoch so terminal publication retains the same logical
identity without accepting writes from the crashed owner.
Operator-supplied ordinary tool-round recovery uses the same operation
boundary: it creates an identity for a terminal continuation or transfers the
identity of a stale running continuation. Recovery rejects a marker that claims
an epoch newer than the durable session instead of normalizing impossible
state.

Healthy terminal rows do not consume the batch recovery result limit. Candidate
discovery is explicitly paginated: each call inspects no more than
`inspection_limit` sessions through no more than ten keyset store pages of at
most 1,000 rows each and returns
`IncompleteSessionsRecoveryPage.next_cursor` when more candidates remain.
Reusing that opaque cursor with the same recovery semantics reaches older
incomplete evidence without retaining or loading the complete history. The
cursor is bound to statuses, inactivity boundary, reason, and metadata; result
and inspection page sizes may change between calls. Repair does not run
providers, tools, or terminal hooks, and a recovered failure explicitly reports
when its original error details were not durably available.

Durable session identifiers are now limited to 2,048 UTF-8 bytes, ordinary
session-list cursors to 4,096 bytes, and batch-recovery cursors to 8,192 bytes.
The versioned built-in list cursor encodes its identifier component, and the
recovery cursor separately encodes the complete custom-store cursor, so every
value accepted at one layer is guaranteed to fit the next layer even when it
contains JSON metacharacters. Custom `SessionStore` implementations must keep
their opaque `SessionListResult.next_cursor` values within the 4,096-byte
contract. Session-list cursors issued before this release candidate are
ephemeral and must be restarted rather than reused after upgrade.

### Aggregate usage remains exact across every public summary

Session, causal-budget, multi-session, and usage-breakdown summaries now use the
same lossless aggregate counter model as usage rollups. Aggregate counters are
nonnegative decimal strings on the JSON wire, so cumulative values can exceed
signed 64-bit and JavaScript safe-integer ranges without becoming invalid or
losing precision.

This advances the server contract from version 2 to version 3. Generated clients
and independently deployed dashboards built for contract version 1 or 2 must be
regenerated from the current OpenAPI document and deployed with the matching
server. The dashboard continues to fail closed when its exact contract version
does not match the server.

The read-only session CLI schema advances from version 1 to version 2 for the
same reason. Aggregate counters in `session show` and `session usage` JSON/JSONL
output are now canonical decimal strings; per-model-call counters remain
signed-64-bit JSON integers. Every `session usage --jsonl` record declares
`schema_version="2"`, including model-call, unmatched-ledger, and aggregate
records.

Persisted `EvalRun` baselines advance from schema version 2 to version 3.
Version 3 stores identity-free aggregate usage, represents counters above signed
int64 as canonical decimal strings, and enforces Cayu's durable-JSON contract on
both export and import. Versions 1 and 2 were prerelease formats and must be
regenerated rather than being interpreted under the new shape.

### OpenAI subscription token rotation is power-loss durable

Successful local subscription login, refresh, and logout now synchronize both
the private replacement file and its containing directory before returning.
Credential writes are serialized across processes, reject unsafe filesystem
objects, create missing auth-path components privately, re-establish directory
entry durability after interrupted setup, preserve complete post-replacement
state when a later durability check fails, and fail explicitly on platforms
that cannot provide the required local durability primitives. Refresh rotation
also retains native worst-case capacity allocation until the complete new
credential record has been written.

### MCP manifest drift keeps one durable accepted baseline

MCP manifest authorization now keys history by a stable connection identity
instead of by the changing tool-name set. Every later fingerprint is evaluated
against the last accepted baseline, including after application reconstruction
or session retention, and a blocked candidate cannot become model-visible or
replace that baseline. Baseline comparison, decision-event publication, and
accepted replacement are atomic across built-in stores, so concurrent
reconnects cannot silently install competing versions.

Authorization now binds that source manifest to the exact registered adapter
exposure sent to the model. Subsets, aliases, and provider-facing contract
changes participate in drift evaluation, while adapter execution remains bound
to the original advertised MCP tool and session. Durable baselines retain
separate bounded evidence for the advertised and exposed sets.

`McpServerSpec.connection_id` provides the required stable
application/tenant/endpoint namespace whenever MCP tools are exposed to a run.
It remains optional for direct discovery/client use, but runtime admission fails
closed when it is absent because a display name cannot safely identify a
connection across applications or tenants. Manifest baselines and events persist
only validated hashed identities, fingerprints, fixed-size opaque tool IDs,
bounded change summaries, and policy outcomes. MCP-enabled custom stores without
the new atomic history capability fail closed. After upgrading from revision 21,
review each MCP connection and set an explicit `connection_id` to authorize its
new first-seen namespace.

SQLite and PostgreSQL deployments must run `cayu storage migrate` before
deploying this change. Schema revision 22 is breaking because older workers do
not maintain the accepted-baseline table and therefore cannot safely share a
database with revision-22 workers.

### Read-only session inspection

`cayu session` now provides bounded, backend-neutral `list`, `show`, `usage`,
`tools`, `events`, and `transcript` views through public `SessionStore`
contracts. Project-aware target resolution uses explicit flags first, followed
by `CAYU_DATABASE_URL`, typed `[tool.cayu.session_store]` configuration, and the
single conventional local target `data/cayu.db`. Inspection validates existing
SQLite/PostgreSQL schema and never creates or migrates it.

### One local SQLite convention

New scaffolds and examples place Cayu's SQLite-backed runtime state in
`data/cayu.db`. The product-level name reflects that the database can hold
sessions, tasks, knowledge, budgets, and other Cayu runtime state. This is a
prerelease convention change: Cayu does not discover
or migrate alternate filenames. Applications that intentionally use another
path must configure or pass that path explicitly.

### Server configuration is explicit and source-agnostic

`create_server(...)` now accepts one immutable `ServerConfig` covering access,
API and dashboard exposure, generated docs, CORS, and lifecycle policy.
`deployment_name` is descriptive metadata only: changing it never relaxes or
tightens server policy. Applications must select `AuthenticatedAccess` or
deliberate `OpenAccess`, and can resolve credentials through any external
provider before constructing the config.

The optional `cayu[server-settings]` extra adds typed environment and `.env`
loading with explicit constructor values taking precedence over process
environment and dotenv values. `create_server(...)` requires the resolved
configuration, and `mount_cayu(...)` requires an explicit access policy; there
is no parallel flag-based policy path. See [server
configuration](server-configuration.md) for profiles, external authentication,
and settings variables.

### Virtual egress supports GitHub-style CLI tokens

`VirtualCredentialSpec(credential_kind="opaque_token")` now brokers opaque
credentials carried as `Authorization: token …`. This enables unmodified
GitHub CLI REST calls inside an enforced Linux runner while keeping the real
token in the trusted vault/broker path. The new
[GitHub CLI recipe](recipes/github-cli-virtual-egress.md) includes a runnable
no-key proof, a strict REST-read profile, and the separate authorization
requirements for GraphQL and mutations.

Authorization parsing now preserves the presented scheme. A mismatched,
unsupported, or omitted scheme is denied before vault resolution; a value in
Cayu's virtual namespace therefore cannot fall through to credentialless
egress merely because it used an unrecognized scheme.

Virtual-egress leaf certificates now use a validity window of at most 398 days
for compatibility with platform trust paths that reject longer-lived leaves,
including the macOS certificate verification exercised by Go-based CLI
clients. Session CA lifetime remains unchanged.

### Recovery takeover fences checkpoint ownership atomically

`SessionStore` now requires
`fence_run_and_transform_checkpoint(...)` for checkpoint-authorized ownership
takeover. The operation must persist its checkpoint transform and increment the
session run epoch in one transaction, and must roll back both changes if the
transform returns `None` or raises. Built-in in-memory, SQLite, and PostgreSQL
stores implement the contract.

Cayu uses this boundary when replacing an expired incomplete-session recovery
claim. A stale recovery owner can no longer refresh session activity between
claim replacement and epoch fencing, reopen an unfenced retry window, and race
a session fork or explicit compaction. If the database commits a replacement
claim but its acknowledgement is lost, Cayu reconciles the preassigned claim,
releases it, and preserves the original error instead of leaving a new live
lease that blocks an immediate retry.

Initial incomplete-session recovery now uses the same atomic boundary: status
and inactivity checks, claim persistence, and run-epoch advancement occur in
one transaction. The claimant renews the exact claim after the storage result
is observed, so a delayed caller whose lease was already replaced cannot fence
or clean up the replacement worker. Ambiguous initial acknowledgements are
reconciled by claim identity and expected run epoch before cleanup.

Manual ordinary-tool recovery now installs and heartbeats the same durable
claim while atomically transitioning to `RUNNING`. Multiple API workers can no
longer fence one another while reconciling the same call. A takeover that finds
the prior owner's terminal result closes an orphaned live session to resumable
`INTERRUPTED` state instead of restoring an ownerless `RUNNING` status. Lost
claim heartbeats actively stop an in-flight continuation, finalize live session
state, and abort environment setup before the run fence and claim are released.
The recovery supervisor remains an interruptible process-local owner while
event delivery is paused. A bounded durable-status watcher uses the last
pre-claim terminal-interruption event as its baseline and stops only for an
`INTERRUPTING` state or a newer explicit operator terminal event. It therefore
observes an interruption requested through another API worker without mistaking
the recovery's own resumable `runtime_interrupted` transition for an external
stop. Completion does not depend on the stream consumer asking for another
event, and both paths preserve the durable operator-interruption reason instead
of replacing it with a generic stream-abandonment outcome.
An interruption that becomes durable before the recovery claim is acquired now
wins that atomic race; recovery finalizes the existing stop request instead of
reopening the session as `RUNNING`, including when another worker has already
finished the transition to `INTERRUPTED`. A pending operator-interruption marker
also remains authoritative when recovery first loads the session: if the
operator path crashed before writing its terminal event, recovery completes
that event and leaves the tool outcome unapplied for an explicit later retry.
Manual recovery is rejected atomically while a descendant interruption cascade
is incomplete, matching the existing resume and fork guard. The interruption
takeover carries a preassigned durable claim identity and expected run epoch,
so a lost database acknowledgement can be reconciled before cleanup rather
than stranding the terminal event or adopting a replacement worker's claim.
Cancellation during an ambiguous claim acknowledgement remains authoritative
while the preassigned claim identity is reconciled and cleaned up.
Explicit recovery-stream closure also reports a finalization or fence-release
failure to its caller instead of silently consuming that cleanup failure.

### Runtime publications have durable atomic receipts

Session stores now expose `publish_runtime_publication(...)` for crash-sensitive
model-step and tool-round commits. One call atomically publishes the detached
transcript batch, replacement checkpoint, new events and their side-effect
handoffs, session timestamps, and an insert-only, store-owned receipt. The
receipt binds the logical publication identity, interaction identity, intent,
source fences, transcript cursors, and ordered referenced events. References are typed ID-and-canonical-
content-digest pairs derived from an exact `Event`; missing or changed referenced
content fails before mutation and on every receipt load or replay.

An exact replay loads and verifies the receipt before lifecycle, run-epoch, or
transcript-cursor fences and performs no writes. Reusing a publication id with a
different request, or finding a malformed receipt, fails closed with
`SessionRuntimePublicationConflict`. Existing `publish_session_operation(...)`
callers and custom-store overrides keep their current API; the new protected
runtime-publication hook is non-abstract and raises `NotImplementedError` until a
custom store implements the atomic boundary.

Custom stores must also implement the model-completion prepare, complete,
promote, active-load, and exact-abandon hooks with equivalent atomic semantics.
These hooks remain non-abstract so an out-of-tree store can still be imported
and migrated, but `CayuApp` has no lossy compatibility fallback: the first
model/tool publication raises `NotImplementedError` until the store is updated
and passes the shared publication conformance suite.

Model completions also have a pre-dispatch staging boundary. A stable logical
step id now groups per-dispatch stage ids and monotonic dispatch ordinals. Stores
atomically publish a discoverable active-stage marker with preparation, retain
immutable terminal provider material, and claim one per-logical-step winner in
the same transaction as the final runtime publication. Zero-message attempts,
live retries, recovery under a newer run epoch, concurrent promotion, and lost
acknowledgements therefore converge without a second provider request or a
second authoritative completion. `dispatch_authorized` is true only for the
transaction that inserted a new preparation, so exact, superseded, stale-epoch,
and already-published preparation calls cannot authorize another provider call.
Superseded terminal evidence remains durable but cannot publish or refresh
current-run liveness; historical receipt replay preserves unrelated later work.
When the runtime can prove provider dispatch never began, it can atomically
abandon that exact active preparation. The content-bound abandonment tombstone
is acknowledgement-loss replayable and cannot remove a re-prepared generation;
terminal evidence, a logical-step winner, or a publication receipt makes
abandonment fail closed. Once provider dispatch may have begun, the stage remains
in flight for conservative recovery instead of being discarded.

Every authoritative assistant message, including an ordinary text response, is
projected through workload-secret redaction before the atomic model publication.
For tool-strategy structured output, schema validation still uses the original
in-memory arguments. The same publication checkpoints a typed, redacted
validation snapshot, and live closure, store verification, and recovery bind to
that snapshot instead of recomputing validity from the altered durable value.

Tool lifecycle lookup is now scoped by stable round identity as well as provider
call id, bounded before materialization, and backed by the existing
pending-action index. Reused call ids from older well-formed rounds are ignored;
roundless or malformed evidence still fails closed. SQLite maintenance, session
fork, and session deletion also preserve active stage and pending-round
evidence. Because version-1 receipts bind positional transcript and event
material, SQLite pruning and transcript compaction conservatively retain every
session that already contains a receipt.

### Session metadata updates preserve runtime-owned state

`SessionStore.update_metadata(...)` and
`PATCH /api/sessions/{session_id}/metadata` now replace only user-authored
metadata. Top-level `cayu:` entries and `subagent` are runtime-owned: built-in
stores preserve them atomically and reject callers that include them in a
replacement. An empty object clears the user-authored portion without erasing
tool-policy or subagent-coordination state.

This is an intentional prerelease contract correction. Clients that previously
round-tripped the complete `ApiSession.metadata` object must omit runtime-owned
entries from the PATCH body. Custom `SessionStore` implementations must preserve
the same boundary; `copy_session_user_metadata` and
`replace_session_user_metadata` provide the shared validation and transactional
merge primitives.

### Experimental OpenAI subscription sign-in

Developers can now run local Cayu agents using their own ChatGPT subscription
through `OpenAISubscriptionProvider`. `cayu auth openai login` provides a PKCE
localhost flow, `--headless` provides device authorization, and
`status`/`logout` manage Cayu's private `~/.cayu/auth.json` credential store.
Access tokens refresh before expiry and requests use Cayu's existing Responses
stream/tool normalization.

This is an experimental Codex-backend integration rather than a documented
OpenAI Platform API. Requests identify themselves as Cayu and never adopt a
first-party Codex originator. The adapter stops at upstream rejection, exposes
no embeddings or remote token-counting capability, and treats flat-plan usage
as unpriced. See [OpenAI subscription authentication](openai-subscription.md)
before enabling it.

### Custom SessionStore implementations must support event side-effect handoffs

The `SessionStore` interface now requires durable persisted-event side-effect
claim, finish, exact lookup, and inspection methods. Implementations must
support retry deadlines through `retry_delay_seconds`, forward inspection
pagination through `after_sequence`, and raise
`PersistedEventSideEffectClaimLost` when a stale worker tries to finish a
replaced claim. Built-in in-memory, SQLite, and PostgreSQL stores implement the
new contract; custom stores must be updated before constructing `CayuApp` with
this release.

Server adapters bound their initial side-effect recovery wait to 30 seconds by
default through `event_side_effect_startup_timeout_seconds`; unfinished durable
handoffs continue through lifecycle recovery. A transient final delivery-ack
write no longer fails already-completed runtime work, and the built-in
OpenTelemetry sink suppresses recent in-process event replays.

### Microsandbox guest networking now defaults to deny-all

`MicrosandboxRunner.create(...)` now supplies `microsandbox.Network.none()`
when `network` is omitted. This is an intentional prerelease security change:
code running in a newly created Microsandbox no longer receives ambient guest
network access by default.

Applications that intentionally relied on implicit unrestricted networking
must opt in visibly:

```python
from cayu import MicrosandboxRunner
from microsandbox import Network

runner = await MicrosandboxRunner.create(
    "trusted-network-client",
    network=Network.allow_all(),
)
```

Do not use unrestricted networking for untrusted model-authored code without a
separate enforced egress boundary. Existing callers that pass `Network.none()`,
a Cayu virtual-egress policy, or another explicit provider policy retain their
chosen behavior. `MicrosandboxRunner.from_existing(...)` cannot retrofit a
policy; the creator of the existing sandbox owns its creation-time network
contract.

### New Run selects a registered agent explicitly

The control-plane New Run page now loads the registered agent inventory before
enabling submission. A single registered agent is selected automatically;
applications with multiple agents require an operator choice, and every
dashboard run sends that exact agent identity to `/api/run`. Empty inventories
and initial inventory failures keep execution disabled with a useful
explanation. The page also states that this direct runtime operation does not
invoke application-specific entrypoints or orchestration.
