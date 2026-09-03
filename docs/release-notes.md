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

### Durable dispatch ownership now uses store time and reusable fencing

Cayu now has a small internal durable-operation ownership component for
feature-owned journals. Strict bounded records retain the logical operation,
claim, owner, fencing generation, and store-stamped lease evidence; typed
transitions distinguish acquisition, exact acknowledgement replay, renewal,
expired takeover, release, settlement, fencing, phase advancement, identity
conflict, and indeterminate acknowledgement. The consuming store performs the
comparison and timestamping atomically. Workers can request a bounded duration
but cannot supply `now` or an expiry cutoff. Deterministic in-memory and SQLite
tests and transactional PostgreSQL conformance cover concurrent claim and
takeover races, stale settlement, clock behavior, and pre-commit versus
commit-before-error recovery. Exact committed release and settlement retries
return their original acknowledgement even after the feature phase advances;
foreign operation identities still fail before phase classification. A skewed
worker-authored feature timestamp cannot delay a store-time claim, takeover, or
live-owner publication.

Memory-intervention runtime dispatch is the first production consumer. Its
execution journal advances to schema version 2 and replaces feature-local owner
and lease fields with the shared fenced value. There is no version-1 ownership
reader or dual-write path: old prerelease execution records are rejected rather
than reinterpreted as unowned. Exact retries in one worker reuse their logical
claim, heartbeat writes are limited to one per 10 seconds under a 30-second
lease, foreign-owner polling is capped at four reads per second, and stale
generations cannot publish control or terminal evidence. Those publications
renew the exact generation and then recheck lease liveness using store time
inside the final compare-and-set, so exact expiry without takeover is also
fenced. Ordinary phase compare-and-set cannot rewrite ownership. Custom
execution stores must implement both store-time boundaries; the base class
provides no lossy CAS fallback.

Runtime-generated session creation also has a bounded, secret-free durable
claim reference. A purpose-separated HMAC and explicit key identity bind the
exact request without giving durable-storage readers an offline oracle for
guessed prompts or secrets; key material is never serialized. Memory
interventions derive a restricted subkey from their rotated restart-stable
request key, while generated workflow children use attempt-local random keys
and a claim identity that remains stable across fresh attempts. Trusted helpers
reconstruct the opaque process-local claim only for the exact request, session,
invocation provenance, logical operation, and key. One typed authentication
boundary verifies running sessions with transient input and terminal sessions
after that input has been cleaned up, including the final prepared-message
digest, while distinguishing missing, foreign, incomplete, malformed, and
tampered evidence. Both memory interventions and generated workflow children
use this boundary. Task, provider-operation,
compaction, and independent-fork ownership were audited but intentionally not
migrated because their receipts, post-dispatch quiescence, or checkpoint/run-epoch
fences are materially different from a transferable renewable lease.
Custom memory-intervention runtime runners must accept the new restricted
reference-key argument and keep it ephemeral; it cannot mint the durable parent
request fingerprint.

### A credential-free campaign exercises causal memory end to end

The checked causal-memory reference corpus now runs repeated `as_declared`,
`automatic_recall_off`, `omit_items`, and adversarial `replace_items` trials from
one verified AgentSnapshot. Its deterministic provider still traverses Cayu's
ordinary recall, admission, checkpoint compaction, context exposure, durable
intervention, portable assertion, and paired report paths. The runner rejects
cross-trial overlay reuse and stale, expired, irrelevant, or unauthorized
provider exposure, then proves exact recovery in a new process with provider
dispatch disabled. Mixed recovered/new runs audit new requests by exact trial
identity, and abnormal terminal executions remain explicit report rows instead
of being survivor-filtered. The published result claims only a measured output
change under the declared intervention, not hidden model use or universal
causality.

### Storage migration preflights authority before schema writes

`cayu storage migrate` now fixes the exact input revision and ordered migration
path, validates every clean-break condition, and checks the configured public-
authority alias keyring before schema mutation. Migrations from an initialized
database must acknowledge each exact pending breaking revision with repeated
`--acknowledge-breaking REVISION` arguments. The revision-73 prerelease recall
boundary additionally accepts `--reset-empty-recall-state`: the preflight proves
all six checkpoint/delivery tables are empty, then rebuilds only those tables
from the selected Runtime's migration definitions. Populated state still fails
closed.

SQLite migration now creates an application-consistent retained backup by
default, migrates and validates a backup-derived staging database, and atomically
publishes it only after alias reconciliation, `PRAGMA integrity_check`, and
`PRAGMA foreign_key_check` succeed. `--backup PATH` selects the retained backup;
`--waive-backup` is the explicit no-retained-backup escape hatch. PostgreSQL
migrations require either an operator-attested `--backup-sha256 SHA256` or
`--waive-backup`; their already revision-resumable execution remains under the
schema advisory lock.

Successful migrations emit a versioned receipt containing the backup digest,
input and output revisions, complete step list, breaking acknowledgements,
secret-free authority fingerprint, exact runtime-build provenance, migration
identity, execution mode, and backend validation checks. Alias keys and database
credentials are never included. SQLite serializes receipt recovery with atomic
publication and rejects output paths that alias its database, sidecars, receipts,
or retained backup, rejects symbolic-link targets, and reuses an authenticated
retained backup after a pre-publication retry only when a fresh locked snapshot
proves that the live input is unchanged. PostgreSQL commits the pending receipt
with revision progress and replays the original evidence when final delivery is
retried.

### Docker fan-out can share exact read-only inputs

`ImmutableInputProjection` and `ImmutableInputStore` now materialize a bounded
regular-file tree once and durably reference it from many strict Docker coding
environments. Reuse identity includes exact content and executable modes,
format and limits, target, policy, Runtime compatibility, and authorization
scope. The store converges across threads and fresh processes, recovers exact
orphaned publications after acknowledgement loss, rejects mutation with typed
evidence, and reports content-free size, reference, reuse, wait, and cleanup
diagnostics.

Docker accepts only manager-issued immutable mounts. Before exposure it verifies
the daemon's exact read-only mount state and proves that a root write is refused.
The mutable `/workspace` finalizer cannot traverse or publish immutable inputs;
it records durable container-closing intent, closes the container, and then
releases durable references. Fresh recovery reconciles that intent against an
exact Docker lookup. Deterministic allocation replay also resolves the same
container rather than issuing a duplicate. The 100-environment fan-out path uses
one physical materialization rather than 100 workspace copies.
Bindings now distinguish shared read-only projection, explicit bounded mutable
copy, ordinary workspace materialization, and unsupported behavior so callers
cannot silently downgrade the requested guarantee.

### Recovery cleanup now has shared finite deadlines

`CayuApp` now supervises recovery cleanup through a shared
`RecoveryCleanupPolicy`. The default per-step deadline is 30 seconds, the
default sequence deadline is 120 seconds, and at most 256 active or
outcome-unknown tasks are supervised per app. A hanging stream close, finalizer,
heartbeat stop, claim release, fence release, environment cleanup, or
supervisor shutdown can no longer hold its caller indefinitely. Independent
cleanup explicitly grouped into one phase receives a bounded concurrent
attempt. Ordered steps remain dependency barriers and resume under a reserved
continuation only after an outcome-unknown predecessor settles. The original
Runtime failure remains authoritative and typed timeout evidence is attached to
it. Failure ordering remains stable if cancellation arrives during a later
cleanup step, and cancellation carries deadline evidence for every retained
independent owner. Same-step ordering preserves whichever event was already
authoritative: a settled failure remains ahead of later cancellation, while a
failure produced in response to forwarded cancellation remains behind it.

Timed-out work remains strongly owned until it settles, and its operation's
durable claim, fence, run-operation, provider-operation, or lifecycle marker
continues to be the restart authority. Capacity exhaustion fails before an
untracked cleanup task starts. Operators can inspect content-free counters with
`CayuApp.recovery_cleanup_status()` and use
`CayuApp.drain_recovery_cleanups(...)` during bounded shutdown; drain follows
active-to-retained-to-continuation transitions for the whole configured grace
period. Late retained-owner failures are classified separately from successful
late completion and logged without their potentially sensitive message. The
application manifest and generator plan advance from schema version 14 to 15,
and the manifest fingerprints the complete cleanup policy.

### Durable knowledge enrichment keeps curation off the foreground path

Applications can now submit an already-authorized, bounded `LearningBatch` to
`KnowledgeEnrichmentQueue` and process it explicitly in a fresh
`KnowledgeEnrichmentWorker` process. The job reuses Cayu's task retry-series,
`KnowledgeCurator`, governance, publication receipts, and knowledge store; it
does not start a scheduler, scan transcripts, select a provider, or create a
parallel source of job truth. Exact operation replay returns the durable result
without rerunning semantic components. A durable process-bound dispatch fence is
committed before semantic work, and a bounded immutable semantic preparation is
committed before publication. Retries therefore reconcile without redispatching
the generator or evaluator; a missing preparation after dispatch is exposed as
typed `semantic_outcome_unknown` evidence. Profile- and queue-derived worker routes
prevent one configured worker from claiming another domain's jobs, and
`process_next(...)` distinguishes idle queues from durably rejected malformed
work. Profile drift, changed payloads, stale leases, feedback without
independent-source policy evidence, and
oversized requests fail closed. In-memory, SQLite, and PostgreSQL task stores
share the same contract. See the credential-free
[`durable_knowledge_enrichment.py`](../examples/durable_knowledge_enrichment.py)
fresh-process example.

### SyncBinding fan-out has an aggregate staging ceiling

`SyncBinding` copy-in and copy-back now share a configurable process-local
concurrency and byte governor (four transfers and 512 MiB by default). Tar data
uses private file-backed spooling and explicit raw binary streams for local and
Docker runner workspaces, eliminating whole-archive base64/JSON copies on that
path. Exact revision-aware concurrent fan-out can reuse one sealed archive when
source revisions, content, executable modes, paths, the complete exclusion
policy, format, and every copy limit match; mutable or weak sources fail closed.
`SyncBinding.staging_snapshot()` exposes bounded, content-free queue,
reservation, reuse, cleanup, and wait metrics.

### Task continuations retain elected worker authority across recovery entrances

An interrupted task claimed with `claim_interrupted_task_continuation(...)` now
remains fenced by an exact caller-generated handoff generation as well as its
worker through session resume, tool approval, user-input continuation, manual
tool recovery, and provider-operation recovery. Generate the opaque claim token
with `new_interrupted_task_continuation_handoff_id()` and pass it as `handoff_id`.
An exact retry replays a live commit-before-acknowledgement claim; a stale token,
expired lease, or different worker fails closed. Each typed continuation request
accepts `task_worker_id` and `task_handoff_id`; missing, stale, or secret-bearing
authority fails before provider or tool execution, and authority is rechecked
after session admission before side effects begin.
Successful continuations pass both values through heartbeat and every task
terminal path instead of using the workerless completion path.

SQLite and PostgreSQL add a partial unique index for live handoff generations
and permanently register the digest of every claimed generation. The in-memory
store maintains equivalent constant-time indexes. This makes each caller token
one-use even after its task rotates authority or becomes terminal.
Direct worker completion/failure, idempotent terminalization receipts, and
owner-lost cancellation reconciliation all bind the exact generation, so a
restarted process reusing the same worker name cannot adopt its successor's
authority. Legacy non-recovery terminalization and reconciliation digests remain
compatible.

Accepted provider dispositions attached to an elected task remain pending when
generic session recovery has no worker credential; the elected typed
continuation must finish them. If a closed approval fails after its task is
terminalized, the exact immutable terminalization receipt now authenticates a
retry that deterministically publishes the task and session failure boundaries.
Workerless direct provider and approval failures use the matching immutable
failed-task snapshot plus authoritative absence of that claimed-worker receipt,
so process loss between task and session failure remains recoverable without
letting a caller omit an elected worker identity.

The same recovery boundary now covers ordinary failures after any attached-task
continuation has begun. The runtime records a deterministic, redacted version 2
failure marker containing its identity, invocation execution-profile
fingerprint, terminal payload, and turn summary in the task terminalization,
and all typed continuation entrances
recognize its exact elected-worker receipt or proven workerless origin. A fresh
process must resolve collaborators matching that retained profile before it may
run terminal hooks or environment finalization. A retry finishes the same
deterministic `interaction.failed`, failed `turn.completed`, and
`session.failed` evidence without invoking another provider or tool;
already-dispatched model work is classified conservatively as having an unknown
effect. Once terminal evidence is durable, trailing run-operation cleanup is
best effort and cannot hide the authoritative result from a retry. That retry
also validates the retained execution profile and converges terminal-hook slots
that were still unclaimed when the original process disappeared. Fresh
`turn.completed` UUIDs now retain their runtime-generated provenance just like
deterministic replay IDs, so configured secret-fragment redaction cannot reject
an identity generated inside Cayu.

Terminal runtime hooks now reserve each app/agent registration slot with a
deterministic `hook.started` identity and a unique durable claimant before the
hook may apply governed effects. Concurrent exact failure replays therefore run
each hook at most once instead of duplicating task creation, dispatch, forks, or
custom events. A claimant can recover its own lost append acknowledgement, while
a peer stops at an unsettled earlier hook so app-before-agent and registration
ordering remain intact. Completed and failed hook markers bind the same claimant;
another contender may advance past only that positively settled slot. The
owning async stream exposes `hook.started` only after the hook returned and its
outcome marker is durable, so consumer abandonment cannot strand a reservation
before the hook was entered.

Final derived-fork validation now also rejects secret-bearing runtime taint
labels. Filtering runtime-owned keys out of the user-metadata projection no
longer lets unsafe policy authority bypass the pre-mutation fork boundary.

The server/dashboard contract advances from version 41 to version 42 to expose
the optional worker and handoff-generation fields on all seven control-plane
continuation bodies. Upgrade independently deployed servers and regenerate
clients before dispatching elected task continuations. Workerless direct
continuation remains available only for a task with no interrupted-handoff
generation. Terminal workerless replay requires a receipt-capable task store
because a store without atomic claimed-worker receipts cannot prove whether the
cleared terminal owner was direct or elected.

### Session forks are independently dispatchable ordinary sessions

The evaluator-owned `ForkGroup` subsystem, including its API, task namespace,
events, persistence, evaluator, replacement workflow, dashboard, examples, and
legacy readers, has been deleted. Applications now snapshot one exact safe
source, create caller-identified children with frozen first-invocation
and first-dispatch authority, submit each child through ordinary durable
dispatch, and consume or
interrupt each ordinary session independently. Runtime owns exact source and
profile admission, opaque source-incarnation fencing, bounded authority, retry
identity, and generic task/session recovery; application code owns grouping,
evaluation, replacement, and selection. Snapshot rejection detaches transcript
and checkpoint failures before they cross the public API boundary.

This is a deliberate prerelease breaking replacement with no ForkGroup
migration or compatibility adapter. The server contract advances from version
42 to version 43; independently deployed servers, generated clients, and
dashboards must be upgraded together. Existing saved ForkGroup records are not
read or migrated.

Default ordinary resume requests retain the revision-40 schema-2 queue wire
format and base task namespace for rolling-worker handoff. Requests using the
new tool-ceiling, interaction-grant, or profile-adoption controls use envelope
schema 3 and a reserved `.invocation-controls.v3` task namespace. Older workers
do not query that namespace and cannot claim authority they cannot validate;
current workers fairly consume both generations. Dispatches targeting an exact
fork child use relationship schema 2, envelope schema 4, and the independent
`.exact-fork.v4` namespace even when their invocation controls are otherwise
schema-2-compatible. The envelope binds the child's secret-independent
source-state commitment, so revision-40 workers cannot claim a child whose
relationship they cannot parse.
Each frozen first invocation requires an exact source and one caller-selected
`initial_dispatch_id` stored in the child relationship and `session.forked`
evidence. Alternate dispatch IDs, public resume, and inline dispatch cannot race
or duplicate that first invocation; exact concurrent submissions converge on
one queue task. Later settled-child interactions remain ordinary resumes.
Copied recoverable-environment state also retains its authenticated allocation
owner independently of immediate parent lineage. Multi-generation forks created
before intermediate execution therefore allocate for the running descendant
without adopting an ancestor resource, including after ancestor deletion.

### Parent models can discover independently completing children

Agents can opt into `ChildSessionContextContributor`, which recomputes bounded
direct-child lifecycle state immediately before each ordinary provider request.
It projects public identities and exact admitted, running, completed, failed, or
interrupted occurrences without embedding child output or persisting synthetic
transcript messages. Fresh terminal occurrences carry a parent-scoped reference
for `ChildSessionResultTool`; each occurrence is consumed atomically at the final
provider-start transition, after any provider-backed token count, and remains
replay-safe across retries, restarts, resumes, and recovery. One child can be
consumed while siblings keep running—there is no replacement group or cohort
abstraction.

Built-in in-memory, SQLite, and PostgreSQL session stores provide the same v1
query and consumption contract. Selection prioritizes fresh terminals, then
current children, then consumed terminals under explicit child, entry, byte,
reference, and work bounds. Dispatch revalidates the exact current terminal
event and parent/run authority, so stale owners and superseded occurrences fail
closed. Coverage reports typed truncation or unavailability. Transcript history,
current lifecycle state, terminal notification, bounded result retrieval, and
durable user steering remain separate contracts.

Breaking storage revision 79 installs the SQLite/PostgreSQL canonical
child-lifecycle candidate projection, its bounded page index, and database
maintenance triggers. Migration rebuilds only this derived index from existing
session, lifecycle-event, and notification-consumption truth. Reads select the
bounded candidate page first and reconstruct it with constant-count batched
queries; they do not rank or hydrate the complete parent branch. Stop pre-79
session writers before migration.

### Evaluated knowledge maintenance gains application-owned automatic authority

Applications may now pass an already-persisted and independently evaluated maintenance
proposal through `KnowledgeMaintenanceGovernor`. Reviewed mode records a durable route and
leaves the proposal pending without calling an automatic policy. Policy-automatic and
autonomous modes require an exact application-policy identity and may approve, reject, or
route the proposal. Approval and rejection reuse the existing atomic replacement,
supersession, relation, outbox, and receipt transaction; exact retries recover durable
attribution without rerunning policy code. This changes authority only: Cayu adds no
maintenance worker, scheduler, discovery algorithm, provider call, or model-facing tool.

Storage revision 77 adds bounded route-to-review attribution across in-memory, SQLite, and
PostgreSQL knowledge stores. Migration preserves existing reviewed proposals and decisions
but does not infer or backfill automatic authority, and there is no legacy runtime or
dual-write path. Stop pre-77 knowledge writers before migration.

### Dashboard source publication is crash-recoverable

`cayu dashboard eject` now publishes through a bounded destination-scoped transaction journal.
Exact retries recover an interrupted old-or-new tree and resume owned cleanup, while parent,
destination, staging, backup, content, link, reparse-point, and case-alias conflicts preserve every
ambiguous tree and fail closed. A bounded terminal receipt closes post-commit acknowledgement-loss
windows. Exact publication replay now reauthenticates the sealed descendant content: a changed
nonempty absent-or-empty destination fails closed, while an absent or proven-empty destination
retires the stale receipt and is published again. Read-only recovery still preserves later operator
edits. An authorized replacement must bind its exact predecessor receipt. Destructive cleanup
first claims the exact owned tree and durably records a bounded per-entry authority manifest before
deleting descendants. File hashing rejects an in-place size, mode, timestamp, or platform-attribute
change and carries that observation to the destructive boundary. Durable root and descendant authority
includes stable filesystem-incarnation evidence in addition to device, inode, and type, so inode reuse
cannot authenticate a recreated object;
Linux inode generation plus birth time, Windows volume-unique persistent object IDs, and Darwin
non-recycled object IDs or generations provide that evidence, while creation time alone is insufficient.
Windows filesystems without object-ID support and other unsupported filesystems fail closed rather than
treating recyclable file IDs as authority. Cleanup preserves replacement
conflicts observed at its destructive transitions, rejects every Windows reparse point regardless of
tag, and rejects a mounted root or descendant rather than traversing another filesystem; the documented
POSIX boundary does not claim conditional-by-inode pathname deletion or rename against hostile
same-owner name or content mutation between the final observation and kernel namespace operation.
Initial journal and cleanup-manifest publication use one destination-scoped pending file. A later
entrance cannot authenticate an initial
pending journal from that record alone, so it preserves complete and partial forms as one bounded
conflict rather than adopting them. A stage interrupted before its owner marker becomes durable is
likewise preserved rather than inferred to be Cayu-owned, and the private ownership marker no longer
reserves a filename inside the caller's published tree.
Normal stage population uses an identity-pinned bounded writer instead of reopening the private root
by pathname. Tree and cleanup-authority capacity are checked before population or replacement can
reach an irreversible transition, including a 128-level traversal bound. Population modes that would
make the staged tree unauthenticatable or unremovable are rejected before any file is written, and
replacement proves that the original tree is traversable and removable before creating its journal.
Win32 alias and reserved-name rules apply to Windows destinations. POSIX destinations retain
spellings only when native lookup semantics prove them distinct. Case-sensitive Darwin volumes
NFC-normalize canonical aliases without case-folding; an unrecognized filesystem uses a conservative
NFC/casefold alias domain so recovery cannot acquire a second owner. Malformed journal and
cleanup-manifest text is reported through the bounded publication-conflict classifications rather
than leaking codec or parser failures. Metadata-owner, pending-metadata, and destination-alias parent
censuses count all inspected entries, including unrelated names, and fail closed at a fixed bound.
Dashboard bundle validation, output bytes, modes, offline behavior, and the absent-or-empty
destination policy are unchanged.

### Lambda sidecar publication uses the guarded tree owner

`cayu lambda-microvm sidecar export` now shares the dashboard's bounded, crash-recoverable
whole-tree publication owner instead of importing dashboard-private cleanup helpers and carrying
a second backup state machine. Exact retries authenticate the complete emitted sidecar tree,
including raw manifest bytes. `--replace` atomically binds the current terminal
receipt's exact stable identity and terminal hash so a newer packaged sidecar—or another guarded
tree producer explicitly asked to replace that destination—can supersede it without weakening
recovery identity. Recoverable active work from an older request is settled before the successor
starts. A failed final rename restores the exact original directory synchronously when ownership remains
provable; ambiguous concurrent trees are preserved and fail closed. Sidecar bytes, manifest
semantics, modes, displayed content digest, offline operation, and successful CLI behavior are
unchanged.

### New-project scaffolds use the guarded tree owner

`cayu new` now publishes its complete project root through the same durable whole-tree owner.
Generated files and empty data directories retain their process-umask-derived modes; private
memory-key permissions, coding Git setup, and the destination root mode also retain their successful
behavior. If the process exits during final
publication, the same command recovers the exact old-or-new outcome instead of leaving generated
content without a retry owner. Recovery and the new absent-or-empty request share one locked
publication boundary, so removing a prior project and recreating an empty destination does not make
its historical receipt an unrelated failure. Pre-existing non-empty targets and concurrent
replacements still fail closed and are never adopted or cleaned up.

### Generator plans publish through a crash-recoverable transaction

`cayu generate slice`, `tool`, and `service-context` now apply multi-path project edits through a
bounded project-scoped transaction journal. New content is synchronized before durable commit
intent; exact originals are moved into synchronized private backups during the recorded commit
sequence before replacements cross their pinned, atomic no-replace boundaries. A subsequent
non-dry invocation recovers an interrupted transaction before
planning, while `--dry-run` remains write-free and reports that recovery is required. Exact retries
use one content- and identity-bound terminal receipt. Process loss therefore leaves a complete old
project, a complete new project, authenticated recoverable transaction state, or a bounded conflict
when it lands after private-directory allocation but before the external preparation owner is
durable. Recovery preserves that unauthenticated directory for operator inspection; it never adopts,
deletes, or treats it as transaction authority. Recovery also never adopts or overwrites a later
user edit, replaced parent, malformed record, link/reparse point, or
same-content object with different filesystem identity. Private population uses the identity-pinned
stage writer; created-directory modes participate in recovery fingerprints; and terminal cleanup
deletes only the exact bounded inventory sealed into its external cleanup claim. Existing generator
plan schema, generated source, verification commands, and successful CLI presentation are unchanged.

### Semantic watches retain evidence without self-authorizing effects

Applications can now explicitly evaluate one bounded observation through the existing
access-filtered `RecallEngine` and pass exact revision-bound match evidence to a versioned
application policy. `KnowledgeSemanticWatchEvaluator` durably records `ignore`, `emit`, or
`route_to_review` attribution and recovers exact retries without rerunning recall or policy.
The evidence retains bounded match reasons and rejects mismatched recall situations or
impossible duplicate lane ranks before policy execution. Similarity, ranking, recalled
text, and model or curator output cannot authorize the route. An emitted route does not
itself block a tool, send a notification, create a task, call a provider, inject context,
or perform an external effect.

Lexical-only profiles can explicitly require only the deterministic lexical lane; hybrid
profiles can require lexical and semantic coverage. Missing or truncated required evidence
can only route to review, while all optional-lane and omission diagnostics remain visible.
The framework adds no hidden scheduler, polling loop, watch worker, or standing instruction.

Storage revision 78 adds bounded semantic-watch receipts across in-memory, SQLite, and
PostgreSQL. Migration creates an empty table and does not evaluate historical observations,
infer outcomes, backfill records, dual-write, or retain a legacy read path. Stop pre-78
knowledge writers before migration.

### Evals ship focused onboarding and declarative project judge authority

Three package-shipped guides now lead operators through a first Control Plane
suite and baseline, rubric-based AI quality evaluation, and retained production
sessions, multi-stage scenarios, tool/process behavior, and memory evidence.
Generated projects continue to need no Evals-specific Python for deterministic
or captured evaluation. They may now declare one exact default judge in
`[tool.cayu.evals.default_judge]`, including provider, model, privacy,
same-model policy, and time/token/cost ceilings. Projects may deliberately
select the bundled dated public-rate book without Python; it also enables a
reviewed per-trial candidate interruption budget for author-first launches.
Cayu publishes judge authority through a separate tool-free application and
never infers authority, credentials, or pricing from ambient state.

Release acceptance now exercises case duplication, explicit launch subsets,
structured scenario input, and the complete installed-wheel browser/CLI/report
journey. The real-provider generated-project check additionally proves both
production-first and author-first evaluation, a labeled same-model rubric,
captured/fresh/judged artifacts, explicit candidate/judge work and cost
thresholds, and stable comparison exits. CI selection now fails open for all
package guide and Evals live-acceptance inputs.

### `cayu new` emits the complete Cayu application convention

Normal generated projects now begin with stable homes for configuration,
agents and prompts, tools and policies, environments, workflows, operations,
knowledge, memory, domain code, integrations, evals, observability, tests, and
ignored runtime data. `app.py` remains composition-only, while
`agents/registration.py` is the explicit generator and manifest-provenance seam.
Generated `AGENTS.md`, the minimal `CLAUDE.md` bridge, ownership docstrings, and
the source-controlled `[tool.cayu.scaffold]` plan give fresh coding agents the
architecture and exact proof commands before they write code.

`cayu new` now exposes canonical agent, service, and coding presets; SQLite and
Postgres database profiles; neutral and maintained provider profiles; optional
Docker coding execution; truthful capability discovery and selection; minimal,
interactive, dry-run, and structured JSON modes; and compatibility aliases for
the prior template/composition flags. Every maintained path stages and publishes
atomically. Declared convention projects receive read-only actionable layout and
registration drift findings from `cayu check`; custom projects remain freeform.
The coding preset populates the same canonical homes instead of concentrating
implementation in root `composition.py`. The service preset currently supports
SQLite only: `--preset service --database postgres` fails during planning, before
writes, until its application-owned product-operation store has a coherent
shared Postgres implementation.

### Durable runtime identity is bound to immutable build provenance

Sessions now retain a typed `RuntimeBuildProvenance` manifest and bind its exact
lowercase SHA-256 fingerprint into execution-profile schema 6. Wheel installs
derive structural identity from hashed `RECORD` entries; OCI and other immutable
deployments can provide a bounded explicit manifest; editable source installs
use an explicitly weaker deterministic content identity. Source revision remains
diagnostic and semantic `runtime_version` remains separate. Legacy sessions are
reported as provenance unavailable rather than being assigned the current
worker's build.

The identity propagates through session creation, resume, fork, lifecycle
receipts, prepared durable children, storage projections, server responses,
`cayu session show`, snapshots, and portable bundles. A worker with another
build fingerprint rejects queued or resumable work before provider dispatch.
The server contract advances from version 38 to version 39, the
session-inspection CLI schema to version 8, and prepared durable-subagent intent
to schema 3. Production
deployments can set `CAYU_REQUIRE_STRONG_RUNTIME_BUILD_PROVENANCE=1`; mixed-build
rollouts must drain or route existing exact-profile work to matching workers.

### AgentBundle ships as one deterministic `.cayu` file

The downloadable representation of one `AgentBundle` is now a regular `.cayu`
file with media type `application/vnd.cayu.agent-bundle`. Container schema v1
is a deterministic ZIP64 envelope with an explicit version record, exact
canonical `index.json`, stored entries, normalized metadata, and only the
transferred digest paths declared by the bundle. Its transport SHA-256 is
reported for download verification but does not replace `snapshot_root`,
`bundle_id`, or export authority.

Path and stream APIs pack, inspect, unpack, export, and import in bounded chunks.
Output streams must be empty, readable, seekable, and truncatable so a failed
write can roll back to empty and a completed container can be validated before
acknowledgement. Caller cancellation settles and resets any off-thread stream
writer before returning, keeping the same destination safe for an exact retry.
Once validated publication and durable protection release both commit, that
receipt wins a racing cancellation; release failure resets the stream instead.
The coordinator retains root protection until atomic file publication and uses
the existing verified directory/CAS importer for atomic root-and-pin
publication. Full containers are self-contained; thin containers visibly name
and enforce their exact destination inventory. Strict raw ZIP and logical
bundle validation reject compression, encryption, traversal, links, duplicate
or reordered names, malformed or contradictory ZIP64, overlapping/truncated/
trailing data, size or digest disagreement, and existing closure/secret
violations before a final file or snapshot root becomes visible.

Use governed `cayu agent bundle export` and `import` with explicit SQLite
snapshot-store, filesystem object-store, subject, binding, authority-scope, and
pin-owner inputs. They invoke the coordinator's protected export and atomic
root-and-pin import surfaces; `pack` and `unpack` remain separate representation
conversions. `inspect` and `examples/portable_agent_bundle.py` exercise the rest
of the one-file copy and materialization workflow. The unpacked directory remains
the canonical CAS/debugging representation; registry, Cloud, browser, and
desktop integrations can use the documented extension and MIME association
without being implemented in this release.

### Streamable HTTP can opt into the stateless MCP 2026 protocol era

`HttpMcpClient` now accepts the explicit
`protocol_era=McpProtocolEra.MODERN_2026_07_28` opt-in. That path uses
`server/discover` instead of the legacy initialize handshake, sends the required
per-request `_meta` and HTTP routing headers, validates modern result and cache
metadata, and never creates or deletes an MCP protocol session. Tool header
mirroring is derived only from admitted `x-mcp-header` annotations and publishes
atomically with private tool dispatch authority. The legacy HTTP path remains
the default and its behavior is unchanged.

This is a pinned modern HTTP core, not a claim of complete MCP 2026 support.
Automatic fallback, modern stdio, response caching, `subscriptions/listen`, and
MRTR / `input_required` remain separate work.

### Evals present memory structure, correct use, and causal evidence separately

Portable suites can now assert bounded ranges for memory items admitted to a
trial and provider exposures proven by its runtime-native attribution evidence.
Only complete, determinate evidence is scoreable; unavailable, truncated,
contradictory, changed, or indeterminate attribution remains unavailable rather
than becoming a false failure. The Control Plane can add this structural check
directly, add a reference-backed structured-judge template for semantic memory
use, and display both layers without implying that exposure proves correct use.

Fresh and captured result views now expose bounded memory sources, lifecycle
states, limitations, exact admitted-item and proven-provider-exposure counts,
and the matching published assertion detail. Operators can also upload an exact
stored `MemoryExperimentReportRequest`, validate it server-side against its
referenced results, inspect causal pair coverage and dispositions, and download
the validated report JSON or standalone HTML report. This action reports an
already defined repeated campaign; it does not invent candidates or execute
trials.

The portable corpus schema advances to version 4, the run schema to version 11,
and the published-result schema to version 10. The server/dashboard contract
advances from version 39 to version 40. Upgrade independently deployed servers,
generated clients, and dashboards together.

### Evals separate per-run concurrency from shared runtime capacity

Evaluation targets and durable run records no longer impose a universal
32- or 100-trial concurrency ceiling. The existing target and run
`max_concurrency` values remain finite per-run authority and dispatch controls,
with a portable representation maximum of 2,147,483,647 shared by public models,
SQLite, PostgreSQL, and browser clients.
Applications may now share an `EvalExecutionCapacity` across concurrent run
coordinators to bound aggregate active trials in one process. The shared
capacity defaults to 100, is explicitly operator-configurable without a
Runtime-defined upper ceiling, and releases permits across success, failure,
timeout, and cancellation. The 100 default is the N9 deployment target, not a
power-of-two or Runtime-wide limit. Storage revision 72 replaces the obsolete
32 ceiling with the portable representation maximum while retaining the
positive-integer invariant. Migrate shared stores and upgrade workers together;
do not mix pre-72 writers with the widened contract.

The server/dashboard contract advances from version 37 to version 38 for the
widened Evals concurrency schemas. Upgrade independently deployed servers,
generated clients, and dashboards together; a v37 client must not guess the new
authority from a v38 server.

### Complete agents can be exported, copied, imported, and freshly materialized

`AgentBundle` now transports one exact `AgentSnapshotRef` plus its complete
authorized provider-object closure as an ordinary filesystem directory. Full
exports contain every restorable object; thin exports consume an explicit
destination inventory and transfer only missing content-addressed objects.
Streaming component storage and transfer allow agents larger than process
memory, while reports separate root, logical closure, shared, incremental, and
materialized byte counts plus unresolved external bindings.

Import bounds and verifies every index, node, component manifest, and blob before
atomically publishing and pinning the root. Corrupt, truncated, missing, extra,
wrong-root/scope, traversal, symlink, oversized, and unsupported-schema inputs
fail closed. `reusable_agent`, `continuing_agent`, and `evaluation_candidate`
profiles make session disposition explicit. Fresh materialization rebinds
credentials, evaluator, budget, leases, scratch, runtime/session/operation
identities, and starts with no catalogue-discovery grants. Terminal capture
requires a safe closed frontier and emits a durable descendant-snapshot receipt.
Known registered secret bytes and structurally private/evaluator payload fields
are refused on export and again before imported roots are published; secret
values never enter the portable bundle. Filesystem CAS shards are pinned with
no-follow descriptors, snapshot access is checked before fresh authority is
allocated, and restart recovery verifies the exact materialized file closure.

Run `uv run python examples/portable_agent_bundle.py --help` for the public
three-process export/import/materialization example. This is a prerelease schema
version 1 contract and adds no remote registry or encrypted archive format.

### Explicit lineage-scoped parent-to-child artifact handoff

Applications can now register sealed `publish_workspace_artifact` and
`materialize_shared_artifact` tools so a parent can preserve one generated file,
pass its opaque reference explicitly, and let an authorized fork or subagent
reconstruct it in an isolated workspace—even after a process restart. Runtime
validates the exact source and caller session instances, bounded fork/subagent
ancestry, root invocation and origin, causal budget, active grant, policy
fingerprint, stable artifact-store identity, metadata, digest, size, destination
policy, and overwrite precondition. Reference possession alone is not authority;
ordinary artifact read/list scope remains unchanged.

Publication and materialization use deterministic identities, atomic durable
preparations and receipts, cancellation-safe settlement, lost-ack recovery, and
bounded retry/concurrency convergence. Revocation and expiry fail closed.
Recovery receives only a narrow exact-byte artifact reader, not the raw store or
its write/list/delete surface. Files containing a secret registered in the
publishing or materializing invocation are refused before cross-session copy,
and every protocol record must survive exact Runtime secret-scope sealing before
storage. Paths,
content types, bytes, publication count, lineage depth, retention class, and
overwrite behavior are sealed by `SharedArtifactPolicy`. The feature does not
automatically promote generated files into `AgentSnapshot`; applications still
own the scratch/evidence/anatomy disposition decision. No storage migration is
required because the protocol uses existing bounded session-operation records.

### Evals publish explicit repeated-trial reliability

Authored suites now freeze trial count, required passes, and maximum concurrency
as one content-addressed policy. Results retain every trial and classify passes,
candidate failures, runtime errors, evaluator errors, unavailable evidence, and
cancellations separately. Comparisons, JSON/HTML reports, the Control Plane, and
CLI regression exits now treat a worse distribution as a reliability regression,
even when the aggregate status still passes or score tolerance hides the change.

Preflight exposes exact candidate-trial, model-step, and judge-call counts but no
longer mislabels post-observation candidate or judge token and cost stop
thresholds as hard maxima. SQLite and PostgreSQL workers checkpoint each
terminal trial behind the live claim, resume only missing slots after a restart,
and partition independently recoverable runs from one authored-suite launch into
durable concurrency lanes, so its accepted ceiling holds across coordinators
without serializing all independent work. The exact accepted-exposure revision
continues to fence preview, admission, and worker execution; cross-release result
comparison uses a separate content-addressed contract that excludes only release
and presentation metadata while preserving execution, isolation, evidence, work,
and pricing drift. Storage revision 72 is therefore a breaking mixed-worker
boundary: migrate all EvalStore databases and deploy only revision-72-aware eval
workers. Comparison documents
advance to schema version 4; the existing server/dashboard contract version 38
includes these APIs, so upgrade generated clients and dashboards with the
server.

### Evals can verify workspace and artifact outputs without Python assertions

Portable suites and Control Plane now author workspace-file presence/absence,
byte-range, and whole-file SHA-256 expectations plus session/environment
artifact filename, content-type, byte-range, digest, count, and optional public
text expectations. Fresh execution, captured-session drafts, stores, SDK/CLI,
JSON/HTML reports, result presentation, baselines, and comparison retain one
exact assertion and evidence-policy identity.

Workspace bytes never enter portable evidence. Cayu reads only declared paths
and retains structural facts; incomplete reads never publish prefix hashes as
whole-object identity. Artifact listing is owner-filtered, unrelated metadata
is discarded, and only structurally prefiltered candidates are read under fixed
item and byte ceilings. Artifact text is disabled by default and requires a
trusted target profile with `include_artifact_text=True`; supported UTF-8 text
crosses the application redactor before bounded retention. Missing, unavailable,
unsupported, redacted, malformed, truncated, and limit-exceeded observations
remain distinct.

Assertion evidence advances to schema version 5, trajectories to version 5,
eval runs to version 9, published results to version 8, and the
server/dashboard contract from version 36 to 37. Upgrade servers, generated
clients, and dashboards together. No storage migration is required.

### Explicit live MCP catalogue refresh

- `CayuApp.register_agent(..., mcp_toolsets=(toolset,))` now declares that the
  application tracks the complete admitted tool list for that MCP source.
  Existing `tools=toolset.tools` registration remains a static snapshot and
  starts no refresh work.
- `await app.refresh_mcp_toolset(toolset)` performs one bounded full
  `tools/list`, constructs detached immutable adapters, applies the configured
  `McpManifestPolicy`, and publishes every affected agent catalogue through one
  copy-on-write generation. A policy-accepted unchanged manifest retains the
  current generation.
- Refresh fences undispatched calls before discovery starts. Accepted
  publication makes every older adapter snapshot stale; failed or blocked
  refresh quarantines the source until another complete verified refresh
  succeeds. Built-in transports signal only after they own a possibly
  dispatched request; extension sessions without that proof retain the fence
  until their call settles. Their discovery path hashes unredacted contracts
  before redaction and stages the exact public-to-transport name map; only an
  accepted or verified-unchanged refresh commits that map while the generation
  fence is held. Blocked candidates cannot rebind dispatch, and a committed
  complete catalogue replaces stale names instead of merging them forward.
  Already-dispatched calls are not rebound.
- Pre-refresh MCP calls and targeted/discovery references now also check the
  live source generation before generic governed work. Gateway, OpenAI client
  and hosted search, and additional-tools projection share the same canonical
  catalogue-drift rejection; they cannot newly consume a stale grant or reach policy,
  approval, secrets, environments, hooks, or target execution. Existing
  execution-profile recovery and durable session/fork ceilings provide the
  continuation boundary without an MCP-specific durable subsystem.
- Refreshable sources require an explicit stable
  `McpServerSpec.connection_id` that is unique within the application, cannot
  also be registered through static adapters sharing that live source, and
  cannot be refresh-owned by multiple `CayuApp` instances. Server additions
  enter only new application catalogue generations; existing durable session
  ceilings do not widen. Distinct `McpToolset` wrappers around one live session
  share the same source owner and generation fence. Independent sources can
  fetch candidates concurrently while final application publication remains
  serialized and copy-on-write.
- A refresh-owned source whose initialize result declares exact
  `tools.listChanged: true` now consumes
  `notifications/tools/list_changed`. Receipt synchronously marks the source
  dirty and fences old dispatch authority; a short source-owned coalescing
  window turns a burst into one call through the same bounded, policy-governed,
  copy-on-write refresh path as `refresh_mcp_toolset()`.
- Stdio consumes the notification on its existing reader. A signal observed
  after discovery but before application ownership is retained as one
  payload-free freshness marker; ownership installation fences dispatch
  synchronously and reconciles it after the registration transaction completes,
  so catalogue changes cannot disappear in that gap. Streamable HTTP consumes
  notifications interleaved in POST/SSE responses and owns one
  bounded GET/SSE server-message listener when the endpoint supports it.
  Activation fences dispatch synchronously after registration, while a narrow
  registration-only allowance lets consecutive agents share the source before
  the event loop starts. The first valid stream reconciles the catalogue before
  restoring dispatch, closing the gap between initial `tools/list` and listener
  establishment. Later continuity loss applies the same fence. Reconnect
  establishes a new stream, replays a bounded safe SSE cursor through
  `Last-Event-ID` when available, and reconciles once before restoring dispatch.
  A validated stream resets reconnect backoff, so routine server-side stream
  rotation cannot ratchet later continuity fences to the maximum delay.
  HTTP 405 performs that activation/final reconciliation before selecting the
  manual-only fallback. Notifications are never replaced with polling. Listener
  connection and body reads remain idle-bounded and individual SSE events remain
  size-bounded, but a healthy established stream has no finite RPC lifetime or
  cumulative-response ceiling that would force periodic catalogue refresh.
  Stream cleanup retains exact settlement ownership across timeout and
  cancellation. Protocol violations fence the HTTP session with bounded secret-safe
  diagnostics instead of entering a silent reconnect loop.
- A newer signal supersedes any candidate still in discovery or publication.
  A same-signal failure quarantines the source without a retry loop; a later
  signal or explicit successful refresh can recover it. Toolset close cancels
  and joins notification-owned work. MCP 2026-07-28 subscription and cache-hint
  capabilities remain deferred.

### Prepared durable children retain exact runtime identity

Durable-subagent submission intent schema 2 now binds the child runtime name
and version alongside its execution profile and uses that exact identity for
PENDING child creation, admission, and recovery. This closes a packaged-runtime
failure where the frozen profile named a versioned Cayu runtime but the prepared
child defaulted to an unavailable runtime version, causing strict invocation
lifecycle receipt validation to reject queue admission before provider
dispatch. Prepared-child queue tasks move to the
`.prepared-subagent.v2` namespace so older workers cannot claim the expanded
authority record. This is a prerelease protocol boundary: stop v1 workers and
cancel/recreate any remaining unclaimed v1 prepared-child tasks before rollout.

### AgentSnapshot is a Merkle-rooted, lifecycle-managed manifest

`AgentSnapshot` schema version 3 now exposes a content-derived
`snapshot_root` over a recursively verifiable typed Merkle closure. Logical
agent registration and authority scope move to a separate immutable
`AgentSnapshotIdentityBinding`, so identical state can retain one root across
registrations without treating a digest as permission. In-memory and SQLite
stores provide authorized closure reads and inspection, exact shared/unique
size accounting, idempotent pins and releases, lifecycle protections, and
bounded reachability-based garbage collection. Collection requires authorized
access to every logical binding it would remove and atomically rechecks that
binding set. Materialization and recovery require the same authorized access;
a root digest alone cannot start or recover candidate effects. Existing
materialization now protects its root before provider effects and releases that
protection only after verified finalization.

- Root checkpoints advance to schema version 6. Existing user-input pauses that
  predate exact pause authority are retained as bounded ambiguous tombstones:
  they cannot be resumed, recovery reports them explicitly, and an operator can
  retire them with the normal session-interruption API without exposing the old
  prompt or tool arguments.

### External interruption atomically supersedes pending user input

User-input pauses now use exact durable open and close publication receipts
bound to the session incarnation, source interaction and run epoch, tool-round
identity, execution profile, pause content, answer request, and complete
resolution request. Before reconnecting an environment, publishing recovery
evidence, running hooks, or dispatching sibling tools, an answer or manual-
recovery claim atomically advances from `claimed` to `executing`. An operator
interruption can supersede an active pause or a pre-execution claim and retains
that exact authority in the interruption evidence. Once execution admission
wins, interruption fails closed instead of falsely reporting supersession, and
the exact execution owner remains fenced until it reaches a definite outcome.
The losing pre-execution claim cannot dispatch continuation work, clear a later
pause, or leave stale `pending_user_input` state behind.

Identical retries after acknowledgement loss replay the original publication;
conflicting answers fail without mutation. Incomplete-session recovery now
distinguishes active, answering, answered, superseded, and ambiguous pause
evidence and fails closed when the required receipt or authority tuple is
missing. Memory, SQLite, and PostgreSQL stores enforce the same atomic
publication contract.

### Classified transient provider failures retry by default

`RetryPolicy()` now permits five total attempts instead of disabling retries.
The default continues to reject permanent failures immediately and limits typed
but otherwise unknown provider failures to two total attempts. Exponential
backoff now includes up to 0.5 seconds of jitter, and the bounded cap applied to
computed delays and provider `Retry-After` instructions increases from 10 to 30
seconds. An explicit `initial_delay_s=0.0` remains a no-wait override, while
`max_attempts=1` remains the explicit opt-out from retries.

The effective policy remains part of each admitted invocation's durable
finalization identity. Active sessions and recovered work therefore retain
their admitted authority instead of silently adopting the new default.

### Evals can assert lifecycle, approval, and child process behavior

Portable suites and Control Plane now support required, forbidden, and ranged
counts over a closed payload-free process-event vocabulary. An advanced exact
order assertion filters the root trace to the selected fact kinds and checks
the complete filtered sequence and multiplicity. Child-status assertions now
also support interrupted direct children. This covers useful session, tool,
approval, structured-output, and budget-limit behavior without admitting raw
event names, custom events, payload predicates, approval identity, or
executable browser input.

Process evidence is bounded to 4,096 typed facts. Missing root evidence and
bounded prefixes remain unavailable rather than becoming candidate failures or
passes, and incomplete child capture remains unavailable. Safe typed result
details are shared by suite authoring, captured-session promotion, execution,
stores, Control Plane, CLI, reports, and comparison. Published eval results
advance to schema version 7, assertion evidence to schema version 4, and the
server/dashboard contract from version 35 to 36. Upgrade servers, generated
clients, and dashboards together. No storage migration is required.

### Evals can assert bounded public tool arguments and retained results

Control Plane and the portable SDK/HTTP suite contracts now support exact
tool-occurrence assertions over recursive JSON subsets. Finalized public tool
arguments are available under the standard evidence policy; public-safe tool
results remain an explicit trusted-target opt-in. Both paths apply workload
secret redaction and independent size, depth, node, and call-count limits before
evidence reaches storage, APIs, reports, or the browser.

Missing calls and value mismatches remain failures, while unsupported capture,
unavailable data, malformed retention, incompatible identity, overflow,
truncation, and selected-path redaction remain distinct unavailable states.
Fresh, scenario, and captured-session evaluation use the same matcher. JSON and
HTML reports and schema-version-3 comparisons preserve safe actual values,
evidence-state changes, and observed-value changes without raw payload fallback.
Published eval results advance to schema version 6, assertion evidence to schema
version 3, and the server/dashboard contract from version 34 to 35. Upgrade
servers, generated clients, and dashboards together. No storage migration is
required.

### Evals results are explainable across the Control Plane, SDK, CLI, and reports

Captured and fresh immutable results now expose one bounded
`EvalResultPresentationV1` projection. It keeps candidate outcome,
deterministic assertions, semantic quality, evaluator health, runtime, and
evidence completeness separate, and presents structured rubric criteria,
explanations, exact contributions, aggregate/threshold, safe judge/reference
identity, observed usage, and priced or explicitly unpriced cost without
publishing private truth or raw judge prompts/output. Exact-identity comparison
adds per-criterion and aggregate deltas and refuses heuristic pairing. The
Control Plane, protected API, versioned JSON/HTML reports, and CLI comparison
use the same semantics. JSON report downloads now contain both immutable source
and its bound presentation and remain accepted CLI report/compare inputs. The
server/dashboard contract advances from version 33 to 34; deploy those clients
together. No storage migration is required.

### OpenAI Responses can defer catalogue schemas through hosted Tool Search

Agents can select `tool_discovery_mode="openai_tool_search_hosted"` or the
portable fallback mode
`"openai_tool_search_hosted_or_search_tools"`. Exact model ids must be listed
separately in `OpenAIProvider.hosted_tool_search_models`; Cayu performs no
model-family inference and establishes this projection only for the official
OpenAI Responses endpoint.

Cayu sends bounded ceiling-authorized catalogue functions as
`defer_loading=true` candidates followed by the server-executed Tool Search
definition, forces serial tool calls, and validates the exact adjacent hosted
search call/output pair. Loaded name, description, and schema evidence must
match the dispatched candidate projection and current catalogue authority
before Cayu atomically publishes branch-local grants with the assistant tool
round. The resulting direct function calls retain the ordinary policy,
approval, hook, effect, secret, environment, execution, result, and recovery
contracts. Missing, altered, unrelated, oversized, duplicated, or out-of-order
provider evidence fails closed before target work.

If direct or targeted exposure covers the complete session ceiling, the hosted
projection is an explicit zero-candidate no-op and Cayu omits the inert server
search tool from the OpenAI wire request.

Streaming, terminal-only streaming, non-streamed responses, inline replay,
server-state chaining, and background process-loss recovery preserve the same
authority. Background stages persist a digest of the original bounded candidate
projection plus bounded, name-free hashes for exact native-targeted exclusions
and replay-loaded grants, then reconstruct it from frozen session authority
before publishing recovered grants. A credential-free two-request example
exercises the real adapter and runtime without asserting production-model
support or a provider cache hit.

Server-chain ownership now includes the branch discovery generation as well as
the exact candidate projection. Forks therefore neutrally rebuild inherited
history instead of retaining a parent response's loaded server surface, while
the generation identity stays off the provider wire. Provider-native discovery
request footprints advance to schema version 7 and expose only protocol,
candidate/loaded counts, and hosted generation identity; candidate names,
descriptions, schemas, searches, and arguments remain private.

### Agent work context and recall progress are durable, revisioned facts

Applications can publish a bounded `AgentWorkContext` under a stable task ID and
retain immutable exact revisions across restarts. Semantic no-change publication
does not fabricate a revision, while exact operation receipts and
compare-and-swap current pointers make retry and concurrent writers
deterministic. A narrow `AgentWorkContextStore` has copy-safe in-memory, SQLite,
and PostgreSQL implementations without folding work-context ownership into
`TaskStore` or `KnowledgeStore`.

`AgentRecallCheckpoint` independently records the captured knowledge-change
and semantic-index-readiness high-water marks plus the frontiers processed by
one exact agent/task/namespace/access view. Advances are monotonic,
compare-and-swap fenced, and cannot move beyond either captured source
frontier or lower a previously captured high-water mark. Only the task's current
work-context revision and hash may become the new checkpoint basis. A
work-context change requires a full-index processing basis, preventing an old
delta cursor from making older knowledge permanently ineligible. CAS revisions
and processed sequences—not caller wall-clock timestamps—order progress, so
clock skew cannot strand a valid checkpoint.
Checkpoints do not claim provider exposure, notification consumption,
relevance, or task completion; they record only bounded freshness processing.

Checkpoint-aware recall now turns those facts into bounded work without taking
over checkpoint persistence. `AgentRecallProcessor` selects full-index recall
for a missing/changed work context, exact-revision delta recall for newer
knowledge/readiness frontiers, or an explicit no-work result. It returns an
immutable checkpoint proposal; callers retain compare-and-swap advancement and
staging authority. Requests use one exact namespace scope, and results retain
their work-context identity, captured/processed frontiers, and source-event
provenance even when no proposal can safely be made. Transient semantic failure
cannot commit a new full-index work-context basis or consume a final delta retry.
A partial delta advances lexical progress during a semantic failure only when
retained READY events can reconstruct every eligible revision on the next
attempt; otherwise it withholds the complete proposal instead of checkpointing
past failed work.
Full-index candidates, semantic readiness, and attached lineage are constrained
inside the captured store frontier, so a concurrent commit remains for the next
delta instead of leaking into a result whose checkpoint predates it. Lineage
endpoint revision, status, and currentness come from that frontier while live
authorization still applies. Embedding attempts are retained independently, so
a captured readiness frontier resolves both the readiness event and the newest
accepted vector at or before that event rather than aliasing a later refresh.
PostgreSQL frontier-filtered semantic search keeps the HNSW fast path and falls
back to an exact scan only when bounded identity or post-filter completeness
requires it. Operational freshness-read failures use the typed processing error
boundary while preserving their backend cause and cancellation behavior.
Delta ranking is restricted inside every built-in knowledge backend before
top-k selection, including PostgreSQL/pgvector, so unchanged global winners
cannot hide a relevant changed revision. Exact-revision delta search carries
the same captured knowledge/readiness pair through ranking and lineage:
readiness or relations published later stay outside replay, and a same-ID
delete/recreate generation cannot alias the captured revision. The processor
does not inject context,
wake an agent, record exposure, consume notifications, or alter the knowledge
tools.

Additive storage revision 69 installs only the new empty authoritative tables.
It performs no task, session, transcript, or knowledge backfill and adds no
legacy compatibility path. The hermetic performance baseline covers zero-record
construction, indexed current reads, revision appends, checkpoint advances, and
incremental SQLite storage.

### Checkpoint-aware recall can be staged without a crash window

`AgentRecallDelivery` now freezes one exact checkpoint-aware processing result,
its processing and checkpoint fingerprints, full-index or delta classification,
work-context/access authority, and staging attribution. The work-context store
commits that immutable payload and the proposed checkpoint compare-and-swap in
one operation, so cancellation or a failed state write cannot leave either half
visible. Exact retries converge and conflicting delivery, operation, or
checkpoint identities fail explicitly.

In-memory, SQLite, and PostgreSQL stores also expose the same scoped
oldest-pending claim lifecycle with bounded leases, renewal, explicit retry
release, expiry takeover, stale-worker fencing, and typed downstream
acknowledgement. Acknowledgement means durable handoff acceptance only; it does
not synthesize `RecallReceipt`, `ContextExposure`, provider visibility,
notification consumption, or task completion. The store clock owns lease time:
future-dated staging, release, and acknowledgement attribution is rejected so a
caller cannot extend a lease or pin the oldest pending stage. Storage revision
71 installs empty authoritative delivery tables as a clean prerelease break,
with no inferred records, backfill, legacy reads, or dual writes. A
provider-free 50-stage benchmark fixes p50/p95 ceilings for atomic
stage/checkpoint, claim, acknowledgement, and indexed no-pending paths.

The rebuildable PostgreSQL `cayu_knowledge_embeddings` table now keys rows by
identity and accepted readiness sequence so multiple projection attempts remain
available to captured-frontier replay. This is an intentional derived-index
schema break, not an authoritative-data migration: drop the old embeddings
table before startup and rebuild projections from canonical entries. No
authoritative-data backfill or legacy read path is provided.
Projection writes serialize against readiness publication and activate a new
attempt only after its historical row is durable inside the same transaction.
A partial unique index prevents concurrent writers from leaving more than one
attempt current for an identity.

### Evals add structured, bounded model-judge contracts

Portable corpus schema V2 can pin a server-published judge profile and evaluate
one to eight stable weighted rubric criteria against optional evaluator-only
public or private reference truth. Cayu strictly decodes each criterion,
computes the weighted score and threshold itself, and publishes only typed,
redacted criterion evidence under published-result schema V5. Private reference
content, raw judge prompts/output, credentials, and provider options never enter
the corpus or result. Judge calls are tool-free, use an isolated process-local
session store, and are bounded by explicit timeout, token, and optional
priced-cost ceilings; profile/reference/privacy drift, public-reference secret
conflicts, unrepresentable threshold boundaries, and judge failures produce no
candidate score. Successful judgments retain observed judge token usage and an
exact priced cost or explicit unavailable cost state. Server contract version 31
introduced the safe `judge_profiles` target catalog while suite-authoring V1
remains strict and compatible. Version 33 adds suite-authoring V2 plus protected
Control Plane rubric-authoring and fixed-evidence calibration routes; independently
deployed dashboards, servers, and generated clients must be upgraded together.

### Repeated memory experiments publish exact paired reports

`MemoryExperimentReport` now turns fixed memory-intervention executions and
their published Evals evidence into one complete baseline/candidate repetition
matrix. It retains failed, timed-out, cancelled, unavailable, unmatched,
indeterminate, and missing rows; binds case, snapshot, execution-profile,
provider/model, evaluator, and attribution identities; and computes quality
deltas only for comparable pairs. Latency, total-token, memory-preparation, and
memory-context observations receive the same paired, case, and experiment
availability-aware distributions without entering ranking. Canonical cost,
usage, retry/repair, and pricing classifications are embedded from the existing
paired cost-quality contract. Declared safety/privacy and evidence gates run
before deterministic fixed-candidate ranking, so unavailable evidence cannot
become a zero or a survivor-filtered recommendation. Typed dispositions retain
separate incomparable/unavailable counts and distinguish superseded baselines
from eligible candidates that were not selected.

The SDK, `cayu eval memory-report`, deterministic JSON/HTML, and protected
Control Plane report routes expose the same schema. Control Plane construction
authenticates each supplied published graph and its associated execution-profile
snapshot against its exact `EvalStore` run; result-less rows retain declared
experiment-contract profile authority without claiming stored-run provenance.
Published eval runs advance to schema 5 and corpus execution results to schema 3
to retain the canonical source-trial revision. The additive report routes
advance the server contract to version 32; regenerate and deploy
server clients together.

### Evals launches are pinned to server-published execution profiles

The Evals target catalog now resolves a public, secret-free snapshot of the
current candidate, runtime execution identity, fixture/reset/effect posture,
evidence policy, complete resource ceilings, and an opaque commitment to the
request base, bootstrap messages, and compilation limits. Public-safe target
material uses a structural digest; private material uses a process-keyed HMAC
so raw values never cross the API and old work fails closed after restart.
Generated projects receive a safe one-trial, concurrency-one profile without
Evals-specific application code. Every control-plane launch carries the exact
profile revision, and durable admission plus worker restart validation rejects
runtime, target-input, and limit drift before provider dispatch. Explicit
applications may publish greater trial or concurrency limits only with an
application-managed reset contract and stable isolation revision.

### OpenAI Responses can load catalogue tools through client Tool Search

Agents can select `tool_discovery_mode="openai_tool_search_client"` or the
portable fallback mode `"openai_tool_search_client_or_search_tools"`. One exact
`OpenAIProvider.client_tool_search_models` allow-list establishes support;
Cayu performs no model-family inference, rejects compatible endpoints for this
native projection, and freezes the projection before dispatch. The provider
adapter maps Cayu's stable search definition to a client
`tool_search`, validates streamed and non-streamed search calls, runs the same
durable local catalogue search, and returns branch-authorized registered
functions in `tool_search_output`. Loaded functions then use the ordinary
policy, approval, hook, effect, secret, environment, execution, and recovery
path. Native-only discovery does not expose the portable `call_tool` gateway.

Tool Search call/output state survives inline replay, server chaining,
background reconnect, and neutral stale-chain recovery. Completed search calls
are emitted once across lifecycle-plus-terminal evidence. Historical outputs
are filtered against the current branch view, and server response references
carry loaded-tool ownership, so a fork cannot inherit parent addressability
through provider state. A loaded name also unloads if context trimming removes
the search output that supplied its schema. Exact model allow-lists,
configured/resolved delivery, and the loaded-definition projection are bound
into execution-profile or keyed request identity at their respective authority
boundaries.

### Invocations use one frozen authority context and typed store commands

Invocation admission, continuation, settlement, and cleanup now carry one
frozen in-process context containing the exact validated execution profile and
the live registered collaborators selected for that invocation. Restarted work
must reconstruct and validate that context before provider, tool, hook,
environment, verifier, or effect execution. Durable lifecycle mutations use a
closed versioned command family with exact compare-and-swap authority and
per-command replay receipts across the in-memory, SQLite, and PostgreSQL
stores. Custom session stores must explicitly implement and opt into command
version 1; inherited or legacy behavior fails closed before dispatch.
Invocation release requires runtime-owned cleanup authority and one exact
durable terminal proof: the interaction settlement, a same-invocation terminal
session event when no matching interaction settlement exists, or a completed
recovery owner's exact claim. Public recovery keeps an open interaction fenced
until its terminal transition is durably published; an older paused receipt
cannot release the replacement epoch.
The private command ledger is capped at 128 receipts and 8 MiB of canonical
JSON. It rolls off the oldest superseded run-epoch receipt groups before admitting
new authority, while retaining the command being committed and the current
active release reservation. Each active create, admission, or rebind reserves
one item and the worst-case encoded capacity for its mandatory release receipt,
so an individual command that still cannot fit fails before mutation without
permanently exhausting a long-lived session. Exact replay remains available
within the retained window; an older replay fails closed against independently
advanced session state. Receipt replay also authenticates the embedded result
against independently loaded session and execution-profile authority.

Workspace-observation transitions now atomically stamp the current root
checkpoint schema even when a raw built-in store begins without a checkpoint.
Generic checkpoint replacements preserve that observation root, so an ordinary
state write cannot reinterpret or delete active recovery authority.

The root checkpoint schema advances from version 4 to version 5 and reserves
`invocation_lifecycle_receipt` as private runtime authority. Upgrading a
supported older checkpoint preserves unrelated fields but deliberately drops
any older caller-authored value colliding with that newly reserved name. Source
schemas 3 and 4 could have carried active-invocation authority, while historical
generic checkpoint writers could replace or delete that root. Migration removes
any profile and never interprets its absence as proof that the session was
ungoverned. It retains only an authenticated, empty receipt-history tombstone:
the tombstone proves that authority history may have existed but cannot authorize
admission, recovery, settlement, cleanup, or replay. Such sessions remain fenced
and must start a new session rather than resume in place. Generic checkpoint and
metadata publication cannot observe or mutate the receipt ledger. This is a
prerelease authority migration; applications must not depend on private root
names.

Custom session stores opting into lifecycle command version 1 must also own an
atomic `transform_checkpoint(...)` boundary. The runtime adapter now filters
private lifecycle roots before every generic checkpoint/operation callback and
reattaches authenticated authority afterward, independent of whether the raw
store itself performs that filtering.

### Local attempts retain complete-tree cleanup ownership across worker loss

`LocalExecutionAttemptCoordinator` adds a task-backed Linux containment boundary
for trusted local agent-attempt trees. It freezes an immutable attempt/effect
identity and exact task-claim generation before launch, supervises descendants
across timeout, cancellation, and
abrupt parent death, publishes exact process and quiescence receipts, and blocks
task or retry-series replacement until positive terminal settlement. Memory,
SQLite, and PostgreSQL task stores implement the same durable fence. Capability
evidence distinguishes graceful cleanup, hard deadlines, parent-death
containment, and intentionally persistent detached work; process quiescence never
proves the outcome of non-idempotent external effects.

Storage revision 66 is a clean prerelease compatibility break for task workers.
It installs the local execution-attempt authority and retry fence. Drain task
workers and migrate every shared task database before starting revision-66
workers; pre-66 and revision-66 task workers must not share a database because
older claimers do not consult the attempt fence.

### Generated projects can opt in to an explicit coding composition

`cayu new NAME --composition coding` now generates one runnable, ordinary-Python
composition for repository work. It assembles existing bounded file/search/Git
tools, local artifacts, durable reviewed knowledge, background reviewer
delegation with result recovery, and human-input pause/resume. The generated
manifest and control plane expose the ordinary agents, tools, environment, and
stores; no new agent kind, registry, implicit permission, or post-start mutation
is introduced.

Starter selection only chooses implementations. The registered exposure policy
separately governs model visibility, while tool policy, approval policy, and the
ordinary runtime gates remain the independent call-authorization boundary.

Generation requires `git`, `rg`, and the POSIX descriptor-relative filesystem
primitives used by secure `LocalWorkspace` path operations, initializes a clean
Git baseline, and emits a credential-free smoke that exercises the whole
composition. Unsupported hosts fail during generation or application
construction. The selected workspace defaults to the project root and may be
overridden with `CAYU_WORKSPACE_ROOT`, subject to explicit existing-Git-root and
non-filesystem-root checks. The trusted-host local runner does not inherit
arbitrary ambient variables, but still forwards Cayu's minimal operational
allow-list, and is documented as non-sandboxed. The default and service scaffolds
remain unchanged, and service plus coding composition is rejected.

### Trusted host tools can opt into a hard Linux process deadline

`ProcessIsolatedTool` adds an explicit reconstructable JSON-only adapter for
trusted synchronous or native dependencies that may not cooperate with
`asyncio` cancellation or may hold the Python interpreter lock. The parent
owns a versioned bounded protocol, a wall deadline, a Linux child-subreaper
supervisor, TERM-to-KILL cleanup of its complete adopted descendant tree,
result validation, and existing
effect uncertainty. Ordinary tools remain in process, and
`tool_timeout_seconds` remains a cooperative cancellation request for them.
The supervisor admits its worker only after the parent owns the exact process
handle and settlement waiter. Durable preparation is distinct from a
content-bound `worker_not_admitted` settlement when setup, spawn, caller
cancellation, the pre-admission deadline, or the final cleanup fence positively
settles before admission. Recovery prefers that positive zero-dispatch evidence over
the conservative preparation marker. The post-reaping acknowledgement proves
tree cleanup and carries supervisor health separately from the worker status.
A `supervisor_failed` outcome remains a bounded failure, cannot publish an
otherwise valid tool result, and retains independent terminal or diagnostic
failures. Manual recovery follows the same exact zero-dispatch evidence as
automatic recovery, and durable preparation preserves caller task-cancellation
bookkeeping. Descendant reaping retains constant-size worker-leader status rather
than an invocation-long PID history.

Application manifest schema 13 and tool descriptor/capability schema 2 expose
the configured execution boundary, timeout strength, isolated adapter identity
and configuration digest, deadline, and the explicit `sandboxed=false` claim.
Configuration, environment values, arguments, context, and secrets are not
published. The initial hard boundary requires Linux child-subreaper and `/proc`
child-enumeration support and
does not claim hostile-code, filesystem, network, credential, privilege,
kernel, or abrupt-parent-death containment. See
[Process-isolated host tools](process-isolated-tools.md).

### Agents can discover large registered tool catalogues through a stable core

Agents may opt into provider-neutral discovery with
`tool_discovery_mode="search_tools"`. Cayu then sends one stable two-definition
prefix—`search_tools` and `call_tool`—while keeping the application catalogue
out of the provider's top-level tool array by default. Existing agents retain
their current expose-all request shape. Applications may combine discovery with
an explicit exposure policy for a small directly callable core.

Search is local, model-free, deterministic, and bounded. It considers only
canonical descriptors inside the session's durable capability ceiling, omits
the current direct exposure, and matches normalized name, canonical-id,
description, and input-property terms. Results carry bounded descriptions,
exact admitted schemas, descriptor/schema fingerprints, readiness, and opaque
references. The typed view reuses a reference for an unchanged descriptor,
survives ordinary resume, and is not copied to a fork. Each new session or fork
commits its own typed empty view in the branch-creation transaction; missing,
malformed, stale, or foreign view authority fails closed. Built-in in-memory,
SQLite, and PostgreSQL stores implement the atomic initialization seam required
by discovery-enabled agents.

`call_tool` resolves a discovery reference to the effective registered tool and
validates its inner arguments before the ordinary policy, approval, hooks,
effects, secrets, environment, idempotency, execution, result, and recovery
path. Search schemas and references remain in the private model transcript;
public completion events retain only count, view revision, and truncation
evidence. The same two-schema prefix is used by context-pressure accounting and
all provider adapters. In OpenAI native-targeting mode, runtime-owned
`allowed_tools` keeps both discovery functions callable without changing the
cache anchor.

Portable discovery-enabled conversational request footprints use schema version
6 and record only the current view generation/revision, catalogue and ceiling
identities, and grant count. Provider-native discovery adds the content-free
projection summary in schema version 7. These observation fields do not enter
the provider payload: keyed tool-manifest and cache-prefix fingerprints remain
stable as grants are discovered. Adapter coverage verifies the same two-tool
prefix for OpenAI Responses, Anthropic, Chat Completions, Bedrock, and Vertex.

`CayuApp.inspect_tool_discovery_view(...)` and the authenticated
`GET /api/sessions/{session_id}/tool-view` control-plane route expose a bounded,
content-minimized current view. The HTTP route requires a verified non-null
tenant matching immutable session provenance, returns not-found across tenant
boundaries, and marks responses `private, no-store`. Inspection omits grant ids,
opaque references, query hashes, and schemas. A session-incarnation fence also
prevents delete/recreate races from returning replacement-session metadata;
unavailable registered agents and inconsistent durable views return a
content-minimized conflict. Malformed and unmapped public session aliases share
the same not-found envelope, while unexpected failures retain
`private, no-store` and expose only a generic error after bounded redacted
logging. The additive route advances the server contract from version 28 to
version 29; regenerate committed clients and upgrade separately deployed
control-plane consumers together. No storage migration is required.

The credential-free `examples/tool_discovery_validation` fixture closes the
provider-neutral validation loop. It proves discover, invoke, resume, fork,
copied-reference rejection, child rediscovery, and child invocation through the
ordinary runtime. Its bounded report covers deterministic ranking, unnecessary
searches, invalid arguments, model steps, stable request shape, token/cache
categories, local latency, effects, approvals, exact quality, and synthetic
fixture cost. It is explicitly not a provider benchmark or universal savings
claim.

### Fixed memory interventions have an exact durable execution boundary

Applications can now submit one frozen `MemoryInterventionTrialRequest` to
`MemoryInterventionExecutor.execute_trial()`. The bounded SDK derives isolated
session and causal-budget identities, authenticates the complete request with a
restart-stable HMAC key, verifies its `AgentSnapshot`, and binds the overlay
provider, runtime runner, evaluator, materialization, effect receipt, runtime
evidence, evaluation, and final trial result through an exact compare-and-set
phase journal. In-memory and SQLite journals reconcile lost acknowledgements
without duplicating the logical effect, runtime session, or evaluation, and
retain the bounded typed runtime result needed to preserve failed, cancelled,
timed-out, outcome-unknown, conflicting, and indeterminate outcomes as distinct
evidence after restart.

Session binding now includes an absolute runtime deadline, an authenticated
runtime-owned create claim, and a renewable cross-process dispatch lease.
Retries cannot adopt a foreign session under the deterministic id, restart a
missing session after timeout, or fence a live worker merely because they lost
the phase CAS. The canonical adapter pins the isolated store through actual
session admission, and evaluator idempotency uses the full execution identity
so a new immutable case revision cannot recover another revision's result.

The execution adapters are application-owned authority boundaries, not alternate
recall or provider stacks. Their recovery methods must complete or reconcile the
precommitted operation idempotently, and a runtime adapter must enter Cayu's
ordinary recall, admission, context, provider-dispatch, acknowledgement,
completion, and attribution paths using only the candidate-local overlay.
Executable views now require positive isolated-store authority and the exact
trial recall policy. Cayu also provides
`CayuMemoryInterventionRuntimeRunner`, which validates a concrete application
factory, isolated environment, canonical automatic-recall policy, and exact
trial execution profile before using the normal runtime and durable evidence
paths. Recall-policy variants may change only the profile's automatic-recall
component. Caller cancellation is recorded as separate positive journal
authority so other interrupted sessions remain outcome-unknown, and oversized
runtime attribution is reduced to a truthful bounded truncation record after
successful provider work. Accumulating materializations are partitioned by
fixed intervention identity so variants for the same candidate cannot share
mutable state. Snapshot requests without an intervention partition preserve
the existing materialization identities and serialized record shape.

### Knowledge relations bind reviewed lineage to exact revisions

Built-in in-memory, SQLite, and PostgreSQL knowledge stores now publish and read
immutable `supersedes`, `derived_from`, and symmetric `contradicts` relations
between exact revisions of different logical entries. Bounded atomic batches,
canonical contradiction orientation, immutable operation receipts, stable
cursor pagination, two-endpoint access checks, and metadata-only relation events
provide one backend-parity contract for later reviewed consolidation. A relation
does not itself archive, approve, rerank, traverse, or inject either entry into
model context.

Breaking storage revision 60 installs the relation-aware knowledge schema and
outbox. Stop older knowledge workers before upgrading. Fresh databases and empty
earlier knowledge schemas initialize directly; migration refuses a populated
pre-60 knowledge schema before changing data or DDL. Recreate or replace that
prerelease knowledge database explicitly—there is no backfill, dual-write,
metadata fallback, or legacy relation interpretation.

The provider-free performance gate measures a current-runtime zero-relation
entry-publication control, canonical relation preparation, atomic batch
publication, endpoint-indexed reads across unrelated background relations, and
incremental SQLite storage on every CI run.

### Reviewed supersession decisions commit atomically

In-memory, SQLite, and PostgreSQL knowledge stores now accept immutable,
revision-bound maintenance proposals plus explicit non-model approve/reject
decisions. Approval compare-and-swap checks every reviewed current revision and
commits replacement activation, superseded-source archival revisions, exact
lineage relations, metadata-only outbox changes, and a durable receipt in one
transaction. Rejection records review history without changing lifecycle;
contradiction and derivation preserve active sources. Exact retries are
idempotent, while stale, conflicting, denied, failed, and cancelled attempts
leave no partial state.

Breaking storage revision 63 installs the final decision record. Fresh and
completely empty earlier knowledge schemas initialize directly; populated
pre-63 knowledge schemas fail untouched and require explicit replacement. No
backfill, inferred proposal, legacy interpretation, dual write, or compatibility
wrapper is included. A provider-free performance gate covers the zero-decision
path and bounded multi-source applications.

### Maintenance plans are bounded and independently evaluated

Applications can now pass an exact deterministic maintenance-routing snapshot to
`KnowledgeMaintenancePlanningWorkflow`, inject a strict provider-neutral planner, and
evaluate its draft through a separately identified component. Cayu checks exact source,
policy, routing, and configuration bindings; complete source, relation, and evidence
coverage; directed relation orientation; and replacement-kind policy before semantic
evaluation. The independent evaluator can record unsupported synthesis, information
loss, mishandled contradictions, retention or policy violations, and prompt injection as
bounded content-free findings whose codes come from a closed, kind-bound framework
vocabulary. Routed and omitted signals must also exactly partition and match the supplied
request; a result cannot authorize unrelated candidates by copying request fingerprint
fields.

Sources are storage-reauthorized before planning, after planning, and after evaluation.
Source advances remain deterministic stale rejections, while unavailable currentness
checks after planning or evaluation have distinct retryable outcomes instead of being
misreported as semantic rejection.

Count, byte, concurrency, timeout, model-call, and integer micro-US-dollar cost ceilings
are explicit, measured over-budget usage remains visible, and results retain separately
configured planner/evaluator identities and versions. Component calls run in separately
owned tasks so caller cancellation cannot be consumed, and output observed after a stage
deadline is discarded. Truncated or empty routing does not call either component.
Accepted output remains a read-only evaluated draft: the planning workflow itself cannot
persist a pending replacement, publish relations, activate knowledge, or archive
predecessors.

The workflow independently enforces the hard 50-source ceiling before a storage read,
even when a caller constructs its own structurally valid routing-result object. Planner
budgets disclose the configured evidence-count, claim-size, and replacement-size ceilings
before invocation. Provider model identities are application-authorized per stage; an
unknown component-reported identity invalidates the output without copying that identity
into failure diagnostics.

### Accepted maintenance plans persist as exact pending review artifacts

`KnowledgeMaintenanceProposalPublisher` now performs the explicit atomic handoff from an
accepted planning result to review. It revalidates routing and plan bindings, requires one
identical namespace/label/visibility boundary, and compares every exact source against the
active current revision inside the write transaction. A successful publication stores the
pending replacement and chunks, exact source evidence, accepted semantic plan, review
proposal, one knowledge-outbox change, and an immutable replay receipt together.

Publication grants no lifecycle authority: it cannot activate the replacement, archive a
source, or publish a relation. The existing reviewed-decision workflow must consume the
byte-exact persisted proposal, and altered proposals fail closed. Approval performs the
existing atomic lifecycle and lineage transaction; rejection leaves active source
revisions unchanged. A rejected replacement created by this durable publisher can be
explicitly archived or soft-deleted, but only through forward retirement transitions;
it cannot be activated or content-mutated, and its exact audit revision cannot be
hard-deleted or pruned. Attempt telemetry and timestamps do not affect proposal identity,
so equivalent attempts and retries after a lost response or cancellation converge without
duplicate artifacts. A published rejection remains available from its immutable access
snapshot after source revisions advance or expire; approval continues to require every
source to be current.

Validated accepted plans and publisher configuration cache their immutable fingerprints,
so maximum-source publication and replay do not repeatedly hash the complete plan.

In-memory, SQLite, and PostgreSQL stores share the same behavior and verify the full
stored composite before returning it. Storage revision 67 is a breaking prerelease
boundary that creates an empty proposal-publication table without inferring or backfilling
accepted plans for existing knowledge. There is no dual-write, legacy interpretation, or
compatibility wrapper for pre-67 workers.

### Memory interventions have portable, effect-bound evidence contracts

Applications can now declare an explicit `as_declared`, recall-off, omission,
replacement, or negative-control intervention against one exact
`AgentSnapshot` memory frontier. Canonical precommitment, effect receipt, trial
attribution, and memory-specific comparability records bind isolated overlays
without carrying recalled text, mutable store locations, or production
activation authority. Missing, truncated, redacted, contradictory,
indeterminate, conflicting, no-match, and proven-no-exposure outcomes remain
distinct. The portable schema remains separate from execution and adds no
experiment envelope, optimizer, paired report, or Compound dependency.

- Added an explicitly invoked, provider-neutral `KnowledgeCurator` for reviewed learning.
  Applications can submit bounded source-attributed signals, inject separate candidate
  generators and evaluators plus an optional content policy, and atomically persist only
  accepted proposals as pending revision-bound knowledge. Typed per-signal and
  per-candidate outcomes, deterministic cross-process retries, exact evidence,
  cancellation-safe owned publication, and the existing review workflow keep proposals
  auditable and unavailable to normal recall until approval.
- Added one bounded retained-publication lifecycle shared by `RememberKnowledgeTool` and
  `KnowledgeCurator`. Direct SDK components can now seal and drain with `aclose()`, while
  `CayuApp` and server shutdown automatically seal registered knowledge writers and apply
  a dedicated publication grace period before leaving durable receipt reconciliation to
  the next process.

### Controlled scenarios run end to end from the Control Plane

Operators can now save an immutable scenario-v2 revision, review its current
launch binding, and start it directly with **Run scenario**. The durable worker
executes ordered initial and queued input, explicit session resumes, typed
`ask_user` answers, portable JSON, file references, and fresh approval
checkpoints through the target's ordinary `CayuApp`. The Runs view publishes
per-trial phase and cursor state and presents actor-attributed approve/deny
controls only for a current authored checkpoint. Cancellation works while a
trial is executing or awaiting approval.

Approval, user-input, and explicit-resume checkpoints retain the same durable
runtime session across coordinator and store restart. Claim epochs fence every
progress transition and final result publication; other interrupted stages
restart as a fresh attempt. Provider calls or external tool effects dispatched
immediately before lease loss may still repeat unless their underlying system
honors Cayu's idempotency or reconciliation identity.

Scenario output uses the existing corpus result, JSON, HTML, SDK, CLI report,
comparison, and stable CI-exit contracts. Typed queued messages now preserve
file parts across durable delivery rather than reducing them to their text
projection. Captured ordinary `resume(...)` interactions are projected to the
matching scenario resume kind instead of being mistaken for an `ask_user`
answer.

Server contract version 25 adds scenario launch, durable trial progress, and
fresh scenario-approval operations. Upgrade independently deployed servers,
dashboards, and generated clients together.

Storage revision 56 additively adds the bounded scenario-progress document to
eval runs; scenario-aware `SQLiteEvalStore` and `PostgresEvalStore` instances
require it. Storage revision 57 is a breaking session-worker boundary for exact
typed queued messages. Stop revision-56 and older session workers, back up the
store, run `cayu storage migrate`, confirm revision 57, and then start this
build. Do not run mixed revision-56/revision-57 session fleets: older workers
would silently deliver only the text projection.

### Completion-verifier attempts carry immutable execution profiles

Deterministic completion verification now persists a verifier-specific execution
profile before claiming or dispatching the adapter. The record binds the frozen
work contract, candidate proposal, source attempt/profile, verifier reference,
and bounded application-versioned adapter/component identities. Claims,
renewals, decisions, replay, and decision application carry the same profile
fingerprint, while registration drift and missing components fail before verifier
work.

Later attempts may change verifier behavior only through an explicit
`ExecutionProfileAdoptionIntent` authorized by an application
`CompletionVerifierProfilePolicy`. Exact retries reuse the durable authorization
without rerunning policy. Provider-backed verifier dispatch remains reserved for
the downstream provider-verifier executor.

Breaking storage revision 58 adds the profile registry and required claim and
decision attribution. Migration rejects prerelease SQLite or PostgreSQL stores
that already contain verification claims or decisions because their historical
profile cannot be reconstructed safely. Stop revision-57 and older verifier/task
workers, back up and recreate affected stores, migrate, and confirm revision 58
before starting the new verifier workers.

### Accepted verified-work results resolve through frozen application authority

Work contracts now require an immutable application-owned result-resolver ID,
version, and configuration fingerprint. Applications register the exact
side-effect-free resolver during worker startup and can use
`CayuApp.resolve_completion_result(...)` to reconstruct an accepted result from
durable application state, validate its content-bound reference, and apply it
through the existing immutable decision-application receipt. Exact retries after
commit or process replacement are receipt-first and do not require the resolver.
Runtime events expose resolver and result-reference identities, never result
content.

Breaking storage revision 59 makes that resolver identity part of every verified-
work contract. Stop older task producers and workers, take an application-
consistent backup, and recreate any prerelease SQLite or PostgreSQL task database
whose verified-work contract registry is populated. Empty registries migrate
normally. Mixed revision-58/revision-59 task processes and application-only
rollback are unsupported.

### Contract-bound attempts have a dedicated admission and recovery boundary

Applications can now reserve an initial or rejected/continue contract-bound
attempt with `CayuApp.admit_work_attempt(...)` before any governed provider,
tool, hook, or mutating environment work begins. Bounded read-only workspace
instruction loading and side-effect-free provider preflights establish the
source profile first. The receipt binds the exact task, contract, session,
interaction, attempt, worker lease, process generation, source request, and
source execution profile. Continuations stay on the same session, add a new
interaction, and receive the prior rejected decision and bounded typed gaps
without modifying the frozen contract.

Exact retries reconcile preparation and session receipts. An expired process
generation can be replaced without creating a second attempt, and active
recovery requires positive session-quiescence evidence before the new owner may
continue. A committed recovery-session transition whose task-store activation
was interrupted is taken over only from its exact expired historical claim, and
the replacement atomically advances the session epoch. Competing task or
interaction identities cannot hold two unreleased admissions for one session.
Mutable session progress does not invalidate an already-published exact
admission receipt. Cancellation waits for dispatched session- and task-store
mutations to settle before that caller reports a quiescent outcome; durable
claim and session compare-and-set evidence governs cross-process replacement.
Continuation receipts carry every valid bounded decision without duplicating
its gap payload. Renewals and completion proposals are fenced to the current
process, claim, generation, worker, and live lease; proposal publication
releases that live authority atomically without removing the task's permanent
governed marker. Initial preparation also persists the admission's exact
session incarnation on the task so later result publication cannot resolve
against a same-ID replacement session. Ordinary run, resume, recovery, worker,
direct-attempt, fork, and terminal task entrances remain fail closed for
governed sessions. Admission also completes the durable `interaction.started`
side-effect handoff after attempt activation; exact activation retries deliver
that same event without duplicating it.

The automatic verified-work worker is still downstream work, so deployments
must keep contract-bound tasks out of ordinary worker queues and invoke the
dedicated admission entrance from application orchestration.

Breaking storage revision 61 adds the admission and execution-claim tables.
Stop pre-61 task workers, back up each SQLite or PostgreSQL task store, run
`cayu storage migrate`, and confirm revision 61 before starting current workers.
No historical admission or claim is synthesized, and mixed-version workers are
unsupported.

Breaking storage revision 62 changes deferred interaction input to a bounded
object that can retain the runtime-authenticated complete initial transcript.
Migration preserves revision-61 rows as source-only input and does not infer a
missing system or workspace prefix. A migrated session that still awaits its
initial transcript therefore remains recovery-fenced and must be replaced with
a new session. The migration also backfills each revision-61
continuation receipt's exact predecessor admission from the unique persisted
prior-attempt index and rejects missing or conflicting authority. Stop pre-62
session and task workers, back up each SQLite or PostgreSQL store, run
`cayu storage migrate`, and confirm revision 62 before starting current workers.
Mixed-version workers and application-only rollback are unsupported.

## v0.4.0

`v0.4.0` turns Cayu's evaluation, memory, durable task, provider, and workspace
foundations into portable, independently recoverable runtime contracts. It also
adds first-class OpenRouter support and expands the Control Plane path from
captured production evidence to bounded fresh evaluation.

### Upgrade from v0.3.0

Pin the complete application to `cayu==0.4.0`, refresh its lockfile, and run
its own tests. Stop all `v0.3.0` workers and take an application-consistent
backup before changing any durable store. Do not run mixed `v0.3.0` and
`v0.4.0` workers.

The storage schema advances from revision 45 to revision 55. Follow the
revision-specific boundaries below: revisions 46 through 55 include multiple
breaking contracts, and populated prerelease session or task stores may require
the documented recreation procedure rather than an in-place migration. Run
`cayu storage status` and `cayu storage migrate` against every explicitly
configured SQLite or PostgreSQL store, then confirm revision 55 with no pending
migrations before starting `v0.4.0` workers.

The server contract advances from version 16 to version 24, while the public
application manifest and generator plan advance from schema version 9 to
version 11. Regenerate committed manifests, plans, API clients, and dashboard
assets, and upgrade independently deployed servers, workers, generated clients,
and dashboards together. After deployment, verify `cayu version`, run
`cayu check --json`, execute the application's test suite, and exercise its
durable recovery and evaluation paths through a process restart.

### Portable AgentSnapshot manifests bind stateful evaluation lineage

Applications can now capture a strict, versioned, content-addressed
`AgentSnapshot` over exact logical component identities, then verify and
materialize isolated candidate trials through component-owned adapters. The
typed `MemoryStateRef` keeps knowledge, transcript/artifact evidence, work
context, recall/admission/projection policies, receipt/exposure frontiers, index
readiness, and learning disposition distinct. Strong capture requests fail
closed instead of presenting partial or inconsistent state as complete.

Restorable memory and workspace components require candidate-private overlays;
trial reset versus candidate-local accumulation is explicit. Trial and result
bindings connect the starting snapshot and overlays to ordinary Cayu sessions,
hidden-case/evaluator aliases, terminal evidence, eval revisions, usage, and
cost without embedding credentials, private content, hidden truth, provider
continuations, or activation authority. The SQLite snapshot journal supports
fresh-process recovery without repeating materialization effects. This is a
logical reproducibility envelope, not a database, VM, process, or production
activation snapshot, and it adds no runtime storage migration.

Snapshot schema v2 now includes component provider identity and transactional
consistency group in the content address. Materialization also writes a durable
scope plan and stable per-component operation identity before provider effects,
then compare-and-set checkpoints each result. A provider-owned recovery seam
reconciles an acknowledgement-lost or process-interrupted operation by that
exact identity without blindly redispatching it. Same-scope coordinators
converge in both stores, completed components are never replayed after a later
failure, and invalid same-key documents or forged same-revision progress fail
closed while relocation-only references remain portable. Content-addressed
loads and SQLite scope indexes are cross-validated before use, provider effects
require the exact returned durable claim, recovery rebinds every component to
the verified snapshot plan, and trials reject reset-scope, evaluator, or overlay
kind drift. Conflict refreshes also reject revision rollback, removed completed
evidence, or a final pointer to a foreign same-scope progress identity.

### Production sessions can be captured as scenario-v2 stimuli

Authenticated captured-evaluation previews now independently reconstruct an
authority-free scenario when retained production evidence is sufficient.
Runtime-owned transcript attestations preserve exact initial, resumed, and
delivered queued inputs across the in-memory, SQLite, and PostgreSQL session
stores. Approval history becomes a fresh-decision checkpoint. Retained file
inputs become digest-bound artifact requirements after current scope, metadata,
access, and content checks; neither file bytes, queue ids, approval ids, actors,
nor historical authorization enter the scenario.

Conversion is side-effect-free and fail-closed. Redacted or older unattested
input, missing/inaccessible artifacts, contradictory evidence, limit failures,
and a source that changes during capture produce stable factual diagnostics.
They do not disable captured scoring, saving, export, or baseline comparison.
The Control Plane reports the scenario result beside runnable corpus-v2
conversion. At that delivery slice, scenario editing and controlled multi-stage
execution remained later workflow layers; both are included in the current
unreleased build above.

Server contract version 21 adds the required `scenario_conversion` result to
captured-evaluation preview responses and marks scenario conversion ready when
captured session evidence is ready. Upgrade independently deployed servers,
generated clients, and dashboards together.

Storage revision 54 is a breaking reader-safety boundary for the private
resume, queued-input, and source-time file attestations used by capture. It adds
an independent file-attestation proof column and fabricates no historical proof.
Stop revision-53 session workers, run
`cayu storage migrate`, confirm revision 54, and then start this build. Older
sessions remain evaluable as captured results; file-bearing sessions without the
new source-time proof return an exact scenario-conversion diagnostic. Do not run
mixed revision-53/revision-54 session fleets.

### Portable multi-stage eval scenarios

Cayu now defines an immutable scenario-v2 document for ordered initial,
queued, and resumed user input plus fresh-approval checkpoints. Text, portable
JSON, and bounded file requirements can be authored without embedding artifact
contents, secret values, actor identity, approval decisions, providers, tools,
environments, or any other executable authority. Runnable corpus-v2 cases can
be converted explicitly; captured-only cases still require authored stimuli.

Built-in eval stores persist scenario documents behind the same fail-closed
credential-redaction boundary and bounded newest-first catalogs as corpora.
Storage revision 53 is an additive migration that creates the independent
scenario catalog. Run `cayu storage migrate` and confirm revision 53 before
using scenario persistence. Revision-52 workers may coexist because they do not
write this table, but scenario-aware `SQLiteEvalStore` and `PostgresEvalStore`
instances require revision 53. That delivery slice established portable
authoring and durable storage; the current unreleased build above adds
target-bound execution.

### OpenAI can project targeted grants as cache-stable native tools

Agents now select targeted-grant delivery with `targeted_tool_mode`. The
portable `"call_tool"` mode keeps one stable gateway definition, while
`"openai_additional_tools"` projects a grant through the OpenAI Responses
`additional_tools` input contract. The explicit
`"openai_additional_tools_or_call_tool"` mode selects native delivery only for
an exact model listed in that provider registration's
`additional_tools_models`; unsupported providers and unverified model or
endpoint combinations use the gateway before dispatch. Cayu does not infer
native support from model-family names or compatible base URLs.

Native definitions come only from the admitted canonical catalogue. Cayu
commits a schema-free runtime marker immediately after the interaction input
that acquires the grant and expands it at that exact position on every request,
retry, overflow recovery, interruption continuation, and stateless replay. A
fork therefore preserves its inherited prefix and gains addressability only
from a fresh child-scoped grant. Server-state and background Responses
continuations retain only the marker identity needed to prevent an inactive
native item from surviving its Cayu authority.

Returned native calls resolve and atomically consume the same durable grant as
the gateway before entering the existing policy, approval, hook, effect,
secret, environment, concurrency, idempotency, execution, receipt, and recovery
path. Request footprints identify the selected projection and insertion
position without retaining schemas, references, arguments, or results. The
credential-gated `examples/openai_targeted_tools_live.py` contract checks a real
native call and requires cached inherited-prefix tokens on the child's first
request. A grant-free sibling first writes that exact prefix; the memory-writing
child then adds only the native targeted-tool suffix. Native mode keeps one
canonical `call_tool` definition as a stable top-level cache anchor, while a
runtime-owned OpenAI `allowed_tools` choice always excludes that anchor and
makes only ordinary or actively granted functions callable. Both requests use
a session-scoped cache-routing key and GPT-5.6 implicit caching.

This intentionally replaces the prerelease `enable_tool_gateway` registration
argument and `tool_gateway_enabled` manifest field; use
`targeted_tool_mode="call_tool"` for the portable behavior. The public
application manifest and generator plan advance to schema version 12. Server
contract version 26 replaces the agent manifest boolean with the nullable
targeted-tool mode; independently deployed servers, generated clients, and
dashboards must be upgraded together.

### Targeted grants can execute through the portable `call_tool` gateway

`RunRequest` and `ResumeRequest` can now carry strict `TargetedToolGrant`
requests for one canonical tool already registered inside the session's durable
capability ceiling. Cayu issues an opaque interaction-scoped reference backed by
durable expiry, revocation, and positive call-budget state. Exact retries rejoin
one digest-only consumption binding; altered replay, copied references, scope or
catalogue drift, task-boundary changes, and exhausted grants fail closed. Forks
copy no targeted grant or consumption authority and record an explicit reset
event.

Agents select `targeted_tool_mode="call_tool"` at registration. Cayu then keeps one exact
provider-independent `call_tool(tool_ref, arguments)` definition in every
request for that execution profile, while an active grant appends its bounded
runtime-authored descriptor context after conversation history. This preserves
the provider-visible system, tool, and inherited-message prefix across a fork.
Resolving an outer call validates the current descriptor and inner arguments,
then atomically consumes or rejoins the grant before the ordinary policy plan.
The canonical target continues
through the existing approval, hook, effect, secret, environment, concurrency,
idempotency, execution, receipt, and recovery paths; no wrapper executor or
parallel authority path was added.

Provider transcripts retain the outer `call_tool` name and call id but use a
fixed non-authoritative placeholder instead of the opaque reference. Private
recovery state retains the exact runtime-issued reference with its selected
grant identity and revalidates the pair against the durable store before binding
or rejoining. Exact retry and approval continuation therefore rejoin one
consumption even when the reference text collides with workload-secret
redaction. Model-authored lookalikes remain untrusted. Invalid references, stale
scope, exhausted grants, invalid inner arguments, and schemas requiring external
reference retrieval fail before policy or target-side work. General events and
request footprints never expose the opaque reference. Requests without active
grants omit the dynamic descriptor suffix but retain the opted-in gateway tool.
Agents that do not opt in retain their existing request and execution behavior,
and a targeted-grant request against one of those agents fails before issuance.

Server contract version 20 adds the targeted-grant admission count and batch
fingerprint to interaction summary evidence. Independently deployed servers,
generated clients, and dashboards must be upgraded together.

Storage revision 52 adds first-class grant and digest-only use tables plus the
public-alias lookup needed for opaque references. Current session stores require
revision 52. Revision 52 is a clean prerelease break: stop revision-51 workers,
back up and recreate populated SQLite or PostgreSQL session stores, then run
`cayu storage migrate` for empty stores and confirm revision 52 before starting
this build. Migration rejects populated pre-52 session stores without mutation;
there is no fork-evidence backfill or legacy replay path. Do not run mixed
revision-51/revision-52 fleets.

### Owner-lost retry cancellations can be reconciled from positive evidence

Applications can now call `reconcile_task_retry_cancellation(...)` when a
retry-series worker dies after durable cancellation has won. The typed request
binds the exact task, series, attempt, causal budget, original worker and lease,
cancellation marker, reconciliation identity, actor, and bounded validator
receipt. Lease or deadline expiry alone remains insufficient; inconclusive or
stale evidence leaves the attempt fenced.

Validated quiescence, completed-effect, or failed-effect evidence atomically
writes the ordinary cancellation settlement receipt, clears ownership, keeps
the series disposition `cancelled`, and creates no successor. Identical calls
replay that receipt, while changed evidence and late-worker races produce typed,
bounded conflict evidence. In-memory, SQLite, and PostgreSQL share the contract,
and a subprocess SIGKILL test proves fresh-process SQLite reconciliation without
handler replay. Retry-task and claim-worker identities are now capped at 1,024
UTF-8 bytes before work ownership, and reconciliation actors must carry an
explicit provenance source.

Rejected reconciliation evidence now atomically binds the task-scoped
reconciliation idempotency key in a separate durable registry without changing
the fenced task. Exact rejected calls replay the same bounded rejection event;
changed evidence under the key fails with a typed conflict. Breaking storage
revision 55 creates the empty registry. Stop revision-54 task workers, back up
each SQLite or PostgreSQL store, run `cayu storage migrate`, and confirm revision
55 before starting current task workers. No legacy reader or synthesized
rejection backfill is provided.

### Captured and fresh Evals share one release gate

The Control Plane now compares immutable captured-session and fresh-execution
results through one origin-aware server contract. An approved suite baseline is
used by default; operators may override it with another immutable result
revision. Compatibility remains contract-based, allows application release and
manifest changes, and reports typed incompatibilities instead of manufacturing
regressions. Both result origins can be downloaded as deterministic JSON or
standalone HTML.

`cayu eval report` now accepts captured result documents, and `cayu eval compare`
accepts captured-to-fresh, fresh-to-captured, and same-origin published results
with the existing stable `0`/`1`/`2` CI exits. The public SDK exports the shared
result serializers and report renderers. Server contract version 19 adds the
immutable result comparison and catalog report endpoints; independently
deployed servers, generated clients, and dashboards must be upgraded together.

Freshly generated projects document the complete simple-session path: configure
the normal provider, start `cayu serve --dev`, evaluate a session, approve its
captured baseline, and launch one bounded fresh trial without Evals-specific
Python. The credential-gated release check now creates that scaffold and proves
the Control Plane plus JSON, HTML, CLI, and CI round trip with one source run and
one fresh trial. The fresh-launch slice itself adds no storage revision. Current
scenario-aware eval stores require the additive revision-53 migration, while
session stores that capture new resume and queued-input proof require the
breaking revision-54 boundary documented above. The shared restart suites cover
scenarios as well as corpora, results, baselines, and run publication.

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

### Recall and provider exposure become separately auditable

Session stores now persist immutable, bounded `RecallReceipt` records for exact
retrieval/admission decisions and revision-fenced `ContextExposure` lifecycles
for exact model/provider attempts. The automatic-recall runtime now constructs
each receipt from the exact fused result already used for admission, then
creates an attempt-scoped exposure from the final provider request. Per-item
links must match receipt identity,
revision, representation, content hash, and locator exactly. Reused identities,
cross-interaction links, duplicate items, stale transitions, and post-terminal
updates fail closed. A frozen receipt may support multiple model steps within
its owning interaction. In-memory, SQLite, and PostgreSQL implementations share
the same idempotency, concurrency, pagination, and session-deletion behavior.

The lifecycle deliberately distinguishes planned composition, prepared request,
durable dispatch intent, provider acknowledgement, completion, conclusive
failure/cancellation, and ambiguous transport. Planning or dispatch intent is
never reported as positive evidence that the model saw memory. Evidence stores
retain bounded identifiers and hashes, not raw user queries, recalled text,
prompts, or provider payloads. Private structured material uses keyed,
domain-separated HMAC fingerprints so stored digests do not become offline
guessing oracles. Built-in source locators use a closed typed union; custom
locators retain only a domain-separated keyed fingerprint, so arbitrary source
metadata or credentials cannot enter durable evidence. Receipt/exposure page
reads use bounded keyset lookahead and do not count or scan the remaining
history.

Storage revision 51 is additive and creates only empty evidence tables and
indexes. It does not backfill historical sessions or add a legacy compatibility
path. Automatic-recall checkpoints use a purpose-separated HMAC to bind the
receipt ID and exact durable document to the rendered manifest; missing or
mismatched bindings are rejected.
Required evidence persistence fails closed before provider network I/O; retries,
overflow rebuilding, typed provider outcomes, cancellation, and durable
background recovery advance the original attempt lifecycles conservatively.

`runtime_evidence(...)` schema 4 and standalone trajectory schema 4 now expose
the same bounded `cayu.memory_attribution.v1` read model. It correlates receipts,
exposures, item links, and lifecycle truth through session-scoped HMAC aliases
without publishing raw memory identity or content. Global count and byte bounds,
lower-bound omission counts, and distinct unavailable, redacted, truncated,
contradictory, and exposure-level indeterminate states keep incomplete evidence
honest across complete session trees. Historical trajectory promotion remains
read-only. A checked hermetic baseline and dedicated no-coverage current-code regression
lane record preparation, persistence, zero-record and populated public projection
latency, SQLite storage, and serialized-size ceilings without provider calls.

Storage revision 55 is a breaking task-store boundary that creates the empty
retry-cancellation reconciliation rejection registry. It does not mutate tasks
or fabricate historical rejection records. Revision-54 task workers must be
quiesced before migration and cannot share the migrated database.

### OpenAI subscription retries are bounded with migration-explicit authority

OpenAI subscription HTTP and SSE errors now preserve bounded typed retry
classification. Known terminal failures remain terminal, known transient
failures follow the caller's ordinary retry policy, and genuinely unclassified
provider failures use the stricter `RetryPolicy.max_unknown_attempts` ceiling,
which defaults to at most two total attempts. Durable `model.error`,
`model.retry`, and `model.attempt_discarded` evidence keeps the caller's general
`max_attempts` separate from the classification-specific
`effective_max_attempts` and records `reason="unknown_provider"` for unknown
failures through terminal exhaustion. Conflicting explicit SSE status fields
remain terminal and now omit `status_code` instead of publishing a synthesized
status as observed evidence.

This semantic addition deliberately advances model-finalization profile
material from v1 to v2 and the built-in `ModelCompactor` and
`PromptCacheCompactor` materials from version 1 to version 2. Existing
schema-v3 sessions can therefore report `finalization` and, where applicable,
`context_compaction` drift. Start a new session or explicitly adopt the new
profile through an application policy and an authority-authorized adoption
intent; the default policy rejects the drift before work. Pre-change serialized
adoption requests have a different fingerprint and must be resubmitted under
the new contract. Queued-dispatch envelope authority advances from schema v1 to
v2. Readers accept only the current version; discard pre-change records and
recreate disposable prerelease state. No compatibility migration is provided.

### Terminal sessions settle exact model-completion attempts

Failed and interrupted interactions now commit their terminal lifecycle event,
session status, exact model-attempt disposition, and active-stage release in one
session-store transaction. Definite authentication rejection before any valid
provider output is recorded as `failed_before_provider_effect`; late or mixed
authentication failures and other failures after dispatch may have begun are
recorded conservatively as `provider_effect_outcome_unknown`. Live retries
atomically mark the prior attempt `superseded`, and a stale worker cannot later
complete that attempt. Completed attempts retain the existing immutable
completion and publication receipts. Exact acknowledgement replay, settlement
inspection, and typed manual recovery use the same identities in memory,
SQLite, and PostgreSQL. A separate exact dispatch receipt now marks the last
local boundary before provider-controlled code and occupies a reserved runtime
operation namespace that public operation APIs reject. Exact active-stage
validation and receipt insertion share one backend lock or transaction, so a
live retry cannot supersede the attempt between them. Incomplete recovery
automatically releases the exact linked budget batch and abandons a receipt-less
preparation without a provider call, including when the budget dispatch fence
committed before its acknowledgement was lost,
while a receipt-bearing synchronous attempt remains conservatively
outcome-unknown unless provider recovery evidence can reconcile it.
Terminal stage settlement now also requires a durable terminal settlement and
audit outbox row for every linked reservation. An accounting failure therefore
retains the active stage and original provider or cancellation failure for exact
recovery instead of orphaning shared budget capacity behind a terminal session.
`CayuApp.recover_model_completion_stage(...)` is the operator boundary for that
ambiguous synchronous case: it validates frozen reservation authority,
conservatively charges and publishes every linked settlement, verifies the
complete reservation set, and only then atomically terminalizes the interaction
and clears the stage. Exact retries replay the durable result. Direct
`SessionStore.publish_interaction_transition(...)` settlement is a backend seam,
not an operator recovery shortcut.

Custom `SessionStore` implementations must accept the optional
`model_completion_stage_settlement` argument to
`publish_interaction_transition(...)` and atomically persist its settlement,
interaction event, transition receipt, session status, and active-marker
deletion. The request's `settled_reservation_ids` must exactly equal the active
stage's ordered reservation set. They must also implement the protected exact
dispatch and settlement lookups plus the atomic dispatch hook using the reserved
runtime namespace.
These hooks deliberately have no lossy compatibility fallback; custom stores
must provide equivalent transaction and fail-closed validation semantics before
running this release.

Custom `BudgetLedger` implementations must also implement the atomic
`release_pre_provider_dispatch(...)` transition. It releases a complete batch
only when every reservation is unfenced or bound to the supplied exact dispatch
identity, retains idempotent release settlement outbox rows, and rejects a
conflicting dispatch or terminal outcome without partially freeing capacity.

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

### Portable workspace branches reach retained E2B allocations

Workspace adapters now expose typed branch capabilities for isolation, net
changes, cooperative atomic publication, recovery, retention, and lifecycle
inspection. Unsupported remains the exact default. An explicitly enabled
`RunnerWorkspace` on `E2BRunner` retains its bounded branch journal and private
overlay inside the allocation, supports fresh-process recovery by branch ID,
and shares the local isolation, conflict, publication, rollback, and resource
conformance contract when paired with an explicitly durable, cross-process
binding-claim provider. Public evidence contains content identities only; raw
files, provider handles, credentials, environment data, and command output stay
inside the provider boundary.

`LocalWorkspace` now applies both durability boundaries: durable creation,
recovery, retention, and recoverable-by-ID inspection require a branch journal
whose typed durability is `durable` and a provider whose typed claim scope is
`durable`. `SessionWorkspaceBranchStore` preserves the wrapped session store's
exact durability declaration, so development-only stores such as
`InMemorySessionStore` cannot advertise or admit cross-process durability. The
built-in process-local authority registry continues to support attached local
branches but cannot activate durable recovery by itself.

The public application manifest and generator plan advance from schema version
9 to version 11, and the server contract advances from version 22 to version
24, so `cayu inspect`, server environment inventory, generated clients, and
packaged control-plane consumers can inspect both the bounded branch declaration
and the actual content-free lifecycle states of attached branches.
Regenerate committed manifests, plans, and API clients, and upgrade separately
deployed servers and dashboards together. No storage migration is required.

### OpenRouter is a first-class provider choice

Generated applications can now select `openrouter` through `cayu new --provider
openrouter` or `CAYU_PROVIDER=openrouter`. Live use requires
`OPENROUTER_API_KEY` and an explicit `CAYU_MODEL` slug; Cayu does not select a
mutable, free, or paid default. Optional attribution and bounded router metadata
remain application-controlled, and routing preferences continue through
`provider_options["openrouter"]`.

The Chat Completions seam now preserves streamed OpenRouter
`reasoning_details` as opaque private state and replays the complete unchanged
sequence across tool continuations, retries, recovery, and durable stores.
Selected-route evidence and effective model identity are bounded, OpenRouter
errors retain typed retry/context identity, and cache-read, cache-write, and
reasoning usage normalize without treating provider-reported cost as a Cayu
PriceBook estimate.

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

### Generated Evals targets and immutable baseline indices

Generated project serving now publishes a bounded, stable evaluation target for
every registered agent while retaining executable authority only in process.
Revision 47 adds the origin-aware immutable Evals result index,
actor-attributed baseline pointers, and idempotent baseline mutation audit. It
indexes existing fresh results without copying or fabricating execution
documents. Every fresh-result writer must maintain that index atomically, so
revision-46 workers must be stopped before migration and cannot share the
migrated database.

### Tool exposure remains continuous across durable interactions

Ordinary same-session resumes and queued dispatches now seed the next policy
request from the latest runtime-attested durable exposure profile. This keeps
phase selection and `profile_changed` evidence continuous across process
reconstruction; malformed or caller-authored lookalike evidence fails closed.

Tool exposure now emits the typed, content-minimized `tool.exposure.recorded`
evidence reserved by the public contract: profile and resolved fingerprint,
registered/ceiling/exposed counts, provider/model/step identity, and profile
transition state, with no separate tool-name list, tool definitions, arguments,
or policy reasoning. Application-selected profile ids are public and must be
stable non-secret labels. Conversational request footprints advance to schema
version 3 and bind the same exposure summary to their keyed tool-manifest and
cache-prefix identities. `LLMJudge` now runs under a durable zero-tool
capability ceiling even when its registered agent has tools; adversarial
candidate content cannot expose or execute them. The paired tool-exposure
economics fixture reports requests, retries, cache categories, provider usage,
quality, and cost without claiming a universal winning strategy.

### Local workspace branches survive process replacement

An exact `LocalWorkspace` can create an isolated branch from a complete
observed workspace revision. Branches expose the ordinary workspace API,
deterministic content-free net changes, explicit rollback, and conflict-checked
publication. Source snapshots, overlays, evidence, lifetime, and active branch
count are bounded; unsafe paths, symlinks, special files, stale baselines, and
publication conflicts fail closed. Other workspace backends remain explicitly
unsupported.

Local branches can opt into durable recovery by supplying a
`WorkspaceBranchStore`, such as the runtime's `SessionWorkspaceBranchStore`
adapter for `SessionStore`, together with stable branch, idempotency, run-epoch,
and binding authority. Creation, publication intent and progress, commit,
rollback intent, rollback, conflict, expiry, failure, and ambiguity survive
process replacement across bundled file-backed SQLite and PostgreSQL stores.
Recovery mutates only source paths that still match an exact recorded before
state, recognizes already-applied paths, and reports durable ambiguity instead
of guessing when external content appears. Stale run owners remain fenced
through mutation and terminal settlement, and publication keys stay bound to
their first bounded change-set attempt.

### Deterministic completion verifiers publish independently owned decisions

Applications can register a side-effect-free deterministic completion verifier
under the exact durable verifier identity carried by a work contract, then ask
`CayuApp.verify_completion_proposal(...)` to evaluate a persisted proposal. The
runtime resolves and invokes the adapter with bounded immutable context, binds
its outcome to the live durable claim and frozen contract, and publishes the
decision without applying it to task or session state. Exact completed retries
reconcile from the store without requiring the process-local adapter, while
missing registrations, provider verifier kinds, malformed outcomes, conflicting
identity, and capacity exhaustion before claim mutation fail closed.

The complete contract, task binding, attempt, proposal, verifier claim,
decision, and decision-application receipt lifecycle persists with matching
semantics in `InMemoryTaskStore`, `SQLiteTaskStore`, and `PostgresTaskStore`.
Session execution authority is also durable: an ordinary admission and a
contracted task attachment race to one database-owned decision, and neither
process restart nor task terminalization weakens the winner.

Breaking storage revision 49 adds those task-store records and the task's
immutable contract reference. Stop all revision-48 and older task workers, take
an application-consistent backup, run `cayu storage migrate`, and confirm
revision 49 before starting current workers. Existing ordinary tasks migrate
with no contract binding. Mixed-version task workers and application-only
rollback are unsupported because older workers can complete contracted tasks
through an ordinary terminal entrance.

`CayuApp.apply_completion_decision(...)` owns the public transition from a
durable verifier decision to task state. It validates the immutable authority
chain, applies the bounded exact request through the cancellation-quiescent
store boundary, and requires the atomic application receipt before returning
the applied task. Exact retries replay the receipt snapshot, including after
later task progress, while accepted results remain application-owned and must
be reconstructed from a durable result reference whose digest matches the
accepted proposal.

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

The storage schema advances from revision 36 to revision 45. Follow the
revision-specific migration boundaries below: revisions 39 through 45 contain
breaking durable contracts, and populated legacy knowledge or task stores may
require the explicitly documented rebuild or drain procedure. Run `cayu storage
status` followed by `cayu storage migrate` against every configured SQLite or
PostgreSQL store, and confirm revision 45 with no pending migrations before
starting `v0.3.0` workers. Mixed-version deployment and application-only
rollback across these boundaries are unsupported.

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
state. Execution-target assembly is a later slice, so automatically assembled
projects currently report `eval_target_not_configured` and do not mount Evals
mutation routes or workers.

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

An exact `LocalWorkspace` can now create an isolated, process-local branch from
a complete observed workspace revision. Branches expose the ordinary workspace
API, deterministic content-free net changes, explicit rollback, and
conflict-checked all-or-none publication. Source snapshots, overlays, evidence,
lifetime, and active branch count are bounded; unsafe paths, symlinks, special
files, stale baselines, and publication conflicts fail closed. Other workspace
backends remain explicitly unsupported, and durable reconstruction after
process loss is not part of this first slice.

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
