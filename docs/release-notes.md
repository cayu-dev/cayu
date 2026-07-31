# Release notes

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

## v0.1.0 (unreleased)

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
