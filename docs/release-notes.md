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
receipt binds the logical publication identity, intent, source fences, transcript
cursors, and ordered referenced events. References are typed ID-and-canonical-
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
