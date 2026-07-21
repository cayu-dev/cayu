# Release notes

## v0.1.0 (unreleased)

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
