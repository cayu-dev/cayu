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
