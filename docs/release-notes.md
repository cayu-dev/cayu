# Release notes

## v0.1.0 (unreleased)

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
