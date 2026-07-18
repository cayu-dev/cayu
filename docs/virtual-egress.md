# Virtual Egress Credentials

A credential-brokering path for isolated runners. An app (for example a FastAPI
app using Stripe) is configured with a **virtual credential** that looks normal
to the app, while a trusted **egress broker outside the runner** swaps in the
real vault secret on the way to the provider. The runner never receives the real
secret — not in env, files, `/proc`, logs, or on the wire. Isolation strength
still comes from the explicitly selected runner: ordinary Docker containers are
not a secure sandbox boundary.

## Why not just `secret_env`?

`secret_env` (mode `raw_env`) resolves a vault secret and injects the **raw
value** into the runner workload process environment. Redaction scrubs logs,
but the secret is still *present* in the workload and the agent can read it.
That is a convenience mode, not a security boundary.

## Credential modes

| Mode | Secret readable by agent? | Use for |
| --- | --- | --- |
| `raw_env` | Yes | Trusted local/dev environments. The existing `secret_env`. |
| `trusted_tool` | No | Host-side bounded actions (send one email, create one invoice). |
| `virtual_egress` | No | Normal apps under development that need provider SDKs to work unchanged. |

See `cayu.CredentialMode`. `raw_env` remains fully supported and
backward-compatible; it is simply labeled as *raw secret injection*.

## Security invariant

Stronger than redaction — the property is **non-possession**:

- Vault secret → trusted broker code only.
- Runner workload env → virtual credential only.
- Runner workload network → in an enforced adapter, cannot reach the
  credentialed provider except through the broker.
- **Fail closed:** the virtual-egress environment factory refuses any runner
  family without a registered enforcing adapter (`UnsupportedEgressError`)
  rather than downgrading to raw injection.

## How it works

```
Docker container                 Host (trusted)
  |  HTTPS to https://api.stripe.com
  |  Authorization: Bearer sk_test_cayu_vc_...
  v
Internal Docker network (no internet route)
  |  only reachable egress is the sidecar
  v
Egress sidecar  --->  TransparentEgressProxyServer (per-session CA, TLS MITM)
                        |  TransparentEgressBroker.handle_request:
                        |   1. look up virtual credential (registry)
                        |   2. authorize (HttpEgressPolicy) — DENY happens here,
                        |      before any vault resolve
                        |   3. resolve SecretRef from vault (broker code only)
                        |   4. swap Authorization -> real secret
                        |   5. forward upstream, scrub response
                        v
                      api.stripe.com
```

In the explicitly selected Docker topology shown above, direct egress is blocked
by construction: the container joins an `--internal`
Docker network with no route to the internet, so the only reachable egress is a
dual-homed sidecar that forwards to the in-process broker. This network control
does not turn the container into a secure sandbox boundary.

## API shape

Most apps use the root-level setup API: `CredentialMode`, `HttpEgressPolicy`,
`ApprovedEgressDestination`, `VirtualCredentialSpec`, and
`VirtualEgressEnvironmentFactory`.

Lower-level extension points live under `cayu.egress` and `cayu.runtime.egress`:
custom `EgressPolicy` implementations, `SandboxEgressAdapter` registrations,
proxy exposure adapters, and the broker/proxy contracts used by adapters. Each
egress adapter also creates its matching runner, so enforcement cannot be
prepared and then accidentally discarded by an unrelated runner factory. Keep
provider-specific business authorization in the app, provider-scoped
credentials, or a custom policy.
Supported credential shapes are closed for now: `stripe_bearer` and
`opaque_bearer`.

Install the TLS/Docker pieces with the optional extra:

```
pip install 'cayu[egress]'   # adds cryptography
```

## Runtime integration (CayuApp)

`virtual_egress` is a first-class, session-lifecycle-managed mode via
`VirtualEgressEnvironmentFactory` (`cayu.runtime.egress`). Register it as an
environment factory; per session it mints grants, stands up the broker plus the
explicitly selected enforced runner, and tears everything down at session end
(the workspace binding's
`finalize` hook — which the runtime already calls — revokes grants, removes the
network/sidecar, and stops the proxy).

Credential `env_name` values must be unique. Duplicate names are rejected at
factory construction so the sandbox cannot receive one virtual value while the
broker minted several ambiguous grants.

```python
from cayu import (
    CayuApp, EnvironmentSpec, HttpEgressPolicy, SecretRef, StaticVault,
    VirtualCredentialSpec, VirtualEgressEnvironmentFactory,
)
from cayu.runtime import VIRTUAL_EGRESS_EVENT_TYPES

app = CayuApp()
vault = StaticVault({"stripe_test_key": "sk_test_..."})
factory = VirtualEgressEnvironmentFactory(
    resolver=vault,                              # resolves the real SecretRef, broker-side only
    policies={"stripe-example": HttpEgressPolicy(
        name="stripe-example",
        allowed_hosts=["api.stripe.com"],
        allowed_endpoints=[("POST", "/v1/customers")],
    )},
    credentials=[VirtualCredentialSpec(
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        policy_name="stripe-example",
    )],
    runner_kind="docker",                      # explicit trusted container selection
    image="cayu-egress-fastapi-stripe:demo",     # FastAPI + Python HTTP client
    event_emitter=app.scoped_event_emitter(
        event_types=VIRTUAL_EGRESS_EVENT_TYPES,
    ),                                          # stream only virtual-egress audit events
)
app.register_environment_factory(EnvironmentSpec(name="billing"), factory, default=True)
```

Sessions on that environment run in the explicitly selected Docker container
with `STRIPE_SECRET_KEY` set to the virtual credential; the real key is swapped
in only by the broker.
Grant revocation is enforced against in-flight broker requests: teardown marks
the grant revoked, waits for active request leases to drain, and the broker
re-checks liveness after vault resolution before forwarding upstream.

### Credentialless approved destinations for coding agents

An approved destination can use the same enforced egress network without a
secret. This is useful when a coding agent needs one public documentation page,
source archive, or package-registry endpoint but must not receive general
internet access. `ApprovedEgressDestination` is structurally separate from
`VirtualCredentialSpec`: it has no `SecretRef`, mints no virtual value, adds no
guest environment variable, and never calls a resolver.

This complete environment registration permits exactly one Python documentation
request and denies other hosts, paths, methods, ports, direct IPs, and metadata
access through the same adapter boundary:

```python
import asyncio

from cayu import (
    ApprovedEgressDestination,
    AgentSpec,
    CayuApp,
    EnvironmentSpec,
    ExecCommandTool,
    HttpEgressPolicy,
    Message,
    OpenAIProvider,
    RunRequest,
    VirtualEgressEnvironmentFactory,
)

app = CayuApp()
factory = VirtualEgressEnvironmentFactory(
    policies={
        "python-asyncio-doc": HttpEgressPolicy(
            name="python-asyncio-doc",
            allowed_hosts=["docs.python.org"],
            allowed_endpoints=[("GET", "/3/library/asyncio.html")],
        )
    },
    approved_destinations=[
        ApprovedEgressDestination(
            destination="docs.python.org",
            policy_name="python-asyncio-doc",
            protocol="https",
            port=443,
        )
    ],
    credentials=[],
    runner_kind="docker",  # explicit trusted container selection
    # No resolver is required because this environment has no credentialed route.
    image="python:3.13",
)
app.register_environment_factory(
    EnvironmentSpec(name="coding-sandbox"),
    factory,
    default=True,
)
app.register_provider(OpenAIProvider(), default=True)  # reads OPENAI_API_KEY
app.register_agent(
    AgentSpec(
        name="builder",
        model="gpt-5.4-mini",
        system_prompt="Use the shell tool to inspect approved public documentation.",
    ),
    tools=[ExecCommandTool()],
)


async def main() -> None:
    request = RunRequest(
        agent_name="builder",
        session_id="bounded-docs-demo",
        messages=[
            Message.text(
                "user",
                "Use python3 and urllib.request to fetch "
                "https://docs.python.org/3/library/asyncio.html, then show that "
                "https://example.com is denied.",
            )
        ],
    )
    async for event in app.run(request):
        print(event.type, event.payload)


asyncio.run(main())
```

Cayu v1 intentionally supports only broker-terminated HTTPS on port 443 for
credentialless declarations. A redirect response is returned to the guest
client normally; if the client follows it, the resulting request passes through
the broker again and the target must have its own declaration and policy. Cayu
does not infer or auto-approve mirrors, CDNs, package dependencies, or other
transitive hosts. Caller-provided headers, including `Authorization`, are
forwarded unchanged on a credentialless route; Cayu neither invents nor rewrites
authentication there.

Before the default upstream opens a socket, Cayu resolves the approved hostname
once, rejects every non-global, loopback, link-local, multicast, reserved, or
unspecified result, and connects to one validated address while preserving the
original HTTP `Host` and TLS SNI names. This pins the authorization decision to
the connection target and prevents DNS rebinding to private networks or cloud
metadata. Application-owned `HttpxUpstream(routes=...)` mappings are an explicit
trusted-control-plane override for private service origins; they still reject
loopback, link-local/metadata, multicast, reserved, and unspecified addresses.

Credentialed and credentialless declarations can be combined in one factory.
The adapter receives the union of their hostnames for enforcement preflight,
while only `VirtualCredentialSpec` entries create guest credential values. A
factory with neither kind is rejected before adapter allocation, so an empty
configuration opens no proxy or network capability.

Credentialless authorization also requires a session-isolated broker transport.
Docker provides this automatically: its dual-homed sidecar listens only on the
random per-session internal network, and authenticates every sidecar-to-broker
connection through a private host-mounted credential file. That transport
credential is never mounted or injected into the container. A host/LAN peer
cannot use the broker listener, and an unrelated container on Docker's shared
bridge cannot use the sidecar.

There is no default runtime path. Every factory must receive an explicit
`adapter` or `runner_kind`; a registry-backed selection must name a registered
adapter. Omitted and unsupported selections fail before grants, proxies,
runners, or workspaces are created. `runner_kind="docker"` selects ordinary
Docker container execution and is intended for trusted development, CI,
conformance, and packaging. It is never an implicit fallback for untrusted code.
Microsandbox is Cayu's primary local runner for untrusted code; if that microVM
runtime is unavailable, setup fails rather than falling back to Docker. E2B and
Microsandbox use registered `SandboxEgressAdapter` implementations that both
prepare the proxy and create the matching network-restricted runner. Remote/raw
proxy exposures have an additional credentialless requirement described below.

### Managed runner and workspace composition

Pass `workspace_factory` when tools need files inside the enforced runner. The
factory receives the lifecycle-managed public `Runner`, not the raw provider
runner, and may return a `Workspace` synchronously or asynchronously. When
`workspace_factory` is set and `inner_binding` is omitted,
`VirtualEgressEnvironmentFactory` uses `NativeBinding`: workspace and command
operations target the same sandbox, and the runtime exposes both through
`ToolContext`. Cayu verifies that a native workspace is bound to the managed
runner it supplied. To attach a durable workspace outside the sandbox instead,
pass an explicit non-native `inner_binding` (for example `SyncBinding`) that
defines how the two resources compose.

Provider-native workspaces request a narrow typed capability from the managed
runner. That capability exposes only native filesystem operations and stable
sandbox identity; it has no `close()` method and cannot bypass environment
finalization. Runner-backed workspaces retain the managed runner privately and
prove their binding by identity without publishing a runner accessor. Finalization
revokes grants first, finalizes the workspace binding while enforcement is
still present, then closes the provider runner, proxy/network, and session CA.
Applications must finalize the environment binding (the normal `CayuApp`
lifecycle) or close the managed runner; they must not retain or close a raw
provider runner.

### Microsandbox

Microsandbox runs locally and exposes the host as
`host.microsandbox.internal`. Its adapter creates a deny-by-default network
policy that allows only DNS to the host gateway and TCP to the per-session Cayu
proxy port. The session CA is copied into the root filesystem before boot.

Microsandbox publishes both IPv4 and IPv6 gateway addresses for that hostname
and rewrites them to the matching host loopback address. Cayu therefore binds
the same ephemeral proxy port on both `127.0.0.1` and `::1`; guest clients work
regardless of which address family they select. Both listeners share one broker
and session CA, and both are closed during teardown. Setup fails closed if
either listener cannot be established.

The default is deliberately loopback-only. It does not expose the proxy port on
the host's LAN interfaces. The adapter exposes no bind-host override; paired
loopback listeners are part of its fixed Microsandbox exposure contract.

The built-in host exposure is shared host infrastructure rather than a
session-exclusive transport. It therefore supports virtual-credential routes
but fails closed for credentialless destinations. A custom Microsandbox
`ProxyExposure` may support them only when its returned `ExposedProxy` truthfully
sets `credentialless_isolated=True` and the endpoint is unreachable from other
sandboxes and untrusted peers.

```python
from cayu import (
    CayuApp,
    EnvironmentSpec,
    HttpEgressPolicy,
    MicrosandboxWorkspace,
    SecretRef,
    StaticVault,
    VirtualCredentialSpec,
    VirtualEgressEnvironmentFactory,
)
from cayu.egress.microsandbox_adapter import MicrosandboxEgressAdapter

app = CayuApp()
vault = StaticVault({"provider_key": "sk_test_..."})
policies = {
    "provider": HttpEgressPolicy(
        name="provider",
        allowed_hosts=["api.example.com"],
        allowed_endpoints=[("GET", "/v1/data")],
    )
}
credentials = [
    VirtualCredentialSpec(
        env_name="PROVIDER_KEY",
        secret=SecretRef(name="provider_key"),
        destination="api.example.com",
        policy_name="provider",
        credential_kind="opaque_bearer",
    )
]

factory = VirtualEgressEnvironmentFactory(
    resolver=vault,
    policies=policies,
    credentials=credentials,
    adapter=MicrosandboxEgressAdapter(),
    image="python:3.13",  # Microsandbox OCI image
    workspace_factory=MicrosandboxWorkspace,
)

app.register_environment_factory(
    EnvironmentSpec(name="microsandbox-egress"),
    factory,
    default=True,
)
```

For every session on `microsandbox-egress`, file tools receive a
`MicrosandboxWorkspace` whose read, write, list, delete, and canonical-path
operations use the same microVM as command tools. The application never
constructs `_EgressManagedRunner`, accesses `_runner`, casts to
`MicrosandboxRunner`, or owns a second cleanup handle.

Install both optional dependencies with
`pip install 'cayu[egress,microsandbox]'`.

### Durable reconnect

`VirtualEgressEnvironmentFactory` returns a versioned, JSON-safe reconnect
envelope for adapters that can securely reattach a sandbox. `CayuApp` writes
that envelope to the session checkpoint and passes it back to a newly
constructed factory on resume or recovery; applications should not persist a
live runner, broker, or adapter object themselves.

For Microsandbox, the durable identity is the sandbox name plus immutable
numeric provider creation timestamp, the original host listener port, the
guest-visible proxy endpoint port, a non-secret ownership id, and the attested
owner session and environment:

```json
{
  "version": 1,
  "runner_kind": "microsandbox",
  "session_id": "sess_...",
  "environment_name": "billing",
  "capability": "supported",
  "identity": {
    "sandbox_name": "cayu-egress-sandbox-...",
    "sandbox_created_at": 1752582896.789,
    "proxy_listener_port": 43127,
    "proxy_endpoint_port": 43127,
    "ownership_id": "9f2d9a6f7a2c4e32a624b37d104ca4f1",
    "owner_session_id": "sess_...",
    "owner_environment_name": "billing"
  }
}
```

The guest endpoint port is part of the sandbox's immutable deny-by-default
network policy, while the listener port identifies the host socket that must be
reclaimed. Recording both permits a session-isolated exposure to map one to the
other without confusing the two during reconnect. A host-local, mode-`0600`
attestation and process-independent lock bind every identity field to the
sandbox name, so concurrent callers cannot evade the single-owner rule by
presenting a different port. The provider creation timestamp prevents a
removed-and-recreated same-name sandbox from being accepted as the original
incarnation. Attested session and environment ownership prevents a valid inner
identity from being rewrapped in an envelope for another runtime scope. The
default attestation
directory is stable across Cayu process restarts; deployments may set
`reconnect_state_dir` to a private host-local runtime directory. Reconnect must
validate the attestation, reclaim that exact listener, and reproduce the
recorded guest endpoint before attaching. A custom exposure whose mapping is
not stable across calls fails closed before the sandbox is attached. It
then creates a fresh credential registry, broker, grants, proxy, and session
CA; attaches the recorded sandbox; replaces the guest CA; and reruns the
positive proxy/TLS and negative direct-egress preflight before any agent
command can run. The old
virtual credential is unknown to the fresh registry and cannot authorize a
request through the new broker. A second concurrent owner receives
`EgressReconnectConflictError` before a proxy is opened or the sandbox is
mutated.

An interrupted Microsandbox session detaches rather than removes its sandbox so
the checkpoint can be resumed. Completed, failed, cancelled, or explicitly
closed sessions remove it. Reconnected teardown keeps the normal ordering:
revoke and drain grants, release the runner, then close the proxy boundary.

The envelope never contains real or virtual credential values, proxy bearer
authority, or CA private material. The factory validates version, runner kind,
session, environment, exact adapter-owned identity allowlists, and common
replayable-authority field-name variants before attaching. Custom adapters that
declare reconnect support are trusted extension code and must reject every
identity field they do not own; the framework's generic scan is defense in
depth, not a substitute for that allowlist. Missing sandboxes and ownership or
port conflicts have typed, actionable errors. Parent checkpoint metadata copied
into a fork is recognized as parent-owned and ignored so the child creates a
new isolated sandbox; other cross-session metadata is rejected.

Reconnect support is explicit by adapter:

| Adapter | Virtual-egress reconnect |
| --- | --- |
| Microsandbox | Supported; attested single-owner sandbox plus host-listener and guest-endpoint ports, fresh grants/broker/CA, full preflight. |
| Lambda MicroVM | Unsupported until a durable external single-owner claim is available; lower-level runner reattach is not sufficient. |
| Docker | Unsupported; raises `UnsupportedEgressReconnectError`. Rebuild explicitly. |
| E2B | Unsupported; raises `UnsupportedEgressReconnectError`. Rebuild explicitly. |

Unsupported adapters never interpret reconnect metadata as a request to create
a replacement sandbox. Their initial factory result contains a versioned
`"capability": "unsupported"` marker and a non-secret reason instead of a fake
runner identity. If that checkpoint later returns on resume, the factory raises
`UnsupportedEgressReconnectError` before preparing an adapter or creating a
runner. Catch it in trusted application code if an explicit
Git/artifact/memory reconstruction flow is appropriate; do not label that
rebuild as a reconnect.

### E2B

E2B sandboxes run in E2B's cloud, so they cannot reach a loopback listener in
the Cayu process. `E2BEgressAdapter` therefore requires a `ProxyExposure` that
opens a raw TCP tunnel or private route to the local CONNECT proxy and returns
an `ExposedProxy`. The advertised endpoint must be a dedicated IPv4 literal; E2B's
hostname-aware filtering inspects the tunneled `CONNECT` destination, so a
hostname allowlist cannot act as a transparent raw proxy relay. The adapter
fails closed on hostname and IPv6 exposures and permits only the IPv4 endpoint.

```python
from cayu import E2BWorkspace
from cayu.egress import EgressAdapterRegistry
from cayu.egress.e2b_adapter import E2BEgressAdapter
from cayu.egress.proxy_exposure import ExposedProxy, ProxyExposure

# Application-owned adapter that exposes the supplied local host/port through
# a raw TCP tunnel and returns an IPv4-literal endpoint such as
# ExposedProxy(proxy_url="http://203.0.113.10:8443"). For credentialless
# destinations the endpoint must be session-exclusive, and the exposure must
# return ExposedProxy(..., credentialless_isolated=True).
tunnel = MyE2BProxyExposure()

registry = EgressAdapterRegistry()
registry.register(E2BEgressAdapter(exposure=tunnel))

factory = VirtualEgressEnvironmentFactory(
    resolver=vault,
    policies=policies,
    credentials=credentials,
    runner_kind="e2b",
    adapter_registry=registry,
    image="base",  # E2B template name or id
    workspace_factory=E2BWorkspace,
)
```

Install with `pip install 'cayu[egress,e2b]'`. A normal HTTP reverse proxy is
not sufficient: the endpoint must preserve raw HTTP CONNECT and tunneled TLS
bytes. The adapter refuses to start without an exposure implementation. For a
credentialless or mixed session, a default/shared/public exposure is also
refused: `credentialless_isolated=True` is an explicit contract that only the
intended sandbox and Cayu's trusted host can reach that endpoint. Setting it on
a shared tunnel does not create isolation; the application-owned exposure must
enforce the claim with a per-session private route or equivalent boundary.

E2B exposes Firecracker MMDS at `169.254.169.254` inside the guest even when
external internet access is denied. The adapter uses the same public
`E2BRunner.create_hardened(...)` contract as offline E2B consumers: before
handoff it installs a root firewall reject for that address, removes the default
user from administrator groups, makes `sudo`/`su` root-only, and installs the
session CA as a protected root-owned file. Before virtual-egress callbacks run,
a fresh guest process must prove the common non-root, metadata, and
protected-file boundary. Virtual-egress preflight then proves the proxy/TLS
route, direct public TLS denial, MMDS GET, and token acquisition before any
configured setup command runs. The preflight retains its configured timeout,
and each setup command retains its 300-second execution limit; the adapter adds
all of those allowances to the bounded handoff deadline.
Cayu repeats the common guest and protected-file verification before publishing
the runner. All later commands and native workspace operations are pinned to
that same verified guest user, regardless of the template's configured default.
Missing hardening tools, retained guest privilege, a writable CA, or any
preflight failure closes the sandbox. This security mode intentionally removes
guest privilege escalation; bake other privileged setup into the E2B template
or use the typed protected-file/directory bootstrap contract rather than
relying on `setup_commands` plus sudo.

Both adapters run a per-session preflight before returning the environment. It
must reach the proxy, complete TLS using the session CA, fail a raw public-IP
socket, and fail a cloud-metadata socket. Any failure closes the sandbox,
revokes grants, tears down the proxy/exposure, and aborts environment creation.
Proxy environment variables alone are not a security boundary: without the
runtime-native deny policy, a process can ignore them and open a direct socket.

For a session with multiple grants, the positive proxy/TLS check samples the
first configured destination; it does not connect to every provider during
boot. Proxy reachability and CA trust are session-wide. The runtime-native
deny-all policy permits only the proxy endpoint, while raw public-IP and metadata
probes verify the general direct-egress boundary without relying on guest DNS.
Provider-specific authorization is still enforced for every request by its
grant and `EgressPolicy`.

## Audit events

With `event_emitter` wired, the session event stream carries secret-free
telemetry events (payloads never contain the real value): `credential.mode.selected`
and `egress.grant.minted` at start, `egress.request.authorized` /
`egress.request.denied` per outbound request, and `egress.grant.revoked` at
teardown. Request events include `authorization_kind`, which is
`virtual_credential` or `credentialless`; credentialless records have a null
`grant_id`. They are emitted best-effort from the proxy path and can lag
or reorder around terminal session events; the enforcement decision itself is
still synchronous inside the broker. Payloads carry grant id, destination,
method, path, policy name, and decision/status.

## Scope: what virtual egress does and does not cover

Virtual egress governs the **runner workload credential** — the value the
workload can read from its env/files/`/proc`. Two adjacent things are
deliberately *out of scope*:

- **MCP server secrets.** `McpServerSpec.secret_env`/`secret_headers` are
  resolved **host-side** — injected into the MCP *server* subprocess or the host
  HTTP client that talks to a remote MCP server — and are **never** placed in the
  runner workload. They are the `trusted_tool` (host-side) boundary, so they do
  not weaken the workload non-possession guarantee and are not gated by
  `credential_mode`.
- **The broker proxy listener.** `DockerEgressAdapter` binds the in-process proxy
  to the narrowest interface the sidecar can still reach — loopback on Docker
  Desktop, the docker bridge gateway on Linux — falling back to `0.0.0.0` (with a
  loud warning) only if neither can be determined. Pass `proxy_host=` to override.
  Every connection must first complete a sidecar-only authenticated outer
  CONNECT, so the listener is not a credentialless generic proxy even when it is
  host- or LAN-reachable. The sidecar's guest-facing listener binds only its
  random per-session internal-network address. The transport credential is
  mounted into that sidecar alone and never enters guest env, files, events,
  evidence, artifacts, or persisted binding metadata.

## Credential modes on runners

`LocalRunner`/`DockerRunner` take `credential_mode` (default
`raw_env`) and `allow_raw_secret_env` (default `True`, backward-compatible). This
runner-level mode gates raw injection only: raw `secret_env` is refused when
`credential_mode` is any non-agent-readable mode (`trusted_tool` or
`virtual_egress`) or when `allow_raw_secret_env=False`. It does not by itself prove
network enforcement. Full virtual-egress enforcement is provided by
`VirtualEgressEnvironmentFactory` plus a registered `SandboxEgressAdapter`.

## Example: FastAPI + Stripe

The FastAPI app and agent see ordinary Stripe configuration. The app makes a
normal HTTPS request to Stripe with the key from `STRIPE_SECRET_KEY`:

```
STRIPE_SECRET_KEY=sk_test_cayu_vc_7f3b4a...

stripe_request = urllib.request.Request(
    "https://api.stripe.com/v1/customers",
    data=urllib.parse.urlencode({"email": user.email}).encode(),
    headers={
        "Authorization": "Bearer " + os.environ["STRIPE_SECRET_KEY"],
        "Content-Type": "application/x-www-form-urlencoded",
    },
    method="POST",
)
urllib.request.urlopen(stripe_request, timeout=25)
```

The app makes a normal HTTPS call to `https://api.stripe.com`. Because the
Docker container is on the enforced egress network with the session CA trusted,
Python's standard HTTP client follows the proxy/CA environment automatically.
The broker captures the call, authorizes it, swaps in the real key, and returns
Stripe's response. The agent can print `STRIPE_SECRET_KEY`, grep the filesystem,
and read `/proc` — it only ever finds the virtual value.

Run the end-to-end example:

```
python examples/fastapi_stripe_virtual_egress.py
```

It starts the FastAPI service inside the explicitly selected Docker container,
calls `/customers`, shows the real key was injected only upstream, verifies
direct egress is blocked, and finalizes the factory-managed environment.

### SDK caveat — env-transparent vs. pinning clients

How much (if any) SDK config is needed depends on whether the client honors the
standard env knobs:

- **Env-transparent clients — zero app-specific config.** curl and Python
  `requests`/`urllib` honor `HTTPS_PROXY` and the mounted CA
  (`SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE`), so they work unchanged with this
  adapter.
- **Pinned or proxy-ignoring clients need adapter work.** If a provider SDK ships
  its own trust store or ignores standard proxy env, it should not be the
  headline demo for this adapter. Without a transparent/native egress hook, such
  clients fail closed rather than bypassing the broker.

A test-mode-only guard is on by default: the broker refuses to inject a real
`stripe_bearer` key that is not a test key (`sk_test_`/`rk_test_`); pass
`require_test_mode_credentials=False` to allow live keys.

Virtual egress does not attempt to infer provider business semantics from
request bodies or opaque provider ids. A provider-owned plan or price id is just
provider-owned state unless the broker calls back into that provider with a real
credential. Use provider-scoped credentials, application authorization, or a
custom `EgressPolicy` when you need business-level limits such as spend caps.

## Runner support

| Runner | Status |
| --- | --- |
| `docker` | Egress enforced (per-session internal network + sidecar-only broker authentication + TLS MITM), including credentialless routes; reconnect unsupported. Container isolation is not a secure sandbox boundary. |
| `microsandbox` | Virtual credentials are enforced with a deny-by-default host policy allowing only the Cayu proxy port. Credentialless routes require a custom session-isolated exposure; reconnect supported. |
| `e2b` | Enforced with a dedicated E2B-reachable, IPv4-literal raw TCP proxy exposure and fail-closed preflight. Credentialless routes additionally require `credentialless_isolated=True`; reconnect unsupported. |
| `lambda-microvm` | Enforced in the integrated image: a VPC connector limits destinations, while a dedicated agent network namespace has no default route and can reach only a narrow relay to the Cayu proxy. Credentialless routes require a session-isolated exposure; virtual-egress factory reconnect unsupported. |
| `local` | Unsupported by the virtual-egress factory. Direct runner construction may still set `credential_mode` for raw-secret checks, but that is not an egress boundary. |

Notes on the Docker adapter:
- The broker proxy binds a host-reachable interface so the sidecar can reach it via
  `host.docker.internal` on both Docker Desktop and native Linux. A private
  sidecar-only outer CONNECT authenticates that hop independently of provider or
  credentialless request authorization.
- The sidecar discovers the interface without the default route and binds its
  guest-facing listener only there. Other containers on Docker's shared default
  bridge cannot turn the sidecar into a credentialless proxy.
- A custom `sidecar_image=` must provide `/bin/sh`, `ip`, `awk`, `sleep`, and a
  `socat` build with `PROXY` plus `proxyauthfile` support, as the default
  `alpine/socat` image does. Startup checks that the authenticated listener
  reached its ready state and fails closed when the image is incompatible.
- `setup_commands` run on the enforced network with the egress overlay applied, so they are
  subject to the **same egress policy** as the app. Bake tools that need arbitrary hosts into the
  image rather than installing them from `setup_commands` under `virtual_egress`.

The Lambda MicroVM adapter uses `VpcTaskProxyExposure` to advertise the trusted
Fargate task's private IPv4 address. Startup proves the proxy and session CA work
and that direct public-IP connections fail. Its default
`metadata_isolation="required"` mode also probes the link-local metadata path. A
reachable path raises `UnsupportedEgressCapabilityError` with capability
`metadata_isolation` after trusted setup commands and before agent execution.
Running the probe last ensures setup cannot invalidate the boundary after it
was verified. Its structured remediation identifies the enforceable-topology
and explicit-unverified paths.
A successful required probe produces a typed claim in the versioned
`cayu.egress_capabilities.v1` projection stored in factory environment and
result metadata. Configured mode is stored separately under
`egress_configuration`; it is never treated as observed proof. Adapters without
runtime claims emit an explicit unclaimed envelope rather than an ambiguous
empty dictionary. Version 1 proof sources, observations, reasons, and
remediations use documented closed vocabularies so configuration tokens and
secret-shaped values cannot be smuggled into evidence fields.
The closed vocabularies are defined by `EgressCapabilityClaim`; capability and
adapter identity remain bounded extensible tokens that reject common
secret-shaped forms. An envelope may carry up to 64 claims, and each claim may
carry up to 16 uniquely named adapter-specific boolean or JSON-safe integer
facts. Arbitrary string detail values and free-form text are not evidence.

The integrated image leaves the root sidecar in AWS's managed guest network but
runs every ordinary command in a dedicated `cayu-agent` network namespace. That
namespace has only a point-to-point veth and no default route, so link-local
metadata, the guest's managed network, and public destinations are unreachable.
A root-owned TCP relay bound to the veth gateway forwards only to the enforced
private Cayu proxy, and an interface-scoped INPUT rule rejects attempts to reach
the sidecar's port 8080 through that gateway. Agent commands also run as UID
1000 through `setpriv`, with no effective, inheritable, ambient, or bounding
capabilities. Sidecar replies and authenticated lifecycle commands remain in
the trusted root profile, so managed ingress and EFS/S3 Files mounts continue
to work. The explicit `metadata_isolation="unverified"` opt-out remains for
custom or legacy images without that process boundary; it skips the metadata
probe, records an `unverified` claim with bounded reason/remediation codes, and
can never produce a verified claim. Execution-role scope remains defense in
depth, not a substitute for the network proof.

Interrupt finalization revokes the grant, syncs/unmounts the workspace, then
suspends the MicroVM. Terminal outcomes terminate it. Durable reconnect metadata
includes the owning session id for trusted application-specific recovery. The
lower-level runner adapter can
reattach a known MicroVM for trusted application-specific flows, but the
virtual-egress factory reports reconnect as unsupported because Cayu does not
yet have a durable external claim that prevents two orchestrators from
simultaneously reconfiguring the same MicroVM. Applications must choose an
explicit rebuild or trusted recovery flow instead of treating that attach as a
secure virtual-egress reconnect.

`examples/aws/lambda_microvm_agent` composes this adapter with an EFS access
point (default), an opt-in S3 Files access point, S3 artifacts, Secrets Manager,
a private Fargate receiver, and an exact host/method/path egress policy.

## Verification

`tests/egress/` proves non-possession rather than redaction:

- Shared deterministic conformance: one registration contract describes the
  Docker, E2B, and Microsandbox adapter factories, runner kinds, bounded
  destinations, live prerequisites, capability proof classes, teardown bounds,
  and proof sources. It proves fail-closed resolution, runner/binding pairing,
  grant scoping, cancellation-safe retryable cleanup, revocation-before-release,
  and that seeded downgrade/cleanup/pairing/revocation defects are detected.
- Unit tests (no Docker): modes, registry, HTTP policy deny-before-resolve,
  broker pipeline (no secret in any event/response/exception), fail-closed
  adapters, and the in-process TLS interception + credential-swap harness.
- Adversarial E2E (`tests/egress/test_docker_egress_e2e.py`, gated on a running
  Docker daemon): real containers assert that `env`/`/proc/self/environ` show only
  the virtual value, a secret-blind recursive Bloom scan finds no real secret,
  the allowed call succeeds through the session CA with the real key swapped
  only upstream, direct public-IP and metadata egress are blocked, and the
  credential is rejected after the session closes. Its credentialless lane also
  proves that a direct host-broker client lacks the sidecar credential and an
  unrelated default-bridge container cannot reach the sidecar listener. This runs in the existing
  uncredentialed Docker live CI job and as `docker-live-virtual-egress` in the
  nightly registry.
- Microsandbox E2E (manual/nightly): set
  `CAYU_RUN_MICROSANDBOX_EGRESS_E2E=1`, then run
  `uv run python scripts/nightly_verification.py --check microsandbox-live-virtual-egress --strict`.
- E2B E2E (manual/nightly): set `E2B_API_KEY`,
  `CAYU_E2B_PROXY_EXPOSURE_COMMAND` (a raw TCP tunnel command template using
  `{host}` and `{port}`), `CAYU_E2B_PROXY_URL`, and
  `CAYU_RUN_E2B_EGRESS_E2E=1`, then run
  `uv run python scripts/nightly_verification.py --check e2b-live-virtual-egress --strict`.
- Lambda MicroVM metadata boundary (manual/nightly): deploy the integrated AWS
  example, set `CAYU_AWS_METADATA_ISOLATION_LIVE=1`,
  `CAYU_AWS_METADATA_ISOLATION_STACK`, and an AWS region, then run
  `uv run python scripts/nightly_verification.py --check aws-lambda-microvm-metadata-isolation-live`.
  The check verifies required-mode proxy success, public-egress and metadata
  denial, process/filesystem/credential inspection, vault-canary and
  AWS-credential non-possession, revocation, credential-free workspace
  sync/release, and cleanup. Its structured adapter evidence must report
  the exact versioned `cayu.aws_lambda_microvm_metadata_isolation.v1` schema,
  including the deployed execution role, UID/GID and capability boundary,
  `no_new_privs`, route-less namespace, and sidecar-port denial. The guest
  returns keyed candidate fingerprints; only the trusted control task compares
  them with vault/server/database values.

The deterministic adapter, runner, preflight-failure, and teardown tests run in
normal CI. Live results emit typed bounded evidence with adapter, scenario,
status, proof source, safe reason/observation tokens, bounded duration, and
cleanup outcome; the records never contain secrets, headers, provider response
bodies, or command output. E2B and
Microsandbox E2Es are intentionally opt-in because they need
cloud credentials, virtualization, or external tunnel infrastructure. The
nightly registry reports them as `skipped` when an opt-in flag, runtime, API key,
or E2B tunnel input is absent; a successful real-runtime run reports `verified`.

The complete managed close path—grant drain, runner stop, adapter binding
release, and audit drain—is cancellation-safe and bounded by
`teardown_timeout_s` (15 seconds by default). Grant scope is validated before
resources are allocated, and revocation completes before adapter resources are
released. A timeout or resource-release failure leaves the binding open and
reports a bounded, secret-free error; calling `close()` again resumes or retries
the same cleanup tasks. Only a completed teardown is marked closed. Prepare and
factory rollback use the same deadline and annotate the original failure with
safe cleanup-phase/type evidence if rollback remains incomplete.
