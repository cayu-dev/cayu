# Virtual Egress Credentials

A secure sandbox credential path. A sandboxed app (for example a FastAPI app using
Stripe) is configured with a **virtual credential** that looks normal to the
app, while a trusted **egress broker outside the sandbox** swaps in the real
vault secret on the way to the provider. The sandbox never receives the real
secret — not in env, files, `/proc`, logs, or on the wire.

## Why not just `secret_env`?

`secret_env` (mode `raw_env`) resolves a vault secret and injects the **raw
value** into the sandbox process environment. Redaction scrubs logs, but the
secret is still *present* in the sandbox and the agent can read it. That is a
convenience mode, not a security boundary.

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
- Sandbox env → virtual credential only.
- Sandbox network → in an enforced adapter, cannot reach the credentialed
  provider except through the broker.
- **Fail closed:** the virtual-egress environment factory refuses any runner
  family without a registered enforcing adapter (`UnsupportedEgressError`)
  rather than downgrading to raw injection.

## How it works

```
Sandbox app                       Host (trusted)
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

Direct egress is blocked by construction: the sandbox joins an `--internal`
Docker network with no route to the internet, so the only reachable egress is a
dual-homed sidecar that forwards to the in-process broker.

## API shape

Most apps use the root-level setup API: `CredentialMode`, `HttpEgressPolicy`,
`VirtualCredentialSpec`, and `VirtualEgressEnvironmentFactory`.

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
environment factory; per session it mints grants, stands up the broker + enforced
Docker sandbox, and tears everything down at session end (the workspace binding's
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
    image="cayu-egress-fastapi-stripe:demo",     # FastAPI + Python HTTP client
    event_emitter=app.scoped_event_emitter(
        event_types=VIRTUAL_EGRESS_EVENT_TYPES,
    ),                                          # stream only virtual-egress audit events
)
app.register_environment_factory(EnvironmentSpec(name="billing"), factory, default=True)
```

Sessions on that environment run in the enforced sandbox with `STRIPE_SECRET_KEY`
set to the virtual credential; the real key is swapped in only by the broker.
Grant revocation is enforced against in-flight broker requests: teardown marks
the grant revoked, waits for active request leases to drain, and the broker
re-checks liveness after vault resolution before forwarding upstream.

Docker is the default runtime path. E2B and Microsandbox use registered
`SandboxEgressAdapter` implementations that both prepare the proxy and create
the matching network-restricted runner. If no adapter is registered for the
requested runner kind, setup fails closed.

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

```python
from cayu.egress import EgressAdapterRegistry
from cayu.egress.microsandbox_adapter import MicrosandboxEgressAdapter

registry = EgressAdapterRegistry()
registry.register(MicrosandboxEgressAdapter())

factory = VirtualEgressEnvironmentFactory(
    resolver=vault,
    policies=policies,
    credentials=credentials,
    runner_kind="microsandbox",
    adapter_registry=registry,
    image="python:3.13",  # Microsandbox OCI image
)
```

Install both optional dependencies with
`pip install 'cayu[egress,microsandbox]'`.

### E2B

E2B sandboxes run in E2B's cloud, so they cannot reach a loopback listener in
the Cayu process. `E2BEgressAdapter` therefore requires a `ProxyExposure` that
opens a raw TCP tunnel or private route to the local CONNECT proxy and returns
an `ExposedProxy`. The advertised endpoint must be a dedicated IPv4 literal; E2B's
hostname-aware filtering inspects the tunneled `CONNECT` destination, so a
hostname allowlist cannot act as a transparent raw proxy relay. The adapter
fails closed on hostname and IPv6 exposures and permits only the IPv4 endpoint.

```python
from cayu.egress import EgressAdapterRegistry
from cayu.egress.e2b_adapter import E2BEgressAdapter
from cayu.egress.proxy_exposure import ExposedProxy, ProxyExposure

# Application-owned adapter that exposes the supplied local host/port through
# a raw TCP tunnel and returns an IPv4-literal endpoint such as
# ExposedProxy(proxy_url="http://203.0.113.10:8443").
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
)
```

Install with `pip install 'cayu[egress,e2b]'`. A normal HTTP reverse proxy is
not sufficient: the endpoint must preserve raw HTTP CONNECT and tunneled TLS
bytes. The adapter refuses to start without an exposure implementation.

E2B exposes Firecracker MMDS at `169.254.169.254` inside the guest even when
external internet access is denied. Before handoff, the adapter installs a root
firewall reject for that address, removes the default user from the sudo group,
and makes `sudo`/`su` root-only. A fresh guest process must prove it cannot
remove the rule, and preflight must prove MMDS GET and token acquisition both
fail. All later commands are pinned to that same verified guest user, regardless
of the template's configured default. Missing hardening tools or retained guest
privilege fail closed. This security mode intentionally removes guest privilege
escalation; bake privileged setup into the E2B template rather than relying on
`setup_commands` plus sudo.

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
teardown. Request events are emitted best-effort from the proxy path and can lag
or reorder around terminal session events; the enforcement decision itself is
still synchronous inside the broker. Payloads carry grant id, destination,
method, path, policy name, and decision/status.

## Scope: what virtual egress does and does not cover

Virtual egress governs the **sandbox process credential** — the value the
sandboxed app can read from its env/files/`/proc`. Two adjacent things are
deliberately *out of scope*:

- **MCP server secrets.** `McpServerSpec.secret_env`/`secret_headers` are
  resolved **host-side** — injected into the MCP *server* subprocess or the host
  HTTP client that talks to a remote MCP server — and are **never** placed in the
  sandbox container. They are the `trusted_tool` (host-side) boundary, so they do
  not weaken the sandbox non-possession guarantee and are not gated by
  `credential_mode`.
- **The broker proxy listener.** `DockerEgressAdapter` binds the in-process proxy
  to the narrowest interface the sidecar can still reach — loopback on Docker
  Desktop, the docker bridge gateway on Linux — falling back to `0.0.0.0` (with a
  loud warning) only if neither can be determined. Pass `proxy_host=` to override.
  The listener is credential-gated (an unguessable virtual credential + policy),
  so it never exposes a real secret regardless of the bind address.

## Credential modes on runners

`LocalRunner`/`DockerRunner`/`SbxRunner` take `credential_mode` (default
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
sandbox is on the enforced egress network with the session CA trusted, Python's
standard HTTP client follows the proxy/CA environment automatically. The broker
captures the call, authorizes it, swaps in the real key, and returns Stripe's
response. The agent can print `STRIPE_SECRET_KEY`, grep the filesystem, and read
`/proc` — it only ever finds the virtual value.

Run the end-to-end example:

```
python examples/fastapi_stripe_virtual_egress.py
```

It starts the FastAPI service inside the enforced sandbox, calls `/customers`,
shows the real key was injected only upstream, verifies direct egress is blocked,
and finalizes the factory-managed environment.

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
| `docker` | Enforced (internal network + sidecar + TLS MITM). |
| `microsandbox` | Enforced with a deny-by-default host policy allowing only the Cayu proxy port. |
| `e2b` | Enforced with a dedicated E2B-reachable, IPv4-literal raw TCP proxy exposure and fail-closed preflight. |
| `local`, `sbx` | Unsupported by the virtual-egress factory. Direct runner construction may still set `credential_mode` for raw-secret checks, but that is not an egress boundary. |

Notes on the Docker adapter:
- The broker proxy binds a host-reachable interface so the sidecar can reach it via
  `host.docker.internal` on both Docker Desktop and native Linux; the unguessable virtual
  credential + destination/policy checks are the trust boundary, not the bind address.
- `setup_commands` run on the enforced network with the egress overlay applied, so they are
  subject to the **same egress policy** as the app. Bake tools that need arbitrary hosts into the
  image rather than installing them from `setup_commands` under `virtual_egress`.

## Verification

`tests/egress/` proves non-possession rather than redaction:

- Unit tests (no Docker): modes, registry, HTTP policy deny-before-resolve,
  broker pipeline (no secret in any event/response/exception), fail-closed
  adapters, and the in-process TLS interception + credential-swap harness.
- Adversarial E2E (`tests/egress/test_docker_egress_e2e.py`, gated on a running
  Docker daemon): real containers assert that `env`/`/proc/self/environ` show only
  the virtual value, a recursive search finds no real secret, the allowed call
  succeeds with the real key swapped upstream, direct egress is blocked, and the
  credential is rejected after the session closes.
- Microsandbox E2E (manual/nightly): set
  `CAYU_RUN_MICROSANDBOX_EGRESS_E2E=1`, then run
  `uv run python scripts/nightly_verification.py --check microsandbox-live-virtual-egress --strict`.
- E2B E2E (manual/nightly): set `E2B_API_KEY`,
  `CAYU_E2B_PROXY_EXPOSURE_COMMAND` (a raw TCP tunnel command template using
  `{host}` and `{port}`), `CAYU_E2B_PROXY_URL`, and
  `CAYU_RUN_E2B_EGRESS_E2E=1`, then run
  `uv run python scripts/nightly_verification.py --check e2b-live-virtual-egress --strict`.

The deterministic adapter, runner, preflight-failure, and teardown tests run in
normal CI. E2B and Microsandbox E2Es are intentionally opt-in because they need
cloud credentials, virtualization, or external tunnel infrastructure. The
nightly registry reports them as `skipped` when an opt-in flag, runtime, API key,
or E2B tunnel input is absent; a successful real-runtime run reports `verified`.
