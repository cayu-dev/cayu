# Project server process

`cayu serve` boots the application factory declared in the nearest Cayu
project. The command follows the
[application construction contract](../src/cayu/guides/application-anatomy.md):
one synchronous, zero-argument factory call creates one process-scoped
`CayuApp`. Durable stores coordinate separate processes.

## Resolve a foreground child's pending action

Query `GET /api/pending-actions?session_id=<parent>&kind=delegated_action` to
discover the exact child action holding a foreground parent. The bounded
`delegated_action` reference contains `child_session_id`, `action_kind`,
`action_id`, and `status`; it contains no child question, arguments, or answer.
The dashboard's **Open child** link leads to the action's owning session.
For a nested foreground chain, `action_kind="delegated_action"` means the
immediate child is also waiting on a child. Follow its pending-action reference
until reaching the approval/input owner; ancestor references cannot resolve it.

Inspect the child, then use `POST /api/tool-approvals/resolve` or
`POST /api/user-input/resolve` with the child's identifiers and the normal
resolution request. The parent reference is discovery, not bearer authority:
the same authentication, policy, actor, and exact-action checks apply as for a
direct child resolution. There is no independently resolvable parent copy.
After the child eventually finishes, durable delivery continues the original
parent call automatically; do not submit unrelated input to wake the parent.
If the child asks again, `session.delegated_action.updated` refreshes parent
discovery without ending another parent run or interaction.

## Serve the control plane

For trusted local development, open access requires an explicit opt-in:

```bash
cayu serve --dev --host 127.0.0.1 --port 8000
```

`--dev` uses `ServerConfig.local_development()`. With or without `--dev`,
`cayu serve` turns on [request timing](server-configuration.md#request-timing),
so `cayu diagnostics requests` can report per-route cost from the running
process. Without `--dev`, the command fails closed unless an authentication
target is configured:

```toml
[tool.cayu]
factory = "app:build_app"

[tool.cayu.serve]
auth = "server_auth:AUTH"
```

The target is loaded inside the project import context, before the
application factory runs, and must be a callable accepted by
`ServerConfig.protected()`: it takes the request and returns an `AuthContext`
(or a compatible mapping), or raises `HTTPException(401)`.

### Operator credentials from the environment

Cayu ships a ready-made target that reads HTTP Basic credentials from
`CAYU_OPERATOR_USERNAME` and `CAYU_OPERATOR_PASSWORD`:

```toml
[tool.cayu.serve]
auth = "cayu.server.environment_auth:OPERATOR_BASIC_AUTH"
```

`cayu new` writes this for every preset served by `[tool.cayu.serve]` (the
`agent` and `coding` presets), and `cayu cloud init` adds it to a project that
has no auth target yet. The module ships in cayu releases newer than 0.8.1; for a
project whose `cayu` requirement still allows 0.8.1 or older, `cayu cloud init`
writes a `server_auth.py` that builds `BasicAuth` from the same variables instead
(see [Cayu Cloud](cayu-cloud.md#initialize-an-agent-project)). The variables are read once, when `cayu serve` loads the
target. If either is unset, empty, or whitespace-only, the command exits before
it builds the application or binds the port:

```text
error: Basic authentication is not configured: CAYU_OPERATOR_PASSWORD is unset or
empty. Set both CAYU_OPERATOR_USERNAME and CAYU_OPERATOR_PASSWORD before starting
the server; it does not fall back to open access. ...
```

There is no fallback to open access. `cayu serve --dev` still runs without
credentials on a loopback host. Importing `cayu.server` does not read these
variables; only loading this target does.

On Cayu Cloud, both variables are provided to each Agent as secrets (an Agent
environment variable with the same name overrides them). Read them with
`cayu cloud service credentials --application APP`.

The target is built from `BasicAuth.from_environment()`, which you can call
yourself to read other variables or to configure an embedded server:

```python
# server_auth.py
from cayu.server import BasicAuth

AUTH = BasicAuth.from_environment(
    "ADMIN_USERNAME",
    "ADMIN_PASSWORD",
    realm="Support tools",
)
```

`from_environment` delegates to the `BasicAuth` constructor, so requests are
checked the same way, with constant-time comparison. A missing or invalid
variable raises `AuthConfigurationError` (a `ValueError`) that names the
variable and never includes its value. Pass `environ=` to read from a mapping
instead of `os.environ`.

### Use your own authentication

`[tool.cayu.serve].auth` accepts any request-to-`AuthContext` dependency, so
operators can sign in with your organization's identity provider (OIDC/JWT from
Cognito, Auth0, Okta, Workday, or an email and password check) instead of a
shared Basic credential. The sketch below is illustrative: `verify_jwt` stands
for a JWT library of your choice, and Cayu does not ship one.

```python
# operator_auth.py
import os

from fastapi import HTTPException, Request

from cayu.server import AuthContext

ISSUER = os.environ["OIDC_ISSUER"]  # e.g. https://example.okta.com/oauth2/default
AUDIENCE = os.environ["OIDC_AUDIENCE"]
JWKS_URL = f"{ISSUER}/v1/keys"  # the provider's published signing keys


def AUTH(request: Request) -> AuthContext:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "Missing bearer token.", {"WWW-Authenticate": "Bearer"})
    try:
        # Verify the signature against the JWKS, then the expiry, issuer, and audience.
        claims = verify_jwt(token, jwks_url=JWKS_URL, issuer=ISSUER, audience=AUDIENCE)
    except Exception:
        raise HTTPException(401, "Invalid bearer token.", {"WWW-Authenticate": "Bearer"})
    return AuthContext(
        subject=claims["sub"],
        tenant=claims.get("org_id"),
        claims={"email": claims.get("email")},
    )
```

```toml
[tool.cayu.serve]
auth = "operator_auth:AUTH"
```

This guards operator access to the Agent's own control plane: the sessions
API, the dashboard, and `/cayu/` on Cayu Cloud. It is not end-user product
login. Customer identity for a product belongs in the service preset's
`AuthenticatedProductAccess` (see
[Serve a maintained public-agent service](#serve-a-maintained-public-agent-service)).
`AuthContext.tenant` is actor provenance only; it does not filter Cayu data.
When a custom target is configured, the `CAYU_OPERATOR_*` variables Cayu Cloud
provides are simply unused, and `cayu cloud init` leaves the target alone.

`--auth module:attribute` overrides the configured target. `--dev` and
`--auth` are mutually exclusive; explicit `--dev` selects the open local
profile even when the project also has deployment authentication configured.
Host and port always reach uvicorn as concrete values.

The command also assembles project identity, release identity, and durable
Evals storage before it constructs the server. It reads `[project].name`,
`CAYU_RELEASE_ID`, and the same `CAYU_DATABASE_URL` or
`[tool.cayu.session_store]` declaration used by session tooling. Development
may create the project-local `data/cayu.db` default; production without an
explicit or already-discovered durable store keeps Evals storage gated rather
than guessing. After the application factory returns, Cayu publishes one
bounded executable eval target per registered agent. An optional
`[tool.cayu.evals.default_judge]` declaration can also publish one exact,
tool-free model judge; it must name an already-registered provider, privacy
policy, same-model decision, and bounded time/token ceilings. An explicit
`[tool.cayu.evals].price_book = "bundled-public"` selection additionally makes
the packaged public-rate snapshot available for generated candidate budgets
and an optional judge cost threshold. Run `cayu guide evals-first` and `cayu
guide evals-ai-quality` for the operator workflows.

Incomplete-session startup recovery is off by default. Opt in with both a
bounded status set and an inactivity fence:

```toml
[tool.cayu.serve]
auth = "server_auth:AUTH"
startup_recovery_statuses = ["pending", "running", "interrupting"]
recovery_inactive_after_seconds = 900
```

Only the recovery statuses supported by `IncompleteSessionsRecoveryRequest`
are accepted. An inactivity threshold without statuses is rejected. The
server's existing persisted-event and interruption-cascade lifecycle remains
part of `create_server`; this option controls only the explicit incomplete
session sweep.

`cayu serve` reports project discovery, factory, auth-target, optional
dependency, server construction, port binding, startup, and termination
failures through its process exit. The command boots one uvicorn process with
one in-memory application object. Autoreload and multi-process workers are v1
non-goals because they require an importable server factory and separate
process-lifecycle decisions. Project-side `SystemExit` during import or
construction is reported as a labeled startup error; Uvicorn's own intentional
process exit keeps its status.

## Serve a maintained public-agent service

Projects generated with `cayu new NAME --preset service` also declare:

```toml
[tool.cayu]
factory = "app:build_app"
service_factory = "service:build_service"
```

For these projects, `cayu serve` loads the service factory with an explicit
`development` or `production` mode and serves its assembled FastAPI product app
on the same listener as the separately mounted `/cayu/` operator control plane.
The service factory, not `[tool.cayu.serve].auth`, owns the distinct customer
and operator policies. Supplying both configurations is rejected rather than
silently choosing one.

Current generated factories also accept the optional, framework-owned
`project_context` keyword and pass it to `create_agent_service(...)`. Older
factories still start, but cannot receive automatic Evals project assembly.
`cayu check --json` reports that exact migration state; use
`cayu generate service-context --dry-run` and then
`cayu generate service-context` for an unmodified generated factory.

`cayu serve --dev` remains loopback-only and selects the generated development
adapters. Without `--dev`, serving refuses to start if the service manifest
reports development or placeholder product access, open or placeholder operator
access, a development-only product identity store, a non-durable runtime session
store, or a missing or non-durable runtime task store. Run
`cayu check --deploy --fail-on warning --json` for the stable diagnostic codes.
Passing an arbitrary ASGI object from `service_factory` is rejected as
unverified; Cayu does not scan host route source or claim authorization it
cannot observe.

Maintained product creation accepts at most 1 MiB of encoded JSON and rejects
duplicate object keys before FastAPI validation. Product responses are marked
`Cache-Control: private, no-store`.

The built-in listener uses HTTP. A production public service must run behind a
trusted TLS-terminating ingress or reverse proxy with the backend listener
restricted to that trusted network. Expose only HTTPS to customers and
operators; neither bearer policy is safe over a directly exposed HTTP
connection.

## Migrate storage before starting a release

A production server validates its Postgres schema at startup and never
migrates it. Deploy each release in this order:

1. Run `cayu storage migrate` once, as a one-off task from the release image,
   with the service's exact environment and secrets. Without `--postgres`, the
   command reads the same `CAYU_DATABASE_URL` or `[tool.cayu.session_store]`
   declaration as `cayu serve`, so the connection string stays out of the task
   command line. Set `CAYU_DATABASE_DIRECT_URL` as well when
   `CAYU_DATABASE_URL` goes through a transaction-pooling proxy.
2. Start `cayu serve` and any `cayu worker` processes only after that task
   exits `0`.

`cayu storage status --json` tells the deploy step beforehand whether the
database is up to date (exit `0`), needs a forward migration (exit `3`), or
cannot be migrated by this build with the given inputs (exit `4`). See
[Deployment migration step](session-store-targets.md#deployment-migration-step)
for the status fields, provider-managed backup references
(`--backup-managed`), and the empty-database rule.
