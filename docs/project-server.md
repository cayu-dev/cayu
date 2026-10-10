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

`[tool.cayu.serve].auth` accepts any request-to-`AuthContext` dependency.
For bearer tokens from your organization's identity provider, use
[`OidcBearerAuth`](#verify-oidc-bearer-tokens). For another authentication
system, expose a callable that verifies the request and returns `AuthContext`,
then set `auth = "operator_auth:AUTH"` to its module and attribute.

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

### Verify OIDC bearer tokens

`OidcBearerAuth` checks JWT bearer tokens issued by an OpenID Connect provider
such as Amazon Cognito, Auth0, Okta, Microsoft Entra ID, Google, or Workday.
Install it with `pip install "cayu[oidc]"`, which adds PyJWT to the server
extra. Build the target from the environment:

```python
# operator_auth.py
from cayu.server import OidcBearerAuth

AUTH = OidcBearerAuth.from_environment(required_scopes=["cayu:operate"])
```

```toml
[tool.cayu.serve]
auth = "operator_auth:AUTH"
```

`from_environment()` reads the issuer URL from `CAYU_OIDC_ISSUER` and the
expected audience from `CAYU_OIDC_AUDIENCE` (a comma-separated list is
accepted) when `cayu serve` loads the target. If either is unset or empty, or
the issuer isn't a usable HTTPS URL, it raises `AuthConfigurationError` (a
`ValueError`) that names the variable and never includes its value, and the
command exits before it builds the application; it does not fall back to open
access. `cayu serve --dev` doesn't load the target, so local development needs
neither variable. Pass other variable names as the first two arguments, and any
constructor option as a keyword argument.

Each request must send `Authorization: Bearer <JWT>`. The verifier:

- reads `{issuer}/.well-known/openid-configuration`, requires its `issuer` to
  match exactly, and loads signing keys from its `jwks_uri` (pass `jwks_url=`
  to skip discovery);
- accepts only asymmetric signatures (`RS*`, `PS*`, `ES*`, `EdDSA`) made by the
  JWKS key named by the token's `kid`, never `none` or `HS*`;
- requires `iss`, `aud`, and `exp`, checks `nbf` and `iat` when present, and
  allows 60 seconds of clock skew (`leeway_seconds=`);
- maps `sub` (or `subject_claim=`) to `AuthContext.subject`. When
  `tenant_claim=` is set, that claim is required and becomes
  `AuthContext.tenant`, which is actor provenance only;
- can require claims or exact claim values (`required_claims=`) and scopes from
  `scope` or `scp` (`required_scopes=`; a missing scope gets 403
  `insufficient_scope`).

A claim name is used as written, so namespaced claims such as
`"https://example.com/org_id"` or `"custom:tenant_id"` work. Pass a tuple such
as `("org", "id")` for a nested claim.

Rejected requests get 401 with `WWW-Authenticate: Bearer realm="Cayu"`, plus
`error="invalid_token"` when a token was sent. Response bodies and logs never
include the token. If no signing keys can be fetched, requests get 503 rather
than 401. If the issuer answers with a key set that has no usable keys, tokens
get 401: the issuer has withdrawn its keys, so this is not an outage.

Keys are cached for the JWKS response's `Cache-Control: max-age`, kept between
one minute and one day (10 minutes when the header is absent). A token with an
unknown `kid` triggers at most one refetch every 30 seconds, so rotated keys are
picked up without letting clients force a fetch per request, and concurrent
requests share one fetch. Every JWKS document the issuer serves (HTTP 200 with
a `keys` array) replaces the cached keys, so a key it removes, or all of them,
stops verifying tokens at the next refresh. Only when the issuer can't be reached, or returns an error, a
malformed document, or one over 256 KiB, do the previous keys stay in use, for
at most an hour after they expire. Set `max_stale_seconds=` on
`OidcSigningKeys` (0 to 24 hours) to change that: a longer window rides out
longer issuer outages, and a shorter one limits how long keys stay trusted
while the issuer can't be checked. Key fetches ask for an uncompressed body and
refuse compressed responses. Issuer and JWKS URLs must use HTTPS;
`allow_insecure_loopback=True` admits `http://` on a loopback host for local
testing only.

Typical values:

| Provider | `CAYU_OIDC_ISSUER` | `CAYU_OIDC_AUDIENCE` | Notes |
| --- | --- | --- | --- |
| Amazon Cognito | `https://cognito-idp.<region>.amazonaws.com/<user-pool-id>` | App client ID | Access tokens have no `aud`; pass `audience_claim="client_id"` and `required_claims={"token_use": "access"}`. |
| Auth0 | `https://<your-auth0-domain>/`, with the trailing slash | API identifier | Custom claims are namespaced; Auth0 Organizations put the organization in `org_id`. |
| Okta | `https://<org>.okta.com/oauth2/default` or another custom authorization server | `api://default` or your API audience | Scopes arrive in `scp`. |
| Microsoft Entra ID | `https://login.microsoftonline.com/<tenant-id>/v2.0` | The API's application (client) ID | Set the API's `requestedAccessTokenVersion` to 2 so access tokens use this issuer. Use a single-tenant issuer; `tid` holds the tenant. |
| Google | `https://accounts.google.com` | Your OAuth client ID | Verifies Google ID tokens; Google access tokens are not JWTs. |
| Workday or another OIDC provider | The `issuer` from its discovery document | The client ID or API audience it puts in tokens | Any provider that publishes discovery and a JWKS works. |

`OidcBearerAuth` verifies tokens that clients already hold. It does not provide
browser sign-in: there are no authorization-code redirects, callback routes,
refresh tokens, or session cookies. For browser access, put an identity-aware
proxy or your own login service in front of the server and have it forward the
token as a bearer header. API clients and scripts can obtain tokens directly,
for example with the client-credentials grant.

The parts work on their own. `await auth.verify_token(token)` returns the
verified claims or raises `OidcTokenError`, `await auth.verify_request(request)`
does the same from a request's header, and `auth.auth_context(claims)` builds
the `AuthContext`. `claims_mapper=` chooses what goes into
`AuthContext.claims`; by default it keeps `iss`, `sub`, `aud`, `azp`,
`client_id`, `scope`, `scp`, `exp`, and `iat`. `OidcSigningKeys` holds the
discovery and key cache and can be shared by verifiers with different
requirements. Pass `http_client=` (an `httpx.AsyncClient`) to send key fetches
through your own proxy or transport.

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

### Customer and operator access with OIDC

The service factory can take both policies from one OIDC provider. Share the key
cache, take the tenant from a verified claim for customers, and require an
operator scope for the control plane:

```python
from cayu.server import (
    AuthenticatedAccess,
    AuthenticatedProductAccess,
    OidcBearerAuth,
    OidcSigningKeys,
)

keys = OidcSigningKeys("https://example.okta.com/oauth2/default")
customers = OidcBearerAuth(
    issuer=keys.issuer,
    audience="api://support-agent",
    signing_keys=keys,
    tenant_claim="org_id",
)
operators = OidcBearerAuth(
    issuer=keys.issuer,
    audience="api://support-agent",
    signing_keys=keys,
    required_scopes=["cayu:operate"],
)

product_access = AuthenticatedProductAccess(dependency=customers.product_dependency())
operator_access = AuthenticatedAccess(dependency=operators)
```

`product_dependency()` returns a `ProductPrincipal` whose `tenant_id` and
`subject_id` come only from the verified token. A token without a non-blank
string tenant claim is rejected with 401; there is no default tenant. Pass
`tenant_claim=` to `product_dependency()` to use a different claim than the
verifier's own. `cayu check --deploy` treats this like any other
`AuthenticatedProductAccess` and `AuthenticatedAccess`. Keep the two policies
distinct, here with a scope; a separate audience also works. A customer token
must not be able to reach the operator control plane.

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

## Check that paused sessions resume on a release

Runtime meets a stored session at one of two boundaries. At `continuation`
(a pending approval, user input, tool round, child wait, model-completion
stage or provider-operation resolution, or an invocation that has not released
its run) the release must reuse the paused invocation's
profile exactly, except for the components Runtime keeps from it (its system
prompt and recorded run settings). At `resume` (the last invocation released
with nothing pending) the release is compared with the session's expected
profile and, when the application configures one, its `ExecutionProfilePolicy`.
To check this before publishing a release:

1. Read the stored profiles from the release that is serving now with
   `GET /api/sessions?include=execution_profile` (or
   `GET /api/pending-actions?include=execution_profile` for only the sessions
   with pending actions). Each response gains an `execution_profiles` list in
   the same order as `sessions` or `actions`. Each item reports the session's
   `boundary`; use `active_invocation.profile` when it is `continuation` and
   `expected` when it is `resume`. An item with `boundary: null` cannot be
   predicted; treat it as at risk. It either has an entry in `issues` (a
   damaged or unreadable record) or no profiles at all (a session created
   before execution profiles).
2. Write one entry per session, copying `agent_name`, `environment_name`,
   `provider_name`, `model` and `causal_budget_id` from the session itself:

   ```json
   {
     "sessions": [
       {
         "id": "session id",
         "boundary": "continuation",
         "agent_name": "assistant",
         "environment_name": null,
         "provider_name": "openai",
         "model": "gpt-5",
         "causal_budget_id": "session id",
         "expected_profile": {"schema_version": 6, "fingerprint": "...", "components": []}
       }
     ]
   }
   ```

   Every field is required. `environment_name: null` means the session has no
   environment, not the application default. `causal_budget_id` selects the
   causal-scoped budget limits the session runs under. If the listing shows a
   redacted or aliased `causal_budget_id` (the server hides ids that contain a
   workload secret), the exact id cannot be passed back, so that session's
   causal budget limits are not predicted.
3. Run `cayu execution-profile predict --input sessions.json --json` as a
   one-off task from the new release image, with the environment the release
   will serve with (including `CAYU_RUNTIME_BUILD_PROVENANCE` when you set it).
   It exits `0` when every entry will resume and `3` when at least one may not.
   Each prediction reports `outcome` (`exact_reuse`, `rejected`,
   `policy_dependent` or `not_comparable`), `admits`, and the
   `changed_component_classes`, for example `direct_tools` when a tool changed.

`cayu execution-profile candidates --json` prints each agent's candidate
profile on its own. Both commands build the application from the project
factory, never start it, load or write sessions, or call models or tools, and
close it before exiting. Whatever the factory does while constructing the
application, such as opening a store, still happens. Profiles include the typed
egress authority (policy names, destinations, allowed methods and paths), so
keep the files as private as the release configuration. See
[Predicting profile admission before a release](runtime-contracts.md#predicting-profile-admission-before-a-release)
for the full rules.
