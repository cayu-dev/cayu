# Server configuration

Cayu separates a server deployment's descriptive identity from its security
and lifecycle policy. `ServerConfig` is the fully resolved, validated contract
consumed by `create_server()`; it does not read environment variables or know
which secret manager an application uses.

Configured projects can apply the same contract through `cayu serve`; see
[project server](project-server.md) for its fail-closed
authentication, startup-recovery, and single-process boundaries.

## Programmatic configuration

Authentication is required unless open access is selected deliberately:

```python
from cayu.server import BasicAuth, ServerConfig, create_server

auth = BasicAuth(username="operator", password=resolved_password)
config = ServerConfig.protected(
    auth,
    deployment_name="production-eu",
)
server = create_server(cayu_app, config=config)
```

Basic-authentication realms are emitted in `WWW-Authenticate`. They must use
visible ASCII characters; embedded quotes and backslashes are escaped as HTTP
quoted-string content.

Custom JWT, OIDC, session-cookie, or gateway authentication keeps using the
existing callable dependency contract:

```python
from cayu.server import AuthenticatedAccess, ServerConfig

config = ServerConfig(
    deployment_name="preprod-eu",
    access=AuthenticatedAccess(dependency=require_operator),
)
```

An application may resolve `require_operator` or its credentials through KMS,
Vault, Kubernetes, a cloud secret manager, or any other source before building
the configuration. Cayu core does not add dependencies on those providers.

For trusted local development, the convenience profile makes every relaxed
choice visible:

```python
server = create_server(
    cayu_app,
    config=ServerConfig.local_development(),
)
```

It selects `OpenAccess`, enables generated documentation, and allows the local
Vite origin. `deployment_name="development"` alone does none of those things.

## Independent policy groups

`ServerConfig` owns these explicit axes:

- `access`: required `AuthenticatedAccess` or deliberate `OpenAccess`;
- `api`: whether the control-plane API is exposed and its mount path;
- `dashboard`: availability, path, directory, runtime data, and an optional
  access override (otherwise it inherits the server access policy);
- `evaluation_promotion`: an optional authenticated, stateless policy for
  reviewing and exporting captured sessions as portable eval cases;
- `evals`: optional authenticated runtime wiring for one trusted corpus target,
  one durable eval store, and the embedded fenced execution coordinator;
- `docs`: generated OpenAPI, Swagger UI, and ReDoc exposure;
- `cors`: allowed origins, methods, headers, and credential behavior; and
- `lifecycle`: replay timeout, startup recovery, inactivity fencing, durable
  side-effect recovery, and shutdown drain limits.

The packaged dashboard uses the configured local control-plane API, so an
enabled dashboard requires an enabled API. Disable both when exposing neither
surface. Server construction fails clearly when the dashboard is enabled but
its configured or packaged asset directory is unavailable or lacks an
`index.html` entrypoint. Blank directory values are rejected rather than being
interpreted as the process working directory. When generated documentation is
enabled, its reserved routes cannot also be used as the dashboard mount. CORS
credentials cannot be combined with wildcard origins, methods, or headers.
Mount paths are decoded ASGI paths: use literal path characters rather than
percent-encoded octets. Dot segments, repeated separators, backslashes, and
control characters are rejected during configuration resolution.

`AuthenticatedAccess` guards state-bearing control-plane routes and the
dashboard; the health route remains open for load balancers. Generated docs
are a separate public FastAPI surface and are not wrapped in the access
dependency, so enable `DocsConfig` only on a boundary where that exposure is
intentional.

When `lifecycle.startup_recovery_statuses` is configured, `create_server(...)`
processes one bounded incomplete-session recovery page before readiness. A
returned recovery cursor is continued after readiness by a managed background
task using the same statuses and inactivity cutoff; the task is cancelled and
awaited at shutdown. This keeps startup latency bounded without dropping older
durable recovery candidates.

The same API access policy guards `/api/contract` and
`/api/system/diagnostics`. The latter is a bounded, probe-free snapshot of
runtime-owned deployment configuration and registrations; it is not a
readiness endpoint or infrastructure monitor. Keep load balancer liveness
checks on `/api/health`.

`DashboardConfig.runtime_config` is serialized into the dashboard HTML and is
therefore browser-visible. Use it only for non-secret client configuration;
server credentials belong in the auth dependency or another trusted server-side
provider.

## Runnable-corpus promotion API

`EvaluationPromotionConfig` enables a narrow, stateless API that converts
runtime-attested terminal session evidence into a portable runnable corpus.
This embedded-server adapter is off by default. Enable it only on an
authenticated server and name the registered source agent whose evidence may be
converted:

```python
from cayu.server import BasicAuth, EvaluationPromotionConfig, ServerConfig

config = ServerConfig.protected(
    BasicAuth(username="operator", password=resolved_password),
    evaluation_promotion=EvaluationPromotionConfig(
        target_key="support.regressions",
        source_agent_name="support-agent",
        application_release_id="support-api-2026-08-06",
    ),
)
```

Preview reconstructs bounded terminal evidence, creates an editable suite and
case, and scores its assertions without executing the application. Export is
enabled only for the exact candidate most recently previewed; it downloads
deterministic portable corpus JSON and does not include the source session ID.
A changed durable snapshot, app manifest, source identity, pricing profile, or
edited candidate requires another preview.

Preview returns only candidates that satisfy the complete export contract.
Cost assertions therefore require a configured pricing profile containing the
selected currency; incompatible drafts are rejected before export is enabled.

`target_key`, `source_agent_name`, and `application_release_id` are public-safe
diagnostic configuration, not secrets or executable lookup authority. The source
agent must already be registered on the supplied `CayuApp`. The application
redactor validates all three values before the server starts. If dashboard
runtime configuration contains a validated `priceBook`, promotion uses that
same exact book for cost evidence; the exported corpus carries only its pricing
fingerprint, never the book itself.

The two promotion API routes are mounted only when this policy is configured:

- `POST /api/evals/promotion/sessions/{session_id}/preview`
- `POST /api/evals/promotion/sessions/{session_id}/export`

Both routes are request-byte bounded and covered by the server authentication
dependency. They are stateless adapters over the configured session store: they
do not save drafts, publish corpora, invoke providers or tools, or run the
exported eval. Configure the same nested fields through `ServerSettings` with
`CAYU_SERVER_EVALUATION_PROMOTION__TARGET_KEY`,
`CAYU_SERVER_EVALUATION_PROMOTION__SOURCE_AGENT_NAME`, and
`CAYU_SERVER_EVALUATION_PROMOTION__APPLICATION_RELEASE_ID`.

This adapter is independent of the control plane's current **Evaluate** action.
Configured projects started with `cayu serve` automatically publish their
registered agents as eval targets and use durable project storage, so operators
can evaluate a terminal session, save it, approve it as a baseline, launch a
current-app trial using the mounted application's provider, tools, environment,
approvals, and policy, compare results, and download reports without adding Evals-specific
Python configuration.

## Durable Evals execution

Configured projects started with `cayu serve` do not construct Evals objects.
Cayu derives the eval store from the project's declared SQLite or PostgreSQL
storage, generates one bounded target for each registered agent, and uses the
normal application provider, tools, environment, approvals, and runtime policy.
`cayu serve --dev` grants only trusted loopback product access; an authenticated
production control plane uses the same automatic assembly. Missing durable
storage or ordinary runtime authority remains visible as an operation-level
readiness reason rather than removing Evals from the dashboard.

Low-level embedded `create_server(...)` integrations have no project authority
from which to derive those decisions. They may explicitly construct
`EvalsConfig(target=..., store=...)` with a trusted `CorpusTarget` and durable
`SQLiteEvalStore` or `PostgresEvalStore`, then pass it through
`ServerConfig.protected(..., evals=...)`. `ServerSettings` cannot manufacture an
application, provider, PriceBook, database handle, or executable target from
environment text. Explicit target and store objects are excluded from model
serialization and safe summaries. See [runtime-native evals](evals.md#server-attached-durable-execution)
for the complete automatic and embedded contracts.

The resolved model is immutable, owns nested runtime JSON, and is evaluated
once when the server is created. A non-secret effective summary is available
through `config.safe_summary()`. Authentication callables and their credentials
are excluded from representations and serialization.

Changing `deployment_name` never changes any policy. It accepts any clean,
non-empty operator-defined value, such as `qa`, `production`, `preprod-eu`, or
`alice-local`. This identity is unrelated to Cayu's agent execution
`Environment`, which owns runners, workspaces, bindings, vaults, and execution
capabilities.

## Optional environment and dotenv loading

Install the source-specific loader separately:

```bash
pip install "cayu[server-settings]"
```

```python
from cayu.server import create_server
from cayu.server.settings import ServerSettings

settings = ServerSettings()
server = create_server(cayu_app, config=settings.to_config())
```

The prefix is `CAYU_SERVER_` and nested fields use `__`:

```dotenv
CAYU_SERVER_DEPLOYMENT_NAME=development
CAYU_SERVER_ACCESS__MODE=open
CAYU_SERVER_DOCS__ENABLED=true
CAYU_SERVER_CORS__ALLOWED_ORIGINS=["http://localhost:5173"]
```

Built-in Basic authentication is also source-loadable without exposing the
password in representations or serialized settings:

```dotenv
CAYU_SERVER_DEPLOYMENT_NAME=production-eu
CAYU_SERVER_ACCESS__MODE=basic
CAYU_SERVER_ACCESS__USERNAME=operator
CAYU_SERVER_ACCESS__PASSWORD=resolved-at-deploy-time
```

For application-defined authentication loaded from an external provider, pass
the resolved policy explicitly:

```python
from cayu.server import AuthenticatedAccess

settings = ServerSettings()
config = settings.to_config(
    access=AuthenticatedAccess(dependency=require_operator),
)
```

If the environment declares `CAYU_SERVER_ACCESS__MODE=external`, an explicit
policy must be supplied to `to_config()`. If it declares `open` or `basic`, an
explicit policy cannot silently replace that selection.

`pydantic-settings` provides deterministic precedence: explicit
`ServerSettings(...)` constructor values, process environment, `.env`, secret
files/customized sources, then defaults. Applications that use another source
can bypass `ServerSettings` completely and construct `ServerConfig` directly.
Missing access policy fails when settings are resolved. Unknown constructor
fields and unknown `CAYU_SERVER_` settings are rejected so misspelled policy
does not silently fall back to a default; unrelated entries in a shared `.env`
file are ignored.

File-secret directories use the same `CAYU_SERVER_` prefix and `__` nesting as
environment variables. For example, a file named
`CAYU_SERVER_ACCESS__PASSWORD` supplies the Basic-authentication password and
can be combined with mode and username from the process environment:

```python
settings = ServerSettings(_secrets_dir="/run/secrets")
```

Only regular files with the Cayu prefix are interpreted. Unknown prefixed
filenames fail validation, while unrelated files in a shared secrets directory
are ignored.

## Mounted host applications

When a product already owns the FastAPI application, select access explicitly:

```python
from cayu.server import AuthenticatedAccess, mount_cayu

mount_cayu(
    server,
    cayu_app,
    path="/internal/cayu",
    access=AuthenticatedAccess(dependency=require_operator),
    # evals=resolved_evals_config,
)
```

The host continues to own its documentation and CORS configuration.
`mount_dashboard()` remains a lower-level helper and does not protect a
separately mounted API automatically.

## Construction contract

`create_server()` requires a resolved `ServerConfig`; access, exposure, CORS,
documentation, and lifecycle policy are not accepted as separate function
arguments. Non-policy FastAPI constructor settings can be supplied through the
validated `fastapi_options` mapping. This deliberately narrow allowlist covers
API metadata, proxy root-path metadata, and a user lifespan that Cayu composes
with its own lifecycle. Routing, middleware, dependencies, exception handling,
request parsing, documentation routes, and debug behavior remain outside this
escape hatch. Cayu retains ownership of its title, documentation routes, debug
mode, startup/shutdown lifecycle, and lifespan composition.
`mount_cayu()` similarly requires an explicit `access` policy while the host
application continues to own the rest of its FastAPI configuration. This keeps
every security-sensitive choice visible in one validated model and prevents
deployment identity or convenience flags from selecting policy.
