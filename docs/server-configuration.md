# Server configuration

Cayu separates a server deployment's descriptive identity from its security
and lifecycle policy. `ServerConfig` is the fully resolved, validated contract
consumed by `create_server()`; it does not read environment variables or know
which secret manager an application uses.

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

`DashboardConfig.runtime_config` is serialized into the dashboard HTML and is
therefore browser-visible. Use it only for non-secret client configuration;
server credentials belong in the auth dependency or another trusted server-side
provider.

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
