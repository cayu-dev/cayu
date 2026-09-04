# Project server process

`cayu serve` boots the application factory declared in the nearest Cayu
project. The command follows the
[application construction contract](../src/cayu/guides/application-anatomy.md):
one synchronous, zero-argument factory call creates one process-scoped
`CayuApp`. Durable stores coordinate separate processes.

## Serve the control plane

For trusted local development, open access requires an explicit opt-in:

```bash
cayu serve --dev --host 127.0.0.1 --port 8000
```

`--dev` uses `ServerConfig.local_development()`. Without `--dev`, the command
fails closed unless an authentication target is configured:

```toml
[tool.cayu]
factory = "app:build_app"

[tool.cayu.serve]
auth = "server_auth:AUTH"
```

The target is loaded inside the project import context and must be a callable
accepted by `ServerConfig.protected()`. It can be a `BasicAuth` instance or a
custom auth dependency:

```python
# server_auth.py
import os

from cayu.server import BasicAuth

AUTH = BasicAuth(
    username=os.environ["CAYU_OPERATOR_USERNAME"],
    password=os.environ["CAYU_OPERATOR_PASSWORD"],
)
```

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
