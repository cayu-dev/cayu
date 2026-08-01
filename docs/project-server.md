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
