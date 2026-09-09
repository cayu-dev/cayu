# Live browser view with durable Docker reconnect

This local composition runs the protected application, broker and WSS server in
one application-owned container. A separate, internal-network browser allocation
survives a durable human-input pause and replacement of the Python worker process.
The application uses public Runtime configuration and tools in `app.py`.

The synthetic model navigates a portal, changes its display, asks for user input,
observes the same page after restart, changes its display again, and proposes one
business mutation. The operator policy permits **view only**. A separate human
review policy permits inspection and approval of the exact synthetic proposal.
The synthetic business database independently counts the committed mutation; the
viewer cannot submit it. No provider key, external account, or paid model is used.

## Run

Use a local Unix-socket Docker host (Docker Desktop on macOS or Linux Docker),
Python 3.11+, Node 22.18+, and a checkout of this revision. From the repository root:

```sh
uv sync --extra dev --extra browser
(cd dashboard && npm ci && npm run build:package)
.venv/bin/python -m playwright install chromium

docker build -f examples/browser_fetch/Dockerfile \
  -t cayu-browser-fetch:12-playwright-1.62.0 .
docker build -f examples/browser_view_reconnect/Dockerfile \
  -t cayu-view-reconnect:local .

PYTHONPATH=src:. .venv/bin/python examples/browser_view_reconnect/run.py \
  --state-dir /tmp/cayu-view-reconnect-demo
```

Add `--explicit-close` to finish with a fresh `browser_session` close operation
after protected input, worker replacement, and approval. The harness requires a
settled closed result, checks both generations of sidecars are gone, and verifies
that the application container survives. Without this flag, normal completion
owns allocation cleanup. The opt-in test runs both endings:

```sh
CAYU_RUN_DOCKER_VIEW_RECONNECT=1 PYTHONPATH=src:. .venv/bin/pytest -q \
  tests/egress/test_docker_viewer_reconnect_e2e.py::test_protected_ui_same_browser_after_worker_restart
```

Choose a **new**, absolute state directory on that Docker host. The harness creates
it with mode `0700`, mounts it and the checkout at their identical absolute paths,
and gives Docker management access only to the application container. `TMPDIR`
points at the shared private staging directory so daemon-side bind mounts resolve.
Do not put ownership/configuration files in the browser workspace.

The harness generates a disposable private TLS certificate for `cayu-control` and
loopback, installs only its public root in the guest, and publishes the protected
port on host loopback. HTTP/WSS clients verify this certificate; the dashboard
browser pins its public key. The broker port is not published. Production
applications must provision their own trusted certificate and secret configuration.

`run.py` drives the compiled existing operator UI. It samples a synthetic color
patch only in memory to prove changing frames from the exact admitted browser.
Browser actions invalidate page epochs, so the harness explicitly rediscovers and
reauthorizes the changed page. At each pause, the UI clears its canvas and reports
that the view is unavailable. A silent connection also expires within five seconds.
After worker replacement, the protected human-review API returns a current review
reference; resolution resumes the same browser and requires a fresh observation
before further browser actions. Reopening the view requires fresh authorization.

The harness records `evidence.json` containing Runtime/image IDs, Docker context,
container/page continuity, current frames, old-ticket rejection, protected review,
one independent mutation, and allocation cleanup. It does not export passwords,
keys, tickets, raw frames, browser transcripts, or operator input. Other files in
the private directory are application state and must not be copied into generic
reports. The fixture's business receipt is synthetic application data.

The application container is verified to survive Runtime allocation cleanup. The
harness then removes its own container; `--keep-server` leaves it available for
inspection. Dispose that exact container yourself when finished. Do not select a
container to delete by a shared application name. The harness's failure cleanup
uses only exact IDs recorded for this disposable run.

## Supported recovery boundary

Keep the exact application container, local Docker daemon, private ownership
directory, SQLite database, runtime/image versions, public TLS trust, and secret
configuration across worker restart. Retained guest processes are frozen during
pause/reconnect. Runtime rotates the egress proxy, sidecar credential, and CA under
an exclusive allocation claim before admitting work. The native guest settles its
old control channel and acknowledges a new invocation's binding and control epoch.
Old viewer tickets and guest channel messages cannot authorize the new generation.

This first composition supports a browser with no takeover history. Active or
prior takeover, sensitive entry, or uncertain manual input cannot be reset by
reconnect: the existing `BrowserControlConflict`/native fence requires explicit
exact-allocation closure and, if desired, a separate application rebuild. Recovery
never converts that outcome into permission to resume agent input or capture.

A replaced/missing application container yields `control_server_unavailable` (or
`configuration_mismatch` if its configured ID changed). Restore the exact original
container if it still exists, or dispose the exact retained allocation and rebuild
explicitly. Never edit a journal or substitute a same-name container. Alias
conflicts yield `control_server_alias_conflict`; remove the conflicting topology
through the deployment owner, not by granting Runtime ownership of foreign guests.
Unsettled Docker mutations retain `ownership_uncertain`; reconcile daemon work
and dispose exact resources before rebuilding. Cross-host failover, application
container replacement, browser-process/profile restoration, and hot policy adoption
are separate operations, not retained-allocation continuity.

Run the combined journey and selected real Docker failure checks together:

```sh
CAYU_RUN_DOCKER_VIEW_RECONNECT=1 PYTHONPATH=src:. \
  .venv/bin/python -m pytest -q tests/egress/test_docker_viewer_reconnect_e2e.py
```

This also kills separate workers before/after network attachment and verifies
that fresh recovery pauses the exact browser and reports `ownership_uncertain`.
The two-allocation test exercises lost attach/detach acknowledgements and stale
cleanup without disconnecting the other allocation.
