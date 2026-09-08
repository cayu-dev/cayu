# Authenticated browser acceptance (explicitly opt-in)

`local_authenticated.py` runs a disposable password/TOTP site through the canonical
browser acceptance command. OpenAI selects the six ordinary browser operations;
an application-owned operator uses the protected HTTPS/WSS API for private input.
The site is delivered through the real egress broker. There is no Playwright
request interception, browser-worker mutation, ambient Chrome profile, or external
account. The resulting standard JSON/HTML report combines site-owned authentication
counts, durable fresh observations, operator handback and encrypted profile receipts.

This example requires **existing** prerequisites; it does not install or pull them:

- a running local Docker daemon and its CLI;
- Cayu's pinned interactive browser image;
- an application/controller Linux image with Python, Cayu's runtime dependencies,
  FastAPI, uvicorn, HTTPX, websockets, and cryptography;
- a compatible Linux Docker CLI binary and a directory of any additional Python
  dependencies to mount into that controller (an empty directory is fine when the
  image already contains all dependencies).

Set `CAYU_ACCEPTANCE_CONTROLLER_IMAGE`, `CAYU_ACCEPTANCE_DOCKER_CLI`, and
`CAYU_ACCEPTANCE_DEPS` to those existing resources. The Docker socket belongs only
to the application/controller, never the browser guest. The controller is trusted
application infrastructure; mounting that socket grants host Docker authority.

Set `CAYU_ACCEPTANCE_MAX_USD` to an explicitly authorized positive allowance of at
most `1`. The example uses `gpt-5.4-mini`, an exact app-wide reserving budget, finite
token/operation/time limits, and an additional physical-request guard. Its price
book uses USD 0.75/4.50 per million input/output tokens; verify these rates before
running. Cached input is conservatively priced as uncached by that guard. Actual
provider billing is separate from the conservative request reservations.

From the repository root, with the repository Python environment active:

```bash
export CAYU_ACCEPTANCE_MAX_USD=0.50
python examples/browser_acceptance/local_authenticated.py
```

Provide the authorized API key as one line on stdin using your preferred private
input/secret-manager mechanism. Do not put it in source, shell history, command
arguments, or Docker environment variables. The controller receives it over stdin.
The host launcher invokes:

```bash
python scripts/run_browser_acceptance.py \
  examples.browser_acceptance.local_authenticated:setup \
  --mode live_authenticated --authorize-authenticated \
  --output-directory "$CAYU_ACCEPTANCE_PROOF_DIR/reports"
```

The launcher creates and retires its own controller and labelled egress resources.
TLS keys are private and removed after the run. Reports and encrypted SQLite proof
stores remain in a private temporary directory, whose path is printed.
Before dispatching egress creation, the adapter callback synchronizes exact resource
names and session labels to `resource-intents.jsonl` in that private directory.
Host cleanup uses this inventory rather than current controller attachments, so
detached resources remain discoverable. The example also journals runner allocation
through the adapter's `create_runner` method, before the real `DockerRunner.create`
call. Networks and sidecars require inventoried names with matching session labels.
Runner removal requires its inventoried name, membership in the exact session-labelled
network, and a full Docker container ID; removal targets that ID before the network.
Unverified runner ownership is retained rather than authorizing name-only removal.
Unrelated Docker resources are not swept.
Successful creation is acknowledged in the journal after the Docker command returns.
An intent without that acknowledgement remains in `unsettled_creations` in the
cleanup report even if discovery is empty: removing the controller does not prove
an accepted Docker request stopped. Verify daemon-side settlement before manually
retiring those exact resources; empty discovery alone is not completion evidence.
Host cleanup failures preserve the original failure and inventoried resource identities in
`cleanup-resources.json` inside that private directory; inspect and settle those
resources before treating a failed teardown as complete. `unverified_networks`
records failed inspection, not permission to remove those networks. Invalid journal
entries are reported by line number in `invalid_journal_lines`; valid entries still
participate in cleanup, while malformed entries never authorize removal. Profile keys
are ephemeral: this local example proves restoration into a new browser, not
reconstruction of the entire campaign after losing the controller. Do not infer an
external site's authentication or availability from this local result.

For an application-owned account instead, supply your own **async context factory**
yielding a `BrowserAcceptancePlanV1`. Build its manifest with an exact
`BrowserAcceptanceAuthenticatedConfigV1`: explicit consent, opaque account and
site-observer revisions, the actual profile fingerprint, SHA-256 of the registered
operator policy identity, one canonical HTTPS origin, exact GET/POST endpoints,
and the authorized cost ceiling. Configure `authenticated_request_count` as a
bounded synchronous, site-owned monotonic counter of successful protected requests
exclusive to this designated account/browser journey, excluding unrelated site traffic;
it must not infer authentication from a requested URL or a model's answer. Change
the observer revision when its success condition changes. Keep credentials outside
these configuration and report objects. The context must retain the exact protected
server and profile/store lifetimes until environment cleanup has positively settled.

The canonical trial requires `navigate, observe, close` twice, two different
browser sessions, observation IDs `acceptance-post-handback` and
`acceptance-restored`, exactly two protected requests, two profile checkpoints,
and complete handback evidence. A model that does not follow that workflow fails
the trial; the harness never rewrites its tool calls to manufacture a pass.
The default no-argument authenticated manifest remains disabled, and the command
does not load authenticated setup without `--authorize-authenticated`.

Construct `BrowserAcceptanceAuthenticationCollector(counter,
observer_revision=config.site_observer_revision)` before the application and register
it in `CayuApp(runtime_hooks=[...])`. Supply that exact object as the plan's
`authentication_collector`, and the same counter as `authenticated_request_count`.
The collector only samples; it does not change calls or results. The harness checks
samples against durable operation identities. Each observation must have positive
phase-local site evidence: requests before the restored browser's navigation cannot
prove restored authentication. JSON/HTML reports retain both phase counts and their
observation/browser digests. Missing, overlapping or conflicting samples fail closed.
