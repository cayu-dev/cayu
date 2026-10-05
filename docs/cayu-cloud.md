# Cayu Cloud CLI

The `cayu cloud` command group deploys and operates complete Cayu Agent applications.
Cayu Cloud is currently invite-only. Login-backed commands are pinned to the production
service at `https://cloud.cayu.dev`; users never select an API URL. Run the remaining
commands from an Agent repository.

```console
cayu cloud login
cayu cloud whoami
cayu cloud deploy .
cayu cloud deployment status DEPLOYMENT_ID --application my-agent
cayu cloud deployment timeline DEPLOYMENT_ID --application my-agent
cayu cloud deployment logs DEPLOYMENT_ID --application my-agent
cayu cloud service status --application my-agent
```

## Command inventory

`cayu cloud --help` is the authoritative summary. The top-level commands are:

- `login`, `logout`, and `whoami` for interactive identity;
- `applications list|archive|archive-status` for Agent discovery and archive;
- `context use|show|clear` and `doctor` for connection selection and diagnosis;
- `init` and `deploy` for local project setup and publication;
- `deployment logs|status|timeline|wait|promote` and `rollback` for immutable releases;
- `runtimes list|status` for retained runtime artifacts;
- `service credentials|destroy|logs|restart|sleep|status|wake` for Agent infrastructure;
- `env list|set|unset` for Agent-owned configuration; and
- `evidence list|show|verify` for local content-free command records.

`deployment timeline` shows the release pipeline milestones and whether a failed
release can be retried or an active release can be cancelled. `deployment logs`
returns the Cloud-owned structured publication activity for that exact immutable
release. Both commands use authenticated, tenant-scoped customer API responses and
emit the same machine-readable JSON envelope as the rest of the Cloud CLI.

If a local deployment wait reaches its deadline while Cayu Cloud is still working, the
command exits `2` with category `deployment_still_running`; it does not report the
Release as failed or cancel it. The JSON error includes the current safe status and,
when Cloud returned canonical IDs, ready-to-run noninteractive `deployment status`,
`deployment timeline`, and `deployment wait` commands. Explicit `--context` and
`--api-key-file` selections are preserved in those commands. The validated application
and deployment identifiers remain available independently if the other identifier is
invalid. Cayu Cloud owns final promotion and Agent service publication, so these commands
only need to observe the durable Deployment Operation rather than reproduce those steps.

If release publication finishes but the Agent service is still starting when the same
local deadline expires, `deploy` exits `2` with category `service_still_starting` and a
ready-to-run `service status` command. The error also carries `waited_seconds` and, when
Cloud reported one, `last_issue`: the last `web_not_ready` message, which includes how
long the web process has been starting. A timed-out service teardown similarly reports
`service_deletion_still_running`; neither result marks the retained Cloud operation as
failed.

## Initialize an Agent project

`cayu cloud init` writes `cayu-cloud.toml`. For a project with a `[tool.cayu]`
factory it writes a web process that runs `cayu serve --host 0.0.0.0 --port 8000`.
Outside `--dev`, that command needs the `server` extra of `cayu` and a
`[tool.cayu.serve].auth` target, so init checks `pyproject.toml` first:

- If the `cayu` requirement in `[project].dependencies` lacks the `server` extra,
  init adds it (for example `cayu[postgres]==0.8.1` becomes
  `cayu[postgres,server]==0.8.1`). Run `uv lock` afterwards and commit both files;
  Cloud installs from `uv.lock`.
- If no auth target is configured, init adds:

  ```toml
  [tool.cayu.serve]
  auth = "cayu.server.environment_auth:OPERATOR_BASIC_AUTH"
  ```

  This target reads HTTP Basic credentials from `CAYU_OPERATOR_USERNAME` and
  `CAYU_OPERATOR_PASSWORD`, which Cayu Cloud provides to each Agent. Read them with
  `cayu cloud service credentials --application APP`.
- That module ships only in cayu releases newer than 0.8.1. When the `cayu`
  requirement still allows 0.8.1 or older (for example `cayu[postgres]==0.8.1`,
  `cayu>=0.8`, or a `[tool.uv.sources]` entry), init instead creates
  `server_auth.py`, which builds `BasicAuth` from the same two variables and works
  on every release, and sets `auth = "server_auth:OPERATOR_BASIC_AUTH"`.
  `result.serve.auth_module` reports it; commit it with `pyproject.toml`. Init never
  replaces an existing `server_auth.py` or `server_auth/`; it refuses instead and
  lists the requirement edits (including development pins) that would let you use
  the ready-made target. After the project requires a newer release you can switch
  to the ready-made target and delete the file.
- An existing auth target is never replaced; init reports that it left it alone.
  A project with a `service_factory` keeps its own product and operator access.

`result.serve` lists each change in `pyproject_changes` and any `next_steps`. When
the edit is not safe, for example dynamic dependencies, no `cayu` requirement, or a
requirement string that appears more than once, init exits `2` with category
`serve_setup_required` and the exact edits to make, without writing either file.
Projects generated by `cayu new` already include both settings, so init leaves
them unchanged. See [Project server process](project-server.md#serve-the-control-plane)
for custom authentication.

## Archive an Agent

Archive retires an Agent you no longer use and keeps its Releases, configuration, data and
history. Unlike `service destroy`, an archived Agent can't be deployed again, and archive
can't be undone. It requires an organization administrator signed in with `cayu cloud
login`; an Organization API key is refused with error `code`
`organization_admin_required`. If Cayu Cloud can't confirm the administrator, the command
fails with `code` `organization_directory_unavailable` and nothing changes; run it again.

```bash
cayu cloud applications archive-status my-agent       # read the current revision
cayu cloud applications archive my-agent --expected-revision 3
cayu cloud applications list --lifecycle archived
```

`archive` takes the exact Agent slug, never a display name, and the revision the decision
was made on; a changed Agent fails with error `code` `application_revision_stale`. The
idempotency key defaults to one derived from the Agent and revision, so repeating the
command replays the same archive. The command waits until Cayu Cloud observes that nothing
of the Agent can run; `--no-wait` returns once it is requested. If the local deadline
expires first, it exits `2` with category `archive_still_running`, the current `blockers`,
and a ready-to-run `applications archive-status` command; Cayu Cloud keeps archiving.
Brief API outages while waiting are retried, like a service wait.

Archived Agents leave `applications list` unless `--lifecycle archived` or `all` is given,
but `--application` still resolves them for reads such as `service status` and `deployment
logs`; `service status` then reports `archived`. An exact slug always names its own Agent,
so an archived Agent's reserved display name never makes another Agent's slug ambiguous.
Deploys, rollbacks, environment changes and service operations on an archived Agent fail
with error `code` `application_archived`, and a deploy that is waiting for its service
stops as soon as archive retires it. The CLI reports a Cloud rejection `code` only for
these documented codes; other rejections carry just `category` and `message`.

A Cayu app hosted elsewhere that still uses the archived Agent's Gateway key gets `403
forbidden` from Gateway inference, and its model-policy channel is refused new snapshots
with a non-retryable `forbidden`. Receipts stay readable. The app sees these as ordinary
provider and policy errors.

`service status` reports `degraded` when a required Agent process repeatedly fails to
start. Its `result.issues` array preserves Cloud's safe structured diagnostics, including
the affected process, failed-attempt count, stable issue code, and remediation hint. Use
the hint together with `cayu cloud service logs --application AGENT_SLUG`; credentials and
raw provider error bodies are never included in the issue payload.

By default, `deploy` also waits for a declared Agent service to reach `running`. A
`degraded` service ends that wait with a nonzero `service_degraded` result. Its
`error.issues` array preserves every structured process diagnostic, while `error.message`
aggregates their safe messages and remediation hints for humans. `--no-wait` keeps the
explicit asynchronous workflow.

The deployment worker publishes the promoted release once. `deploy` reads the service
and waits for its `deployment_id` to match that release; it does not start another
rollout. A brief 404 while publication is pending is retried as a read, and up to five
consecutive throttled, unavailable or HTTP 5xx service reads are retried before the wait
fails. While the service still runs an older release or none, `deploy` also reads the
release: if Cloud reports a `publication_error` (for example a failed database
migration), it exits `2` with category `service_publication_failed`, Cloud's message and
hint, and the structured `error.publication_error`. With `--no-wait`,
`result.service_publication_pending=true` identifies an absent service or one still
serving an older release. `rollback` still asks Cloud to publish the selected release.

Cloud's deployment worker promotes a smoke-tested release on its own, so `deploy`'s
promote request can race it. The request carries the Agent revision read before the
upload; if another release (often the previous one, still finishing) is promoted in the
meantime, Cloud answers HTTP 409 and then promotes the new release anyway. `deploy` treats
that 409 as a race: it keeps reading the exact release until Cloud promotes it, then waits
for the service as usual. It fails only when the release ends terminally (the normal
`deployment_failed` result) or is still unpromoted when the wait ends, or at once under
`--no-wait`. That failure has category `deployment_promotion_conflict`, Cloud's reason,
the release `status`, and `status`, `timeline` and `promote` commands. Cloud 409 messages
include Cloud's reason (for example "Application changed after the expected revision."),
filtered like deployment failure text, instead of only "HTTP 409".

For web Agents, `[web] ready_path = "/ready"` optionally replaces the default `/`. It is
a local absolute path of printable ASCII without spaces or a fragment; percent-encode
anything else. Cloud's Python stdlib container probe GETs the local web port with a
`ready_timeout_seconds` timeout (default 2, 1-30). HTTP statuses below 500, including
redirects and 401/403, prove readiness; redirects are not followed. After a
`ready_start_period_seconds` grace period (default 180, 0-300), three failed probes ten
seconds apart mark the task unhealthy. Cloud returns `web_not_ready` while starting and
`process_start_failed` with a log/recovery hint for unhealthy tasks or repeated exits.
The CLI waits for the healthy task from the current release and preserves these issues.

The probe is also a liveness check: ECS replaces a task that turns unhealthy, so a web
process whose readiness path stops answering for about 30 seconds is restarted, even in
the middle of a run. Point `ready_path` at a cheap handler, raise `ready_timeout_seconds`
for slow ones, and raise `ready_start_period_seconds` when startup recovery takes longer
than the default grace period. Default values are not sent, so older Cloud builds keep
accepting the manifest. Agent services currently use a single task and stop before
starting its replacement, so each redeploy has downtime while provisioning and startup
complete.

When an immutable Release fails to build, `deploy` automatically reads that Release's
timeline before exiting. If Cloud has a safe structured diagnostic, the nonzero JSON
result uses its stable category and message and includes `error.failure` with the phase,
bounded build detail, remediation hint, and automatic-retry decision. Coding agents do
not need to make a second timeline request. Older Cloud deployments and failures without
a safe diagnostic retain the generic `deployment_failed` result. This fallback now includes
`diagnostic_status=unavailable_or_unsupported` and, when identifiers are valid, the failed
application/deployment and a ready-to-run `deployment logs` command. A failed diagnostic
request does not change the original failed deployment result.

Supported schema-version-1 failures preserve safe structured details without requiring
specific English wording: code, phase, summary, repair hint, retry classification, attempt,
and diagnostic reference. Structured failures also include the application/deployment
identifiers and a logs command, including when evidence is unavailable or truncated.
`failure.diagnostic` includes evidence status/reason, build stage,
exit code, a bounded redacted excerpt, and a truncation flag. Invalid, unsupported, oversized,
or unsafe payloads are withheld rather than printed. Older unversioned safe failures remain
supported through the existing compatibility projection.

Read explicit evidence noninteractively:

```bash
cayu cloud deployment timeline DEPLOYMENT_ID --application AGENT_SLUG
cayu cloud deployment logs DEPLOYMENT_ID --application AGENT_SLUG
cayu cloud deployment logs DEPLOYMENT_ID --application AGENT_SLUG --diagnostic-offset 20 --diagnostic-limit 20
```

The log response's `diagnostics` array and `next_diagnostic_offset` provide evidence pages.
Offset/limit flags require a Cloud server implementing this contract; omit them for older
servers. A source error generally requires repairing and uploading new source. Transient
infrastructure failures may allow retrying the same source. Neither a new CLI nor a retry
can recover diagnostic output that the server never recorded.

`cayu cloud deploy .` keeps the same source idempotency for in-progress and successful
submissions. If the create response replays a failed or destroyed attempt with a safe,
automatically retryable Cloud failure, the command creates one new attempt from the same
source and includes a `retry` receipt with the old ID, new ID, failure code, and reason.
The source archive and manifest version are unchanged. A newly submitted attempt that
fails while waiting is reported to the caller rather than retried again.

`--retry-failed` explicitly enables this default; `--no-retry-failed` reports the retained
attempt without retrying it. Non-retryable source failures include the retained failure and
instructions to change the source and deploy again. Terminal errors carry `deployment_id`,
`status`, and exact logs, timeline, and retry commands when the identifiers are safe.

Submit a retry directly, retaining the same key when retrying an uncertain HTTP outcome:

```bash
cayu cloud deployment retry DEPLOYMENT_ID --application AGENT_SLUG \
  --idempotency-key retry-submission-1
```

Without `--idempotency-key`, each invocation uses a fresh submission key. The server creates
an immutable `-retry-N` Release from retained source; it can retry a destroyed failed attempt
if its source bundle still exists. A missing bundle produces an actionable HTTP 409, and the
CLI reports the server's reason as a `deployment_retry_rejected` error without suggesting
another retry. Paused publication and failed service finalization retain the existing server
retry behavior.

`cayu cloud deploy` creates the 8-63 character application slug declared in
`cayu-cloud.toml` when it does not exist, then updates it on later deploys. Slugs use
lowercase letters, numbers, and interior hyphens. `--application SLUG`
selects a different create-or-update slug; check it carefully because a valid typo
creates a separate application.

Local Python deployments require usable `pyproject.toml` and `uv.lock` files at the
root of the actual upload. Run `uv lock` in the directory you intend to deploy.
The CLI rejects missing, empty, excluded, or unusable linked inputs before uploading;
its `source_build_inputs_invalid` error includes the relative `path`, `reason`, and
repair `hint`. Git ignore rules can omit an untracked lockfile even when it exists
on disk. Include the file in the selected source rather than bypassing preflight.
The CLI does not generate locks or change ignore rules during deployment.

A successful wheel build does not prove the lockfile was uploaded or that the frozen
Linux dependency installation will succeed. Remote Git sources are validated by Cloud
after resolving the immutable revision; local preflight does not inspect remote content.

Deploy verifies that the local evidence directory is writable before authentication or
Cloud mutation. Deploy output and stored evidence replace every runtime `environment`
value with `[redacted]`, including values whose variable names do not look secret.

If Cloud deployment and rollout succeed but the final local evidence write fails, the
remote success remains authoritative: the command exits `0`, returns the successful
deployment result, and includes this machine-readable evidence status:

```json
{
  "evidence": {
    "category": "local_state_unavailable",
    "message": "Deployment succeeded, but local evidence could not be recorded.",
    "status": "unavailable"
  },
  "evidence_id": null,
  "operation": "deploy",
  "result": {}
}
```

The `result` object above is abbreviated; the real response retains the redacted
successful deployment result. An unusable evidence destination discovered during
preflight instead exits `2` with `local_state_unavailable`, before Cloud mutation.

Login uses WorkOS device authorization. Cayu opens the complete verification URL when
possible and prints the URL and one-time user code to standard error. Pass
`--no-browser` on SSH, in a container, or when a coding agent should ask a human to
complete authentication in another browser:

```console
cayu cloud login --no-browser
```

The access and rotating refresh tokens are kept in a private local auth file. Cayu
refreshes the short-lived access token before Cloud API calls. `cayu cloud logout`
deletes that local login.

Authentication selection is explicit: a successful `cayu cloud login` clears the
persisted private-context selection, while `cayu cloud context use PATH` selects that
context for later commands. Clearing the context falls back to the saved WorkOS login
and the fixed production endpoint. A saved login for any other endpoint is rejected;
run `cayu cloud login` again to sign in to production.

Private contexts and API keys remain the internal noninteractive path for CI,
operational handoffs, and automation:

```console
cayu cloud context use /private/path/cloud-context.json
CAYU_CLOUD_API_KEY_FILE=/private/path/key cayu cloud doctor
```

## Agent operator credentials

Cayu Cloud gives every Agent its own login for its `/cayu/` control plane and injects it
into the web process, worker, and schedules as `CAYU_OPERATOR_USERNAME` and
`CAYU_OPERATOR_PASSWORD`. Read it with:

```console
cayu cloud service credentials --application my-agent
```

```json
{
  "ok": true,
  "operation": "service.credentials",
  "result": {
    "env": {"password": "CAYU_OPERATOR_PASSWORD", "username": "CAYU_OPERATOR_USERNAME"},
    "password": "...",
    "username": "operator"
  }
}
```

The output contains a live credential. It is printed to standard output only: the command
writes no local evidence record, and nothing else stores it. Only the Organization that owns
the Agent can read it.

An Agent that Cayu Cloud has not published since it started issuing these credentials exits
`2` with code `operator_credentials_not_provisioned`; deploy it again and the next
publication creates them. A Cayu Cloud that predates per-Agent credentials exits `2` with
category `operator_credentials_unsupported`. To use a different login, set the same names
with `cayu cloud env set` (the password as a `--secret`); Agent values win over Cloud's.
The older environment-wide `AUTH_USER`/`AUTH_PASS` pair Cloud also injects is deprecated.

## Local deploy check

Cayu Cloud starts a public service (a project with `[tool.cayu] service_factory`) with
`cayu serve`, which refuses to start a production service while `cayu check --deploy`
reports a `PUBLIC_SERVICE_*` finding: placeholder or development product or operator
access, or non-durable identity, session or task storage. The web process would exit 1
on Cloud and keep restarting.

Before uploading a local directory whose `cayu-cloud.toml` declares `[web]`, `cayu cloud
deploy` runs the same check in-process, in production mode. If it reports any of those
findings, the deploy exits `2` with category `deploy_check_failed` and nothing is
uploaded. The JSON error has `error.deploy_check.blocking` (each finding's `code`,
`severity`, `path`, `message`, `hint` and `documentation_anchor`, for example
`cayu guide diagnostics#public-service-product-access-unsafe`), the command to rerun,
and a `hint`. Other check findings don't block the deploy. A successful deploy reports
`result.deploy_check.status` as `passed`, `skipped` or `unavailable`.

The check uses the environment of the shell running `cayu cloud deploy`; it can't read
Cayu Cloud secrets. If access or storage is configured through `cayu cloud env`, set the
same variables locally for the deploy, the way the service scaffold's own verification
command does:

```console
PRODUCT_AUTH_TOKENS_JSON=... CAYU_OPERATOR_BEARER_TOKEN=... cayu cloud deploy .
```

Pass `--skip-deploy-check` to upload anyway when you know Cloud supplies what the local
check can't see, or when you intentionally deploy a release that won't start.

The check runs in the Python environment running `cayu`, so run the deploy from the
project environment (for example `uv run cayu cloud deploy .`). If the project can't be
booted there, for example because its dependencies aren't installed, the deploy continues
with `result.deploy_check.status` `unavailable` and the reason, since Cloud builds the
release in its own environment. Source diagnostics that prevent safely importing the
service also report `unavailable`, never `passed`. Other source findings, such as a
missing README, do not suppress the service safety checks. Repository sources and projects
without a public service or a `[web]` process are not checked.

When the manifest's web command is `cayu serve`, the deploy also checks that
`cayu serve` can start outside `--dev`, for any project, not only a public service.
The command's arguments are parsed with `cayu serve`'s own parser, and the auth
target checked is the one `cayu serve` would load: `--auth MODULE:ATTRIBUTE` when the
command passes it, otherwise `[tool.cayu.serve].auth`. These checks read project files
and don't need `cayu[server]` installed locally:

| Code | Blocks when |
| --- | --- |
| `SERVE_SERVER_EXTRA_MISSING` | `[project].dependencies` has no `cayu` requirement with the `server` extra or an extra that includes its dependencies (`all`, `server-settings`, or `oidc`). |
| `SERVE_LOCK_MISSING_SERVER_EXTRA` | `pyproject.toml` declares server dependencies through one of those extras but `uv.lock` doesn't; Cloud installs from `uv.lock`. Run `uv lock`. |
| `SERVE_COMMAND_INVALID` | `cayu serve` would reject the web command's arguments, for example `--dev` with `--auth`, an unknown option, or an unclosed quote. |
| `SERVE_DEV_MODE` | The web command passes `--dev`. `cayu serve` refuses it on a non-loopback host such as `0.0.0.0`, and on loopback it is unauthenticated and unreachable from Cloud. |
| `SERVE_AUTH_MISSING` | Neither `--auth` nor `[tool.cayu.serve].auth` names a target and there is no `service_factory`, or `[tool.cayu.serve].auth` is not a non-empty string (which `cayu serve` rejects even with `--auth`). |
| `SERVE_AUTH_TARGET_UNRESOLVABLE` | The effective target's project module or attribute doesn't exist, it isn't callable, or it is the ready-made target and the declared `cayu` requirement allows 0.8.1 or older, which don't ship it. |
| `SERVE_AUTH_WITH_SERVICE_FACTORY` | A public service also sets `[tool.cayu.serve].auth` or `--auth`, which `cayu serve` rejects. |

Any callable auth dependency is accepted, not only Basic auth. The ready-made
`cayu.server.environment_auth:OPERATOR_BASIC_AUTH` target is accepted by name without
reading `CAYU_OPERATOR_USERNAME` or `CAYU_OPERATOR_PASSWORD`, because Cloud provides
them, but only when the declared `cayu` requirement excludes 0.8.1 and older. The
unmodified `server_auth.py` that `cayu cloud init` writes is accepted by name for the
same reason. Any other target is imported in the project context; if it can't be loaded here
for another reason, such as an environment variable or package only Cloud has, the
check reports `unavailable` with the reason instead of failing.

`cayu cloud init` runs the same checks. When it finds a problem, for example public
service access that is still a placeholder or a `uv.lock` to refresh after init added
the `server` extra, or can't run, the result includes `deploy_check` and a `warnings`
entry describing what to fix before deploying.

## Agent environment variables

Cloud-managed variables belong to the long-lived Agent, not to one immutable release.
They override same-named values from `cayu-cloud.toml` and are applied to the web
process, worker, and schedules when the Agent service rolls forward.

Plain values may be supplied as an assignment:

```console
cayu cloud env set MODE=demo --application my-agent
```

Secrets are write-only and must come from a file or standard input. The value is never
accepted as a command-line argument and is never returned by the API:

```console
cayu cloud env set VAPI_API_KEY --secret \
  --value-file /private/path/vapi-key \
  --application my-agent

printf '%s' "$VAPI_API_KEY" | \
  cayu cloud env set VAPI_API_KEY --secret --value-file - --application my-agent
```

List or remove configuration with:

```console
cayu cloud env list --application my-agent
cayu cloud env unset VAPI_API_KEY --application my-agent
```

Set and unset responses include the Agent service rollout status when a live service
is being updated.

Unchanged-source deploys follow the most recent retry with the same manifest and
policy, including nested `-retry-N-retry-M` versions made by earlier releases. An
active or successful retry is reused. When the newest attempt failed with a
Cloud-retryable failure, the CLI retries the original Release with a deterministic
submission key derived from the source and that failed attempt. Cloud adds each new
attempt as `VERSION-retry-N` beside the original and answers any retry of the same
family, from the CLI, the portal, or `deployment retry`, with the one live attempt.
If submission loses its response, the CLI reports `deployment_id` and
`retry_idempotency_key` for `cayu cloud deployment retry DEPLOYMENT_ID
--idempotency-key KEY`. Agent process health failures, on either smoke provider, and
other nonretryable failures require repairing the source.

## Fixed process resources

`cpu_millis` and `memory_mb` at the top of `cayu-cloud.toml` apply to all Agent
processes. Optional overrides belong under `[web]`, `[worker]`, or each
`[[schedules]]` entry. For example:

```toml
[web]
command = "python -m agent.web"
port = 8000
cpu_millis = 4000
memory_mb = 8192

[worker]
command = "python -m agent.worker"
cpu_millis = 250
memory_mb = 512
```

Cloud rounds these to supported static process sizes. `4000`/`8192` becomes
4096 ECS CPU units (4 vCPU) / 8192 MiB; `1000`/`2048` becomes 1 vCPU / 2 GiB.
Memory can require rounding CPU up as well. Deployment creation rejects requests
above the environment's configured ceiling and names valid sizes.

The deploy result and `cayu cloud service status --application AGENT_SLUG` retain
`service.resources`, with requested millis/MiB and effective ECS CPU units/MiB for
each web, worker and schedule. Omitted overrides inherit the top-level values.
Sizing is fixed for a release. Existing manifests are honored on their next deploy,
so an Agent previously using the hardcoded 0.5 vCPU / 1 GiB may incur a higher
running cost; the 1000/2048 example approximately doubles its compute allocation.

Resource values at both the top level and per-process level, along with
`timeout_seconds`, `port`, and `idle_timeout_seconds`, must be TOML integers; quoted
numbers and floats are rejected rather than converted. Unknown keys in `[web]`,
`[worker]`, and `[[schedules]]` are rejected so a misspelled override cannot silently
fall back to the default; a schedule error names the schedule, or its position when
it has no name.

When a declared size is above the environment ceiling or the largest supported
size, Cloud returns HTTP 422 with `detail.code = "manifest_invalid"` and
`detail.valid_pairs`: one to three `{"cpu_millis": ..., "memory_mb": ...}` pairs in
manifest units, within the ceiling, that are accepted as written. The CLI prints
these as copyable values, for example:

```text
Cayu Cloud API returned HTTP 422: Agent resources exceed supported sizes. Valid manifest pairs: cpu_millis = 4000, memory_mb = 8192; cpu_millis = 4000, memory_mb = 9216; cpu_millis = 4000, memory_mb = 10240.
```

The CLI never prints Cloud's free-text `detail.message`. If a `manifest_invalid`
rejection carries no usable `valid_pairs` (for example from an older Cayu Cloud
release), it prints `Cayu Cloud rejected the manifest resources.` instead.
