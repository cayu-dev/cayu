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
- `deployment logs|status|timeline|wait|promote|retry` and `rollback` for immutable releases;
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
serving an older release. `rollback` still asks Cloud to publish the selected release and
returns once Cloud accepts it; `rollback --wait` (with `--poll-seconds` and
`--wait-seconds`) also waits for the service to run that release and reports a
`publication_error` the same way. `deployment wait` reports a ready release's
`publication_error` the same way instead of returning it as ready. While Cloud holds a
release's publication for the serving release's unfinished sessions, these waits keep
going past `--wait-seconds` (see [Unfinished sessions](#unfinished-sessions)).

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
`failure.diagnostic` includes evidence status/reason, pipeline stage,
exit code, a bounded redacted excerpt, and a truncation flag. Invalid, unsupported, oversized,
or unsafe payloads are withheld rather than printed. Older unversioned safe failures remain
supported through the existing compatibility projection.

`failure.diagnostic.stage` names where the release failed: `source_validation`,
`docker_build`, `image_build`, `database_migration`, or `smoke_test` for the release smoke
test (the same name as the `smoke_test` step in `deployment timeline`). A smoke-test failure
has no build output, so its evidence status is `unavailable` with reason `not_applicable`;
read `failure.detail` for the check that failed and `deployment timeline` for the steps.
Cloud may add phases and stages later. The CLI keeps a failure with an unfamiliar
`failure.phase` or `failure.diagnostic.stage`, including its code, message and hint, and
shows the value as Cloud sent it, or `unknown` when it is missing, isn't a lowercase
identifier, or fails the private-text filter. When the CLI normalizes a phase to `unknown`,
it drops `diagnostic_ref`, because the reference names the phase. Earlier CLIs dropped
such a failure and reported only the generic `deployment_failed` result.

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

When unchanged source resolves to a Release that Cloud promoted earlier, no promotion
publishes it again, so the deploy checks the Agent itself. If that Release is still the
Agent's selected Release but no service is running it, for example after `service
destroy`, the deploy asks Cloud to start the service, as `rollback` does, and reports
`service_publication_requested: true`. If the Agent has selected a different Release, or
none, the deploy exits `2` at once with category `release_not_selected`, the
`current_deployment_id`, and ready-to-run `rollback` and `service status` commands.
Select the reused Release with `rollback`, or change `version` in `cayu-cloud.toml` to
build a new one.

`--retry-failed` explicitly enables this default; `--no-retry-failed` reports the retained
attempt without retrying it. Non-retryable source failures include the retained failure and
instructions to change the source and deploy again. Terminal errors carry `deployment_id`,
`status`, and exact logs, timeline, and retry commands when the identifiers are safe.

Submit a retry directly, retaining the same key when retrying an uncertain HTTP outcome.
`--acknowledge-breaking REVISION` adds a breaking storage revision acknowledgement to the
retry (see [Breaking storage revisions](#breaking-storage-revisions)), and
`--session-policy`, `--session-wait-seconds` and `--acknowledge-session` change the
release's choice for unfinished sessions (see [Unfinished sessions](#unfinished-sessions)).
When Cloud's answer doesn't carry an acknowledgement or session choice the retry gave it
(for example because a new attempt was already running), the output adds `not_applied`
next to `result` and says so on standard error:

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
refreshes the short-lived access token before Cloud API calls. If Cloud still rejects
it with HTTP 401, for example when the local clock runs behind, Cayu refreshes the login
once and retries that call, so a long `deploy --wait` keeps polling. `cayu cloud logout`
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

## Breaking storage revisions

Some Cayu releases move the Agent database across a breaking storage revision. Cayu
0.9.0 and later move it from revision 114 to 115. Cloud migrates the database when it
publishes a release, and for a breaking revision it needs the owner's acknowledgement,
because the migration changes how the Agent runs:

- Cloud stops the previous release (web, worker and schedules) while the database
  migrates, and the Agent address shows its starting page until the new release starts.
  If the migration doesn't complete and Cloud can confirm the database is unchanged, it
  starts the previous release again.
- Releases built with the older Cayu can't open the migrated database, so you can't roll
  back or promote them afterwards.

An empty database, such as a new Agent's, needs no acknowledgement. Acknowledge the
revision when you deploy:

```bash
cayu cloud deploy . --acknowledge-breaking 115
```

`--acknowledge-breaking REVISION` can be repeated. Each value must be a whole number from
1 to 1,000,000, and at most 32 distinct revisions can be given; anything else fails with
`invalid_input` before the CLI signs in or uploads. The CLI sends the revisions, sorted and
without duplicates, as the release's `acknowledge_breaking` field. Without the flag the
field is omitted and the deploy's idempotency key is the same as in earlier CLI releases;
with it the key also covers the revisions. The automatic retry of a replayed failed attempt
sends the acknowledgement too, and Cloud keeps a release's acknowledgement on its retries.
Cloud only acknowledges breaking revisions on the actual migration path, so acknowledging
one the database has already passed does nothing.

Without the acknowledgement, Cloud refuses the publication before it stops anything: the
previous release keeps serving. `deploy` and `deployment wait` exit `2` with category
`service_publication_failed` and `error.code`
`storage_breaking_acknowledgement_required`. The message names the revisions, says what the
migration stops and that rollback can't cross it afterwards, and gives the exact command to
proceed. `error.acknowledge_breaking` lists the revisions and `error.commands.retry` adds
them to the same release:

```bash
cayu cloud deployment retry DEPLOYMENT_ID --application AGENT_SLUG --acknowledge-breaking 115
```

Running `cayu cloud deploy --acknowledge-breaking 115` again with unchanged source and
version also works: the CLI finds the release with the same source and deployment
settings, follows any build retries, and adds the acknowledgement to the release whose
publication failed (`result.retry` records it). This also works when the earlier submission
acknowledged different revisions; Cloud retains those acknowledgements and adds the new
ones. Different source or deployment settings are refused with Cloud's HTTP 409 reason;
give the changed deployment a new version.

Cloud reports three other breaking-migration failures the same way, each with its
`error.code`, Cloud's message, detail and hint, `error.publication_error`, and `status` and
`timeline` commands:

- `storage_writers_not_stopped`: the previous release did not stop in time, so the
  migration did not start and Cloud restarted the previous release. `error.commands.retry`
  retries the release after you fix the process that ignores shutdown.
- `storage_migration_state_unknown`: the migration failed and Cloud can't confirm the
  database is unchanged, so it left the services stopped. `error.commands.retry` resumes the
  migration of the same release.
- `storage_newer_than_release`: a rollback, promotion or deploy selected a release whose
  Cayu predates the database. The running service is not changed. There is no retry
  command; select or deploy a release whose Cayu supports the current revision. Use
  `rollback --wait` to see this refusal from the rollback itself.

`deployment timeline` also carries these failures, with phase `database_migrated`.

## Unfinished sessions

Before Cloud replaces the release serving an Agent with another one (after a deploy's
smoke test, a promotion, a rollback, or a publication retry), it reads the serving
release's unfinished Cayu sessions through the Agent's own control plane. A session that is
mid-interaction, or paused on a tool approval, user input or another pending action, may
not resume on the new release, so it blocks the replacement. A session interrupted with
nothing pending can adopt the new release at its next interaction and doesn't block.
Agents without the Cayu control plane are published as before.

You choose what Cloud does about blocking sessions, per release:

- `wait` (the default) holds the publication until no session blocks, for at most
  `--session-wait-seconds` (60 to 3,600; Cloud's default is 900), then refuses. The
  previous release keeps serving while it waits.
- `block` refuses at once.
- `proceed` publishes once every blocking session is acknowledged with
  `--acknowledge-session SESSION_ID`. The flag can be repeated, and `'*'` acknowledges
  every unfinished session, including ones Cloud could not read. It implies
  `--session-policy proceed`.

```bash
cayu cloud deploy . --session-policy block
cayu cloud deploy . --session-wait-seconds 1800
cayu cloud deployment retry DEPLOYMENT_ID --application AGENT_SLUG --acknowledge-session SESSION_ID
cayu cloud rollback DEPLOYMENT_ID --application AGENT_SLUG --wait --acknowledge-session '*'
```

`deploy`, `deployment retry` and `rollback` take the three flags and send them as
`session_policy`, `session_wait_seconds` and `acknowledge_sessions`. An invalid
combination (`proceed` without sessions, sessions with `wait` or `block`,
`--session-wait-seconds` without `wait`, more than 200 sessions, or an empty or
whitespace-padded session ID) fails with `invalid_input` before the CLI signs in or
uploads. Without the flags, the requests are unchanged: a new release gets Cloud's default
`wait` choice, and a retry or rollback keeps the release's current one. As with
`--acknowledge-breaking`, a deploy's idempotency key covers a choice other than Cloud's
default, so a deploy without the flags (or with `--session-policy wait` alone) keeps the
key earlier CLI releases used.

While Cloud holds a publication, the release keeps its `smoke_tested` or `promoted` status
and its `session_preflight.state` is `waiting`; `waiting_for` says whether Cloud waits for
`sessions` to settle, for a `session_read`, or for a sleeping Agent to wake
(`agent_wake`), and `deadline_at` says until when. `deploy`, `deployment wait` and
`rollback --wait` write Cloud's message and deadline to standard error when they change,
and keep waiting so the publication that follows has its usual time. Each time Cloud
reports a new check, a wait is extended to that check's `deadline_at` plus its own
`--wait-seconds`, counted from when the CLI sees the check. The time left until
`deadline_at` is measured on Cloud's clock and capped at 3,600 seconds per check, and a
wait is extended only while Cloud keeps checking, so a stalled preflight can't hold it
forever. Each of `deploy`'s waits (for the build, for a promotion that raced Cloud's own,
for the service, and on a rerun the extra waits described below) is extended on its own.
If a wait still ends first, the `deployment_still_running` or `service_still_starting`
result carries the last message as `last_issue`.

If Cloud selects another release while this one waits, it doesn't publish this one, and
its `session_preflight.state` becomes `superseded`. A release that was never selected stays
`smoke_tested`, and Cloud won't retry it. One that `deploy`'s promote selected before Cloud
held it stays `promoted`. `deploy`, `deployment wait` and `rollback --wait` exit `2` with
category `release_superseded`, Cloud's message, and `status` and `timeline` commands when
the release is `smoke_tested`, or `promoted` while the Agent has another release selected
(a `promoted` release that is still selected, for example after a rollback, isn't
affected). To serve that source anyway, change `version` in `cayu-cloud.toml` and deploy
again, or, for a `promoted` release, select it again with the `commands.rollback` given.

When Cloud refuses, the previous release keeps serving and the commands exit `2` with
category `service_publication_failed` and `error.code` `unfinished_sessions` (Cloud read
the blocking sessions) or `sessions_unreadable` (it could not read them all).
`error.publication_error.acknowledge_sessions` is Cloud's list of the blocking sessions
the release's choice doesn't acknowledge yet (`["*"]` when Cloud could not read them). A
retry replaces the release's choice, so `error.acknowledge_sessions` and
`error.commands.retry` also name the sessions an earlier `proceed` choice of the release
acknowledged. `error.commands.retry_wait` waits for the sessions again, with the
release's own `--session-wait-seconds` when it was longer than the default:

```bash
cayu cloud deployment retry DEPLOYMENT_ID --application AGENT_SLUG \
  --acknowledge-session SESSION_ID --acknowledge-session OTHER_SESSION_ID
cayu cloud deployment retry DEPLOYMENT_ID --application AGENT_SLUG --session-policy wait
```

Cloud lists at most 100 sessions. When its `session_preflight` says the list is
`truncated` (or the CLI can't confirm it isn't), or naming every session would take more
than 200, a retry naming them can't pass, so the CLI offers no `commands.retry` and keeps
Cloud's hint next to the wait alternative. It doesn't suggest `'*'` for you: that would
acknowledge sessions nobody has looked at. Check `deployment status` for the release's
`session_preflight` and decide, or retry with `--acknowledge-session '*'` yourself.

Answering or finishing the sessions and retrying with `--session-policy wait` publishes
the release once nothing blocks. A refused rollback or promotion returns the Agent's
selection to the serving release, and the same retry selects it again.

Running `cayu cloud deploy` again with unchanged source and any of the three flags (even
`--session-policy wait` alone, which keeps the original create request and key) gives the
release that choice. The CLI finds the release, as for `--acknowledge-breaking`, and
retries it with the choice when its publication was refused or is waiting
(`result.retry` records it). A retry of a waiting publication takes only the session
choice, so an `--acknowledge-breaking` given with it is listed in
`result.retry.acknowledge_breaking_not_applied` instead of `acknowledge_breaking`, and in
`result.not_applied`. A release that was still building is followed until it is built: if
Cloud then holds or refuses its publication (also when the refusal arrives just after
that check), the CLI retries it with the choice, and otherwise the choice goes with
`deploy`'s promote request.

Cloud can't always take the choice: the release may already be selected or published, a
promote may race Cloud's own, `--no-wait` or `--no-promote` may stop the deploy first, or
Cloud's finalize may already have checked the sessions under the earlier choice when the
promote stores the new one. After a promote that carried the choice, the CLI therefore
reads the release's recorded check and compares it with the choice: a check that found
nothing blocking agrees with any choice, and one that published by acknowledging sessions
agrees only if the choice acknowledges them too. Cloud records the check's mode
(`session_preflight.policy`) and which listed sessions it acknowledged, not the whole
choice. When it has recorded no check (the deploy didn't wait, or Cloud doesn't check this
Agent's sessions), or the list was truncated, the CLI can't tell.

`result.not_applied` reports what Cloud didn't take: `session_policy` is the choice given,
`release_session_policy` the one the release kept, `checked_session_policy` the choice its
recorded check passed under when that differs, `confirmed` is `false` when the CLI can't
tell, `more_permissive` says whether the release can publish over sessions the requested
choice would protect, and `acknowledge_breaking` lists revisions that weren't applied.
When the release has a recorded check, the check decides `more_permissive`: a check that
found nothing blocking, couldn't read sessions, or refused can't strand a session, and one
that published by acknowledging sessions can only if the requested choice doesn't
acknowledge them (or the list was truncated, or the state is one the CLI doesn't know).
Without a check yet, or while one holds publication, Cloud decides later under the kept
choice, so the kept choice is compared with the requested one: `block` and `wait` never
publish over a blocking session and rank the same, `proceed` ranks above them, and between
two `proceed` choices the one acknowledging a session the other doesn't (or `'*'`) is more
permissive. Nothing is more permissive than a requested `'*'`.

When `more_permissive` is `true` and the release wasn't already published before the
command, the command exits `2` with category `session_choice_not_applied`,
`not_applied`, and `status` and `timeline` commands, because this command let the release
publish over sessions it asked to protect. Otherwise it exits `0` and writes the message
to standard error. A `promoted` release counts as already published only if its recorded
check had passed when the command found it, or the Agent's service already ran it (the
CLI reads the service only then, and an unreadable service counts as not running it); a
`promoted` release can still be mid-check, and then this command's outcome matters.
`deployment retry` reports the same way, with `not_applied` next to `result`, when
Cloud's answer doesn't carry the session choice or an `--acknowledge-breaking` revision it
was given (for example because the publication was waiting, or because a new attempt was
already running).

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

## Local file acknowledgements

Cayu Cloud's source admission blocks code that keeps durable application state in
local database files under `/data` (`durable_file_state_detected`). A cache or
rebuildable index that legitimately lives there can be acknowledged in
`cayu-cloud.toml`, which turns a matching finding into a reviewed warning:

```toml
[storage]
local_files = [
  { path = "/data/cache/*.sqlite", reason = "Rebuildable embedding cache" },
]
```

`[storage]` may contain only `local_files`, an array of at most 50 tables. Each entry
needs a `path` glob of 1-256 characters and a `reason` of 1-500 characters (both
measured after trimming whitespace); other keys are rejected. `cayu cloud deploy`
applies the same rules as Cloud and fails locally with `manifest_invalid` instead of
uploading a source that Cloud would reject with `storage_acknowledgement_invalid`.
The table is accepted in schema versions 1 and 2, because Cloud scans every uploaded
source. It is not sent in the deployment request: Cloud reads it from the uploaded
`cayu-cloud.toml`, which matches findings by source file, database file name, or the
runtime path under `/data`, and also uses it for the post-deploy `/data` audit shown
as `storage_audit` in `cayu cloud service status`.

Nonempty `local_files` acknowledgements must be in the deployment root's
`cayu-cloud.toml`. A custom `--manifest` path containing acknowledgements is rejected
with `manifest_invalid` before upload, for both local and repository sources. Move
the manifest to the root as a regular file and deploy without `--manifest`. Custom
manifests with no acknowledgements (including an empty `local_files` array) remain
supported.
