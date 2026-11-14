# Cayu Cloud CLI

The `cayu cloud` command group deploys and operates complete Cayu Agent applications.
Cayu Cloud is currently invite-only. `cayu cloud login` signs in to the production service
at `https://cloud.cayu.dev`; `cayu cloud login --api-url URL` signs in to another Cayu Cloud
install instead (see [Signing in to another install](#signing-in-to-another-install)). Run the
remaining commands from an Agent repository.

```console
cayu cloud login
cayu cloud whoami
cayu cloud deploy .
cayu cloud applications show --application my-agent
cayu cloud deployment list --application my-agent
cayu cloud deployment status DEPLOYMENT_ID
cayu cloud deployment timeline DEPLOYMENT_ID
cayu cloud deployment logs DEPLOYMENT_ID
cayu cloud service status --application my-agent
```

## Command inventory

`cayu cloud --help` is the authoritative summary. The top-level commands are:

- `login`, `logout`, and `whoami` for interactive identity, and `organizations` to
  [list and choose your Organizations](#organizations);
- `applications list|show|archive|archive-status` for Agent discovery and archive;
- `context use|show|clear` and `doctor` for connection selection and diagnosis;
- `platform` for [which Cayu Cloud revision runs and when it rolled out](#cloud-rollouts);
- `init` and `deploy` for local project setup and publication;
- `deployment list|logs|status|timeline|wait|source|promote|discard|retry` and `rollback`
  for immutable releases, including [candidate releases](#candidate-releases) and
  [downloading a release's source](#downloading-a-releases-source);
- `runtimes list|status` for retained runtime artifacts;
- `service check|credentials|destroy|logs|pause|redeploy|restart|resume|sleep|status|wake` for
  Agent infrastructure, [publishing its Release again](#redeploying-an-agents-service) and
  [its own post-deploy check](#operator-check), and `schedule run` to run a schedule now;
- `env list|set|unset` for Agent-owned configuration, and `settings list|set|unset`
  for the Agent's [managed settings](#agent-settings);
- `data ls|cat|get|stat` to [read the files in an Agent's `/data`](#reading-an-agents-data-files);
- `sandboxes list|show|archives` to [read an Agent's brokered sandboxes](#reading-an-agents-sandboxes);
- `activity status` to [read how Cloud reads an Agent's activity](#clouds-activity-reader);
- `compass agents|show|enroll|disable|verify|pass|passes|job|missions|findings` for
  [Compass](#compass);
- `eval run|runs|compare|baseline` to [run an Agent's evals](#running-an-agents-evals) and
  read its eval runs;
- `api-keys create|list|revoke` for the Organization's
  [API keys](#organization-api-keys), as a signed-in admin;
- `connectors` and `connections list|show|add|authorize|check|agents|rotate|remove|bound|bind|unbind` for the
  Organization's [connections](#connections);
- `models list` for an Agent's [model slots](#model-slots);
- `admin system-agents` to list [the install's system Agents](#the-installs-system-agents),
  `admin arena-enrollments` to list [Arena case enrollments](#arena-case-enrollments) with each
  Agent's corpus schema, and `admin runtime-record show|forget-migration` to read
  [an Agent's runtime record](#an-agents-runtime-record) and forget a migration note in it, as a
  signed-in Cayu platform administrator; and
- `evidence list|show|verify` for local content-free command records.

`deployment timeline` shows the release pipeline milestones and whether a failed
release can be retried or an active release can be cancelled. `deployment logs`
returns the Cloud-owned structured publication activity for that exact immutable
release. Both commands use authenticated, tenant-scoped customer API responses and
emit the same machine-readable JSON envelope as the rest of the Cloud CLI.

A wait is never silent: before it first sleeps, and again whenever that changes, it says on
standard error what it is waiting for and where things stand: which Release it waits for
and the status Cloud reports for it, or which Release the Agent's service should run and
which one it still runs. It stops as soon as Cloud reports a failed Release or service.

If a local deployment wait reaches its deadline while Cayu Cloud is still working, the
command exits `2` with category `deployment_still_running`; it does not report the Release
as failed or cancel it. The message names the Release, and the JSON error includes
`waited_seconds`, the current safe status and, when Cloud returned canonical IDs,
ready-to-run noninteractive `deployment status`, `deployment timeline`, and `deployment
wait` commands. Explicit `--context` and `--api-key-file` selections are preserved in those
commands. The validated application and deployment identifiers remain available
independently if the other identifier is invalid. Cayu Cloud owns final promotion and Agent
service publication, so these commands only need to observe the durable Deployment Operation
rather than reproduce those steps.

While it waits, the CLI rides out up to five consecutive unavailable, throttled or HTTP
5xx reads of the Release, because the Release keeps building in Cloud whether or not one
poll arrives. A request that fails before Cloud answers exits `2` with category
`api_unavailable`, `transport_error` naming how it failed (for example `read_timeout` or
`connect_failed`) and `request` naming it as `METHOD /path`. A later `deploy` of the same
source finds the same Release, so running it again is safe.

If release publication finishes but the Agent service is still starting when the same local
deadline expires, `deploy` exits `2` with category `service_still_starting` and a
ready-to-run `service status` command. Its message says what the CLI was waiting for (the
service to run which Release, and what it still runs). The error also carries
`waited_seconds` and, when Cloud reported one, `last_issue`: the last `web_not_ready`
message, which includes how long the web process has been starting. A timed-out service
teardown similarly reports `service_deletion_still_running`; neither result marks the
retained Cloud operation as failed.

## What an Agent runs

`applications show --application NAME` shows what an Agent runs now: when it last served a
request, its selected release, when it was selected (promoted, or rolled back to), its source
repository and revision, and the Cayu runtime its image is built with, then any candidate or
in-flight release. It also takes the Agent positionally (`applications show NAME`), as it
always has; naming it both ways with different values is refused.
`deployment list --application NAME [--limit N]` lists the Agent's releases newest first
(20 by default) with their status, when each was created and selected, the source revision
and the Cayu runtime. Both print a table; `--json` prints the usual envelope, whose
`result` carries `schema_version` 1.

Selected isn't the same as serving. Cloud keeps the status of a release it refused to
publish, for example for the serving release's [unfinished
sessions](#unfinished-sessions): one a promotion selected first stays `promoted` and
selected while the previous release keeps serving, and only its `publication_error` says
so. Both commands mark such a release `not published` and print Cloud's reason and its hint
for getting past it under the release; `--json` keeps `publication_error` as Cloud returned
it. `service status` names the release the service runs as `deployment_id`.

```console
$ cayu cloud applications show --application my-agent
Agent: My Agent (my-agent)
Lifecycle: active
Last request: 2026-10-30 17:02Z
Selected release: dep_8k2m (version 1.4.0, promoted)
  Selected at: 2026-10-28 21:03Z
  Created at: 2026-10-28 20:51Z
  Source: https://github.com/acme/my-agent @ 3f2a9c1d0b7e4a5f6c8d9e0f1a2b3c4d5e6f7a8b
  Cayu runtime: cayu 0.16.1
  Cayu installed from: registry https://pypi.org/simple
Candidate or in-flight releases: none
```

Cayu Cloud reads the runtime version when it admits a release's source: every image is
built with `uv sync --frozen` from the source's root `uv.lock`, so the `cayu` version
that lock pins is the one the release runs. `deployment status` returns it as
`cayu_runtime` (`version`, `source` and `lock_digest`). Releases admitted before Cloud
recorded it, or whose lock pins no single `cayu` version, show "not recorded".

`deployment status|logs|timeline|wait|source|sbom` take a release id (`dep_...`) on its own: the
CLI asks Cloud which Agent the release belongs to, so `--application` is optional for them. A
Cloud from before that lookup still finds a release an Agent currently selects, and asks
for `--application` for any other release. `promote`, `discard`, `retry` and `rollback`
still require `--application`.

## When an Agent last served a request

`service status`, `applications list` and `applications show` say when the Agent's web
service last answered a request through Cayu Cloud's Agent router, as `last_request_at` (an
ISO 8601 time in UTC, or `null`). Use it to tell which Agents are still used before you
[archive](#archive-an-agent) one. Cloud records it to within about five minutes and
doesn't count its own reads (the inbox, session and activity reads, the home page probe).
`null` means none recorded: an Agent without a web process, one that has served nothing
since Cloud began recording, or a Cloud from before it did, which leaves the field out; the
CLI then gives `null` too. `applications show` prints it as `Last request: 2026-10-30
17:02Z`, or `Last request: never recorded`.

```bash
cayu cloud applications list | jq -r '.result.items[] | "\(.id) \(.last_request_at)"'
```

## Downloading a release's source

`deployment source RELEASE --output DIR` downloads the source bundle `cayu cloud deploy`
uploaded for that release and extracts it into DIR, which it creates when it is missing and
otherwise must be empty. The project's files land directly in DIR (the bundle keeps them
under `source/`). `--archive` writes the `.tar.gz` itself as the file `--output` names
instead, and never replaces an existing file. Like `deployment status`, it takes a `dep_...`
id without `--application`.

```console
$ cayu cloud deployment source dep_8k2m --output ./dep_8k2m
Extracted 214 files and 1 link (3184211 bytes) from release dep_8k2m into /work/dep_8k2m.
Verified digest sha256:9f3c...
$ cayu cloud deployment source dep_8k2m --output dep_8k2m.tar.gz --archive
```

Cloud sends the bundle's SHA-256 with it, and the CLI refuses a download that doesn't
match, writing nothing (`source_digest_mismatch`). Extraction checks every member before it
writes anything and refuses the whole bundle (`source_bundle_unsafe`) for an absolute path,
a `..` segment, a name listed twice, a member written through a link, a device or fifo, or
a link that leads outside DIR, also through a chain of the bundle's own links. Names are
compared the way macOS and Windows compare them, ignoring case, so two names that differ only
in case are refused as a duplicate and `UP` follows a link named `up`. A bundle that unpacks
to more than 100,000 members or 512 MiB of files is refused (`source_bundle_too_large`) as its
headers are read; tar metadata (a long name, PAX records) over 64 KiB is refused too. Python's
`data` extraction filter adds its own checks; a Python without it (3.11 before 3.11.4) refuses
to extract (`source_extraction_unsupported`) and only `--archive` works there. The download
and the extracted files are staged in hidden `.cayu-source-` entries inside DIR (so DIR may be
a mount point or sit under a read-only directory) and published only when all of them are
there. Publication works relative to open directory descriptors and never follows a link
below DIR, so a directory another user swaps for a link meanwhile is never written through;
Windows, which lacks these calls, only keeps the archive (`source_extraction_unsupported`).
Every file, directory and link is created under a name that must not exist yet, so nothing
another process puts in DIR meanwhile is replaced: the command refuses (`output_exists`) and
keeps already-published files for inspection. A failed publication may leave a partial tree;
the error says to inspect DIR and retry with a new or empty directory. Automatic rollback
cannot safely distinguish a published file from a concurrent replacement at the same name.
An archive refused before publication leaves no extracted files. An existing empty DIR keeps
its permissions, and `--output .` extracts into the current directory when empty. `--archive`
keeps the bundle as it is for
inspection. `--json` prints the usual envelope, whose `result` has `output`, `format`
(`directory` or `archive`), `digest`, `bytes` (the archive's size), `files`, `links` and
`file_bytes`.

Cloud refuses with an error `code`:

- `source_not_uploaded` when the release was built from a connected repository: there is
  no uploaded bundle, and the error names the `repository` and `revision` to fetch from it;
- `source_bundle_expired` when Cloud no longer keeps that release's bundle;
- `release_not_found` for a release the Agent doesn't have;
- `developer_required`, `agent_not_yours` or `key_scope_required` when the login or API key
  may not read the Agent's source; and
- `source_bundles_unavailable` when Cloud can't read bundles right now; run it again later.

A Cayu Cloud from before this download answers with category
`source_download_unsupported`.

## Downloading a release's SBOM

When SBOM admission refuses a release, `deployment wait`, `deployment logs` and `deployment
timeline` name the vulnerable packages, but Cloud keeps at most 50 of them. `deployment sbom
RELEASE` saves the whole CycloneDX SBOM that Cloud scanned, exactly as the release's image
build wrote it, to a new file: `--output FILE`, or `RELEASE.cdx.json` in the current directory
without it. It never replaces an existing file (`output_exists`). Like `deployment status`, it
takes a `dep_...` id without `--application`.

```console
$ cayu cloud deployment sbom dep_8k2m --output sbom.cdx.json
Wrote the SBOM of release dep_8k2m to /work/sbom.cdx.json (48213 bytes, 212 components, CycloneDX 1.5).
Verified digest sha256:4be1...
$ jq -r '.components[] | "\(.name) \(.version)"' sbom.cdx.json | grep -i pypdf
pypdf 6.21.2
```

Cloud sends the SBOM's SHA-256 with it. The CLI writes nothing when the download doesn't
match it (`sbom_digest_mismatch`) or isn't a CycloneDX JSON document (`sbom_invalid`).
`--json` prints the usual envelope, whose `result` has `output`, `format` (`CycloneDX`),
`spec_version`, `components` (how many it lists), `digest` and `bytes`.

Cloud refuses with an error `code`:

- `release_sbom_not_found` when the release's image was never built (it stopped before
  `image_built`), so there is no SBOM; the error's `status` is the release's status;
- `release_sbom_removed` when the release was destroyed, which removes its build evidence;
- `release_not_found` for a release the Agent doesn't have;
- `agent_not_yours` or `key_scope_required` when the login or API key may not see the Agent
  (an API key needs `read`); and
- `release_sboms_unavailable` when Cloud can't read SBOMs right now; run it again later.

A Cayu Cloud from before this download answers with category `sbom_download_unsupported`.

## Initialize an Agent project

`cayu cloud init` writes `cayu-cloud.toml`. For a project with a `[tool.cayu]`
factory it writes a web process that runs `cayu serve --deployment-kind cloud --host 0.0.0.0 --port 8000`.
For a project that declares `[tool.cayu.workers]` it also writes a `[worker]` that runs
`cayu worker NAME`, next to `[web]` when there is one; without it, runs the web process
queues for the worker would never execute. Cayu Cloud runs one worker process per Agent,
so with several workers init writes the first one declared and names the others in
`result.notes`; combine them into one worker entrypoint, or run the ones that work on a
timetable as [`[[schedules]]`](#schedules). A project with neither
gets a worker that runs its only console script, or `python -m PACKAGE`.
`result.worker.command` reports a worker written next to `[web]`.

`--schedule NAME=EXPRESSION=COMMAND` (repeatable) writes a [`[[schedules]]`](#schedules)
entry, and `--schedule-timezone ZONE` sets the time zone of every one given, for example:

```bash
cayu cloud init . \
  --schedule 'morning=cron(0 8 ? * MON-FRI *)=python -m herald digest --slot morning' \
  --schedule 'evening=cron(0 17 ? * MON-FRI *)=python -m herald digest --slot evening' \
  --schedule-timezone America/Los_Angeles
```

`--force` replaces an existing `cayu-cloud.toml` but keeps what init never writes:
`home`, `[env]`, `[storage]`, `[webhooks]`, `[operator_check]`, `[[schedules]]` and
`[session_replacement]`. A `--schedule` with the name of a kept schedule replaces it.
`result.kept` names what it kept, and `result.replaced` the fields init generates that
the old file set to something else (a hand-raised `cpu_millis`, an `idle_timeout_seconds`
under `[web]`), so they can be set again; both are repeated in `result.notes`. A kept
value TOML can't write back as it was read (a date or time) stops init with
`manifest_invalid` before anything is written.

`cayu cloud init --check` writes nothing. It reads the existing `cayu-cloud.toml` the way
`cayu cloud deploy` reads a local source, with the same loader and Cloud's own
[schedule rules](#schedules), builds the same source bundle without uploading it, and runs
the deploy check's file checks on the `cayu serve` command, with no login. It doesn't boot the
project, which opens (and can create) its stores, so for a public service its `deploy_check` is
`skipped` and names `cayu check --deploy --fail-on warning --json`, which the deploy also runs. It answers `operation` `init.check` with the
processes and schedules it read (each schedule's `timezone`, UTC unless it sets one), or
the error the deploy would stop with, such as `manifest_invalid`, before uploading anything.

Outside `--dev`, the `cayu serve` command needs the `server` extra of `cayu` and a
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
  requirement still allows 0.8.1 or older (for example `cayu[postgres]==0.8.1` or
  `cayu>=0.8`), or a `[tool.uv.sources]` entry installs cayu from somewhere that
  can't be shown to contain the module, init instead creates
  `server_auth.py`, which builds `BasicAuth` from the same two variables and works
  on every release, and sets `auth = "server_auth:OPERATOR_BASIC_AUTH"`.
  `result.serve.auth_module` reports it; commit it with `pyproject.toml`. Init never
  replaces an existing `server_auth.py` or `server_auth/`; it refuses instead and
  lists the requirement edits (including development pins) that would let you use
  the ready-made target. After the project requires a newer release you can switch
  to the ready-made target and delete the file. A `[tool.uv.sources]` path source
  counts only when it is a wheel, a `.tar.gz` source distribution, or a source tree
  that contains `cayu/server/environment_auth.py`; archives are read by file name
  only. Its path must be relative and stay inside the project, and any symlink on
  the way must be relative and stay inside it too, since Cloud builds a copy of the
  project somewhere else. The deploy must [upload](#what-a-deploy-uploads) that
  archive or module file and each of those symlinks. A git, URL, or index source
  never counts.
- An existing auth target is never replaced; init reports that it left it alone.
  A project with a `service_factory` keeps its own product and Developer dashboard access.

`result.serve` lists each change in `pyproject_changes` and any `next_steps`. When
the edit is not safe, for example dynamic dependencies, no `cayu` requirement, or a
requirement string that appears more than once, init exits `2` with category
`serve_setup_required` and the exact edits to make, without writing either file.
Projects generated by `cayu new` already include both settings, so init leaves
them unchanged. See [Project server process](project-server.md#serve-the-developer-dashboard)
for custom authentication.

Init also writes `[sign_in]` and reports its mode as `result.sign_in`. It writes
`mode = "cloud"`, so Cayu Cloud signs people in to the Agent's pages once its router
sign-in ships, when `[tool.cayu.serve].auth` is (or init sets it to)
`cayu.server.environment_auth:OPERATOR_BASIC_AUTH` or `OPERATOR_ONLY_AUTH` (the b2b
preset's, which keeps people signed in as Cloud members off the Developer dashboard), both of
which accept Cloud's sign-in, and for a worker, which has no pages. Otherwise it writes `mode = "own"`: a project with a
`service_factory`, an auth target of the project's own, or the generated
`server_auth.py` keeps its own sign-in. See
[Sign-in and home page](#sign-in-and-home-page).

For a project generated with `cayu new --with app-database`, init also writes
`[database]` with `migrate = "alembic upgrade head"` and reports it as
`result.database.migrate`. See
[Application database migration](#application-database-migration).

## Agent processes on Cayu Cloud

This section is what an Agent's code can count on once it runs on Cayu Cloud. A
`cayu-cloud.toml` declares up to three kinds of process, all built from one
immutable Release image:

- `[web]` runs one long-lived task serving `port`, reached at the Agent's own
  address. Cloud probes it for readiness (`[web] ready_path`, below) and may
  let it sleep when idle and wake it on the next request.
- `[worker]` runs one long-lived task with no inbound traffic, for example
  `cayu worker NAME`. Cloud runs at most one worker table per Agent and restarts
  the task if it exits.
- Each `[[schedules]]` entry starts a separate short task at the times its
  `expression` names. An Agent declares at most 100 schedules, each with a unique
  `name` of lowercase letters, digits and interior hyphens (at most 63
  characters).

`entrypoint` is used only when the manifest declares none of these. Web, worker
and schedules share the Release image, the Agent's environment variables
(`cayu cloud env`), and `/data`. When Cloud provisions the Agent's PostgreSQL
database, every process gets it as `CAYU_DATABASE_URL` together with
`CAYU_REQUIRE_POSTGRES=1`, so Cayu's SQLite stores refuse to open; keep durable
state in that database.

Cloud runs each `command` without a shell. It splits the string the way a POSIX
shell splits words (quotes group, nothing is expanded) and starts the program
directly, so pipes, `&&`, redirects and `$VARIABLE` don't work. Wrap such a
command in `sh -c '...'`; the image has the BusyBox `sh`, not `bash`.

### What keeps running

- The web process sleeps once it has been idle for its idle timeout: Cayu Cloud's
  default (15 minutes on Cayu's own Cloud), or `[web] idle_timeout_seconds`
  (60–86400). The next request wakes it, which takes a minute or more. While it
  sleeps, nothing it started runs: no background tasks, loops or watchers that
  it started from its lifespan.
- The worker never sleeps. Cloud keeps it running and restarts it if it exits.
- Work that must keep going without requests therefore belongs in `[worker]`:
  polling an inbox or a queue, task workers (`run_task_worker`), timers, and event
  watchers. Declare it in `[tool.cayu.workers]` and `cayu cloud init` writes it as
  `[worker]` (`cayu worker NAME`). Work that runs at fixed times belongs in
  `[[schedules]]`. Don't also start the same work in the web process, or two
  processes will do it.
- External-event waits hosted with `app.host_external_waits(...)` already split this
  way: `cayu serve` takes their webhooks and the worker continues the waiting
  sessions. List the delivery path under
  [`[webhooks] paths`](#webhooks-to-a-sleeping-agent); see
  [external-event waits](external-event-waits.md#run-them-from-cayu-serve-and-cayu-worker).
- `idle_timeout_seconds` can't turn sleep off, and no other manifest value does.

### How the worker stops

A deploy, restart or pause stops the worker with `SIGTERM` and kills it 30
seconds later. A worker that needs longer to finish what it's doing declares its
own grace, a whole number of seconds from 10 to 120 (ECS's largest stop
timeout):

```toml
[worker]
command = "cayu worker scout"
stop_grace_seconds = 120
```

Cloud passes it to the platform and, for a `cayu worker` command that doesn't
set `--shutdown-grace-seconds` itself, sets that flag to the grace less 5
seconds, so the worker's cooperative shutdown finishes before it's killed. Any
other command has to stop within the grace on its own. The web process and
schedules keep 30 seconds. A longer grace makes a deploy that replaces the
worker take up to that much longer, because the old worker stops before the new
one starts. The value is part of the Release; a manifest without it sends the
same payload as before. A Cayu Cloud release that predates the key refuses it.

A task the worker can't finish or hand back within the grace stays fenced; see
[a deploy during a long task](project-workers.md#a-deploy-during-a-long-task).
`cayu cloud service status` reports such tasks (below).

### Draining the worker a deploy replaces

A worker can instead keep running after a later release replaces it, for up to
`drain_seconds` (a whole number from 60 to 172,800, which is 48 hours), and finish the
tasks it holds while the new release's worker takes every new task:

```toml
[worker]
command = "cayu worker scout"
drain_seconds = 10800
```

It is the replaced release's value that counts, so the first release that declares it is
still replaced at once. Cloud drains only an Agent with its provisioned database, since
two workers run side by side, and both see the same `/data`. `cayu worker` asks Cloud
whether its release is draining and, once it is, stops claiming and exits when its tasks
finish ([a deploy during a long task](project-workers.md#a-deploy-during-a-long-task));
another worker command keeps claiming until Cloud stops it at the deadline. The value is
part of the release; a manifest without it sends the same payload as before. A Cayu
Cloud release that predates the key refuses it. `cayu cloud service status` lists each
drain while it lasts, with its deadline and the time left, and `cayu cloud deployment
status` lists the drain a release's publication started.

Before it uploads anything, `cayu cloud deploy` reads the Agent's claimed and running
tasks through the Agent's own task API
([reading an Agent's state](#reading-an-agents-state-from-the-cli)) and, when there
are any, prints a warning to standard error naming them (up to 20 of each) and adds
it to the JSON `warnings`. The JSON `result.running_tasks` has the list (`status`
`read`, with `complete` false when there were more), `not_running` when the service
is stopped or absent, or `unavailable` with the reason the read failed, which never
stops the deploy. An Agent without a `[web]` process has no Cayu API to read, so its
tasks are reported `unavailable` (`no_address`): its worker may still be running them.
Cayu Cloud's deploy check then covers these tasks with the session choice below: `wait`
holds the publication until they finish, `block` refuses, and `proceed` interrupts only
the tasks `--acknowledge-session` names (by task ID, or `'*'`). Once the release is
published, the JSON `warnings` also name the tasks it interrupted, a Cayu Cloud whose
check predates tasks (so the choice didn't cover them), and the drain of the replaced
worker. A candidate release (`--no-promote`) lists the tasks for its later promotion
(`result.running_tasks.candidate`). An Agent with no selected release skips the check.
Reading an asleep Agent wakes its web process.

### Schedules

`expression` uses Amazon EventBridge Scheduler's syntax and is evaluated in UTC, or in
the schedule's `timezone`:

- `rate(VALUE UNIT)` with a unit of `minute`, `hour` or `day` for a value of 1
  and `minutes`, `hours` or `days` otherwise, such as `rate(15 minutes)` or
  `rate(1 day)`;
- `cron(MINUTES HOURS DAY-OF-MONTH MONTH DAY-OF-WEEK YEAR)` with six fields,
  where one of day-of-month and day-of-week is `?`, such as
  `cron(0 14 ? * MON-FRI *)` for 14:00 UTC on weekdays. Numbered days of the week
  run from 1 (Sunday) to 7 (Saturday).

A run starts at its time, without a flexible window, and runs don't wait for
each other: a run that outlasts its interval overlaps the next one. On an
install that runs Agents on Kubernetes, an expression must keep its exact
meaning as a five-field UTC cron: a minute rate that divides an hour, an hour
rate that divides a day, `rate(1 day)`, or a `cron()` without `L`, `W`, `#`, a
year limit, or a stepped day of the week. Each run's output is in
`cayu cloud service logs --process schedule:NAME` ([reading logs](#reading-an-agents-logs)).
`cayu cloud deploy` and `cayu cloud init --check` apply these rules before anything is
uploaded.

`timezone` names the IANA time zone a `cron()` expression is read in, so a schedule
fires at the same local time all year, across daylight-saving changes. Without one it
is UTC. A `rate()` expression counts an interval rather than a time of day, so it takes
no time zone, and an unknown zone is refused. A zoned schedule also can't run in an hour a
daylight-saving change skips or repeats in its zone (1 AM and 2 AM in `America/Los_Angeles`,
2 AM in `Europe/Paris`): EventBridge runs a repeated time once where Kubernetes runs it twice,
so Cloud refuses those, and the error names the hours. An hourly schedule in such a zone is
written in UTC:

```toml
[[schedules]]
name = "morning"
command = "python -m herald digest --slot morning"
expression = "cron(0 8 ? * MON-FRI *)"  # 8:00 AM Pacific, winter and summer
timezone = "America/Los_Angeles"
```

A schedule without `timezone`, or with `timezone = "UTC"`, sends the same manifest as
before. A Cayu Cloud that predates schedule time zones refuses any other with
`schedule_timezone_unsupported`; nothing is deployed.

`cayu cloud schedule list --application AGENT` lists the schedules of the Agent's
selected release: each one's `name`, `expression`, `timezone` (`UTC` when it sets none)
and `command`, and its `input` when it has one.

A schedule can carry a JSON `input`, a table, array, string, number or boolean
of at most 4096 bytes as compact JSON once escaped (quotes and backslashes take
two bytes, other non-ASCII characters six). Each run gets it as
`CAYU_SCHEDULE_INPUT`, compact JSON with sorted keys, beside
`CAYU_SCHEDULE_NAME`; scheduled tasks have no standard input, so the command
reads the variable. The input shows in the schedule and its tasks, so it must
never hold a secret. A schedule without `input` sends the same manifest as
before, and `cayu cloud deploy` checks the input before uploading:

```toml
[[schedules]]
name = "scout"
command = "python -m scout.run"  # reads CAYU_SCHEDULE_INPUT
expression = "rate(6 hours)"
input = { prompt = "Look for new pricing pages", limit = 20 }
```

`cayu cloud schedule run NAME --application AGENT` starts one run of a deployed
schedule now, as its expression would, and prints the run Cloud started (its
output is in the logs); `--input JSON` gives that run its own input instead of
the schedule's. A Release that declares no such schedule is refused with `code`
`schedule_not_found`. While Cloud deploys or changes the Agent's service it refuses the
run, and `--wait` sends it again once that finishes ([while the service
switches](#while-the-service-switches)); so do the `service` changes below.

`cayu cloud service pause --application AGENT` scales the worker to zero and
stops the schedules starting runs; runs already started finish, and the web
process keeps serving, so people can still open the Agent and decide what waits
on them. A pause outlasts deploys, rollbacks, environment changes and restarts
until `cayu cloud service resume`; `service destroy` clears it. Both are
idempotent and print the service, whose `paused` and `paused_at` (also in
`service status`) say where it stands. `schedule run` works while the Agent is
paused. These need a Cayu Cloud release that supports them; an older one
answers `service_pause_unsupported` or `schedule_run_unsupported`, and refuses a
manifest with `input`.

### The runtime image

Cloud builds each Release on the official `python:3.13-alpine` image, so the
Agent runs Python 3.13 on Alpine Linux with musl libc. The build copies the
uploaded source to `/app`, adds `git` and `ripgrep`, and runs
`uv sync --frozen --no-dev` from `uv.lock`; the project's virtual environment is
first on `PATH`. Development dependencies are not installed, and there is no
`gh`, `bash` or C compiler. Every dependency needs a pure Python or `musllinux`
wheel; one that ships only glibc (`manylinux`) wheels and must compile fails the
build. Call web APIs such as GitHub's REST API instead of command-line tools the
image lacks.

### Files under /data

`/data` is a persistent directory of the Agent's own on a network file system,
mounted read-write in the web process, the worker and every schedule run, which
all see the same files. It survives deploys, rollbacks, restarts, sleep and
`service destroy`; no other Agent can mount it. Production backs it up daily and
keeps each backup 35 days; ask Cayu support for a restore. It is for files: Cloud
refuses a Release that keeps durable database state in local files there unless
you acknowledge them (see [Local file acknowledgements](#local-file-acknowledgements)),
and `cayu cloud service status` reports what it finds under `/data` as
`storage_audit`. Cloud names the directory to the Agent's Cayu as `CAYU_DATA_VOLUME`, so
`cayu cloud data` can [read its files](#reading-an-agents-data-files). To copy an Agent's
files and records into another Agent, [export and import them](#export-and-import-an-agents-data).

## Export and import an Agent's data

`cayu cloud applications export` copies an Agent's PostgreSQL database and its `/data` files
into one archive in the Cayu Cloud install's own storage; `import` restores an export into
another Agent (or the same one). Use them to redeploy an Agent under a new slug or in another
Organization and keep its records, to copy production data into a staging or test Agent, to
hand an Agent's data over, or to restore an Agent's data into a fresh Agent. They need an
organization administrator signed in with `cayu cloud login`; an Organization API key is
refused with error `code` `admin_required`.

```bash
cayu cloud applications export my-agent                  # waits; prints the export and its id
cayu cloud applications exports my-agent                 # the Agent's exports, newest first
cayu cloud applications export-status my-agent EXPORT_ID

cayu cloud service destroy --application my-agent-copy   # the target must be stopped
cayu cloud applications import my-agent-copy --from EXPORT_ID
cayu cloud applications import-status my-agent-copy IMPORT_ID
cayu cloud env set OPENAI_API_KEY --secret --application my-agent-copy   # each missing secret
cayu cloud rollback RELEASE_ID --application my-agent-copy --wait   # start it (its selected Release)
```

Both take the exact Agent slug and wait until Cayu Cloud finishes (`--wait-seconds`, 3600 by
default; `--no-wait` returns once it is queued). The result is the export or import record:
its `id`, `status` (`queued`, `running`, `succeeded` or `failed`) and, when it failed,
`error.code` and `error.message`. A failed one exits `2` with category
`agent_data_export_failed` or `agent_data_import_failed`, Cloud's `code` and the record; a
wait that runs out exits `2` with category `agent_data_transfer_still_running` and the
`export-status` or `import-status` command to read it with, and Cayu Cloud carries on.

- An export dumps the database as the Agent's own database role and archives `/data` from a
  one-off process of the Agent itself. Secret values never leave Cayu Cloud: the export
  records only the secret names (`secret_names`). An export of a running Agent is consistent
  within its database, but files written meanwhile may be newer or older than the dump. For an
  exact copy, stop it first with `cayu cloud service destroy`; `service pause` leaves its web
  process and runs already started writing. An archived Agent can't be exported yet
  (`application_archived`).
- An import needs a stopped target: deploy it once, then `cayu cloud service destroy`
  (`target_not_stopped` otherwise). A target whose database already holds records, or whose
  `/data` holds files, is refused with `target_has_data` and left unchanged; `--overwrite`
  replaces both. Its `missing_secrets` lists the export's secret names the target doesn't set
  yet. Set them, then start the target by deploying or rolling back; its next publication
  checks and migrates the restored database. An import interrupted part way fails with
  `import_interrupted`; run it again with `--overwrite`.
- Only one export or import of an Agent runs at a time (`agent_data_transfer_in_progress`).
  `/data` may be at most 5 GiB compressed.

A Cayu platform administrator works on any Organization's Agents by adding
`--organization ORG_ID`, and may import any Organization's export into them:

```bash
cayu cloud applications export my-agent --organization org_source
cayu cloud applications import my-agent --organization org_target --from EXPORT_ID
```

### What a deploy uploads

`cayu cloud deploy .` uploads the project the way Git sees it: tracked files
plus untracked files `.gitignore` doesn't exclude, so `.gitignore` decides what
ships. Outside a Git repository it uploads every file. Either way it never
uploads `.git`, `.hg`, `.svn`, `.venv`, `venv`, `__pycache__`, `.mypy_cache`,
`.pytest_cache`, `.ruff_cache`, `build`, `dist` or `node_modules` directories,
`.DS_Store`, `*.pyc` and `*.pyo` files, `.env`, or any other `.env.*` file except
`.env.example`, `.env.sample` and `.env.template`. Secrets belong in
`cayu cloud env set NAME --secret`, never in the source.

The compressed upload may be at most 250 MiB; the CLI refuses a larger one with
`source_too_large` before uploading anything. `pyproject.toml` and `uv.lock` must
be at the root of the upload, and Git must not ignore an untracked `uv.lock`.
Files listed in `[tool.cayu.improvement] evaluator_only_paths` are uploaded for
evaluation but kept out of the image the Agent runs.

### The release smoke test

Before a Release can serve, Cloud starts each distinct process command (the
web, the worker and every schedule, or the `entrypoint` when there are none) in
an isolated sandbox with no network access and no credentials. A command passes
if it is still running after 2 seconds, or has exited with status 0 by then.
So a process that exits with an error at startup because it can't reach the
network, its database or a secret fails the Release at the `smoke_test` stage; a
schedule that finishes quickly and exits 0 passes. The same sandbox also proves
that the image can't reach the network directly or read cloud credentials.
After promotion, the web process must also pass its readiness probe.

### Approvals in Cayu Cloud

Cayu Cloud Home lists what an Agent is waiting on a person for (approvals,
questions and recoveries), read from the Agent's own inbox, which `cayu serve`
serves from Cayu 0.14.0. Cloud reads it through the web process, so an Agent
without a running `[web]`, such as a worker-only Agent, shows nothing there.
Deciding from Cloud needs `[sign_in] mode = "cloud"`, so the Agent knows who
decided; Cloud can't decide for an Agent on its own sign-in
(`agent_sign_in_required`), so people decide in the Agent itself. A sleeping
Agent is read only while one of its sessions waits for a person.

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

`service status` also returns `result.task_health`, the latest count Cloud read from the
Agent's [activity contract](#runs-and-human-touches) of tasks fenced after their worker
was lost, such as by a deploy during a task that can't be restarted:
`{"fenced_after_worker_loss": 2, "complete": true, "read_at": "..."}`. The count stops at
100 (`complete` is false then), and `read_at` is when Cloud read it. It is `null` until
Cloud has read one, and an older Cayu Cloud leaves it out. While the count is above zero
the output adds a `warnings` entry saying how to list the tasks (`cayu inspect
--fenced-tasks`) and settle them ([when a worker dies
mid-task](triggering-runs.md#when-a-worker-dies-mid-task)). The service status itself is
unchanged: fenced tasks don't make it `degraded`.

`service status` also returns `result.worker_exits`: Cloud's record of the Agent's worker
instances that stopped (an ECS task ARN or a Kubernetes container ID as `instance`, its
`state`, `stopped_at`, `reason` and `exit_code`, newest first, at most 50), with
`looked_at` saying when Cloud last asked the platform, and `status` `read`. It's the same
record the Agent reads at `CAYU_WORKER_EXITS_URL`, so you can confirm that a replaced or
drained worker really exited. A Cayu Cloud that doesn't serve it yet gives `status`
`unsupported` with a message, and a read that fails gives `unavailable`; neither fails the
status. `worker_drains` lists the replaced workers still finishing their tasks
([draining](#draining-the-worker-a-deploy-replaces)), each also as a `warnings` line with
its deadline and the time left.

By default, `deploy` returns only once the Agent's service runs the new release: `service
status` reports the release's `deployment_id` and `running`. Promotion comes first and
selects the release, so `applications list` shows it as current before that: Cloud then
migrates the Agent's database and updates the service, and meanwhile refuses service
changes and schedule runs ([while the service switches](#while-the-service-switches)). A
`degraded` service ends that wait with a nonzero `service_degraded` result. Its
`error.issues` array preserves every structured process diagnostic, while `error.message`
aggregates their safe messages and remediation hints for humans. `--no-wait-live` returns
once the release is promoted, and `--no-wait` once Cloud accepts it, without following
it; `cayu cloud deployment wait DEPLOYMENT_ID --live` waits for the rest later.

The deployment worker publishes the promoted release once. `deploy` reads the service
and waits for its `deployment_id` to match that release; it does not start another
rollout. A brief 404 while publication is pending is retried as a read, and up to five
consecutive throttled, unavailable or HTTP 5xx service reads are retried before the wait
fails. While the service still runs an older release or none, `deploy` also reads the
release: if Cloud reports a `publication_error` (for example a failed database
migration, or unfinished sessions of the serving release), it exits `2` with category
`service_publication_failed`, Cloud's message and the way past it (for unfinished
sessions, the `deployment retry` commands that acknowledge them or wait for them again;
see [Unfinished sessions](#unfinished-sessions)), and the structured
`error.publication_error`. Cloud doesn't change the status of a release it refused to
publish: one the promotion selected first stays `promoted` and selected while the
previous release keeps serving. If the service still runs an older release when
`--wait-seconds` pass, `deploy` exits `2` with category `service_still_starting`. With
`--no-wait` or `--no-wait-live`, `result.service_publication_pending=true` identifies an
absent service or one still serving an older release. `rollback` still asks Cloud to
publish the selected release and returns once Cloud accepts it; `rollback --wait` (with
`--poll-seconds` and `--wait-seconds`) also waits for the service to run that release and
reports a `publication_error` the same way. `deployment wait` reports a ready release's
`publication_error` the same way instead of returning it as ready. While Cloud holds a
release's publication for the serving release's unfinished sessions, these waits keep
going past `--wait-seconds` (see [Unfinished sessions](#unfinished-sessions)).

`deployment wait DEPLOYMENT_ID` returns once the release is ready to promote, or promoted;
with `--live` it then waits, like `deploy`, until the service runs the release, and prints
the service as `service` next to the release. Each wait has its own `--wait-seconds`, as
`deploy`'s do. A staged release isn't waited for (category `release_not_promoted`: Cloud
publishes it only once someone promotes it), nor is a promoted release the Agent no longer
selects (category `release_not_selected`, with `commands.rollback` to select it again).
A release the service already runs counts while its web process sleeps, and a release
that declares no web, worker or schedule has no service, so `--live` returns it once it
is ready, as `deploy` does.

<a id="while-the-service-switches"></a>

While Cloud migrates the Agent's database and updates its service, it refuses `service
restart`, `sleep`, `wake`, `pause`, `resume` and `destroy`, and `schedule run`, with HTTP
409 and "The Agent's service is being deployed or changed right now; nothing changed. Try
again once that finishes." Each of these commands takes `--wait`, which sends the same
request again every `--poll-seconds` (5 by default; `destroy`'s 2) until Cloud takes it,
writing one line to standard error the first time. The refusal comes before Cloud changes
anything, so sending it again is safe. If Cloud still refuses once `--wait-seconds` pass
(1,800 by default; `destroy`'s 180, which also bounds its wait for the teardown), counted
from the first refusal, the command exits `2` with category `service_busy`,
`waited_seconds` and a `service status` command. Any other refusal is never sent again.
Without `--wait`, the error says to pass it. So a script needs no loop between a deploy
and the next change, even when something else is changing the service too:

```bash
cayu cloud deploy . && cayu cloud schedule run scout --application my-agent --wait
```

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

Cloud keeps one immutable Release per source, so a deploy of unchanged source builds
nothing new: Cloud answers it with the Release it already has. The deploy says so on
standard error before it waits on anything, naming that Release, and reports it as
`result.unchanged_source` (its `deployment_id`, the `status` Cloud answered with, and the
same `message`). A Release still building is followed to the end like a new one; to build
a new Release instead, change the source or `version` in `cayu-cloud.toml`.

When unchanged source resolves to a Release that Cloud promoted earlier, no promotion
publishes it again, so the deploy checks the Agent itself. If the Agent's service already
runs that Release, the deploy returns at once, also while its web process sleeps. If that
Release is still the Agent's selected Release but no service is running it, for example
after `service destroy`, the deploy asks Cloud to start the service, as `rollback` does, and
reports `service_publication_requested: true`. If the Agent has selected a different
Release, or none, the deploy exits `2` at once with category `release_not_selected`, the
`current_deployment_id`, and ready-to-run `rollback` and `service status` commands. Select
the reused Release with `rollback`, or change `version` in `cayu-cloud.toml` to build a new
one.

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

Authentication selection is explicit: a successful `cayu cloud login` to production clears
the persisted private-context selection (a login to [another install](#signing-in-to-another-install)
selects that install's context instead), while `cayu cloud context use PATH` selects that
context for later commands. Clearing the context falls back to the saved WorkOS login
and the fixed production endpoint. A saved login for any other endpoint is rejected;
run `cayu cloud login` again to sign in to production.

Private contexts and API keys remain the internal noninteractive path for CI,
operational handoffs, and automation:

```console
cayu cloud context use /private/path/cloud-context.json
CAYU_CLOUD_API_KEY_FILE=/private/path/key cayu cloud doctor
```

## Signing in to another install

A staging or dedicated Cayu Cloud install has its own WorkOS sign-in. Sign in to it as yourself
with its address, the one its portal opens at:

```console
cayu cloud login --api-url https://cloud.example.com --no-browser
```

The login reads that install's own WorkOS client from its `GET /v1/portal-config` and runs the
same device sign-in. It keeps the install's login apart from the production one, in a directory
of its own beside the context selection (`~/.config/cayu-cloud/installs/<host>-<digest>/`
by default, under `CAYU_CLOUD_CONFIG`'s directory when that is set). There it writes the
install's context, `context.json`, which names the install's `api_url` and its `login_file`
instead of an `api_key_file`, and selects it, so later commands act on that install;
`result.context` names it. Select it again later with `cayu cloud context use PATH`, and go
back to production with `cayu cloud context clear` or `cayu cloud login`. A login is used only
for the Cloud it was made for: if the context's `api_url` and its login's differ, commands exit
`2` with `login_api_mismatch`, and a refreshed login whose Cloud, WorkOS client, Organization
or user changed is never sent. `--api-url` and an explicit `--context` or `CAYU_CLOUD_CONTEXT`
for another Cloud are refused together. The URL is compared in one spelling (lowercase scheme
and host, the host in IDNA, no default port or trailing slash), so `https://Cloud.cayu.dev:443/`
is production. The directories Cayu creates for it are private to the user (mode `0700`).

`cayu cloud logout` and `organizations use` act on the login of the selected install context,
and `context show` reports its `authentication` as `workos`. `logout` reports the `api_url` and
the `login` file it removed. It always removes an install's own `installs/<install>/auth.json`,
even a corrupt one, and any other `login_file` a context names only when it holds a login for
that install. If the persisted selection can't be read, for example because its file was
deleted, both act on the production login and say so in `warnings`, naming the unreadable file
and any install login it leaves behind. A context named with `--context` or
`CAYU_CLOUD_CONTEXT` that can't be read is an error, and nothing is removed.

When an install refuses CLI sign-in, the error says where. An address that doesn't serve the
sign-in settings, such as an install's bare API host, exits `2` with
`portal_config_unavailable` and the HTTP status. When WorkOS refuses to start the device sign-in,
the error is `device_authorization_refused` with WorkOS's HTTP status and error code: check that
the install's WorkOS environment allows device authorization (CLI sign-in) for the client named
in the message.

## Organizations

A person can belong to several Cayu Cloud Organizations; a login acts in one of them.

```console
cayu cloud organizations
cayu cloud organizations use org_01...
cayu cloud login --organization org_01...
```

`organizations` lists yours, with your role in each, and `current_organization_id` names the one
this login acts in. `organizations use ORG_ID` makes the saved login act in another of them: the
refresh token is exchanged for one in that Organization, with no new browser sign-in, and Cloud
confirms it (`GET /v1/me`). An Organization that requires its own sign-in (for example SSO), or
one you aren't in, is refused with `organization_unavailable` and the login stays as it was.
`login --organization ORG_ID` signs in and then switches to it, whichever Organization the browser
chose; nothing is saved until the switch succeeds, so a refused Organization leaves the previous
login and context in use. Every later command acts in the chosen Organization; `cayu cloud whoami` shows it. An
Organization API key belongs to one Organization, so these commands need `cayu cloud login`.
With an install's context selected (`login --api-url`), they switch that install's login, and
its context stays selected.

### Where an Agent deploys

A command that changes something first says, on standard error, which Organization it acts in
and on which Cayu Cloud, for example
`cayu cloud: acting in Organization Cayu Demo (org_01...) on https://cloud.cayu.dev`, and its JSON
output carries the same as `organization` (`api_url`, `id`, and `name`, which is null for an
Organization API key). It asks Cloud (`GET /v1/me`) before it changes anything. These commands
are `deploy`, `applications archive`, `deployment promote|discard|retry`, `rollback`,
`service destroy|pause|redeploy|restart|resume|sleep|wake`, `env set|unset`,
`settings set|unset`, `schedule run`, `compass enroll|disable|verify|pass`,
`compass findings accept|dismiss`, `eval run|baseline`, `eval runs cancel`,
`api-keys create|revoke` and `connections add|authorize|agents|rotate|remove|bind|unbind`. Commands that
only read say nothing.

Each Agent also remembers where it deploys. The first successful `cayu cloud deploy` of an Agent
from this machine records its Cloud and Organization in
`~/.config/cayu-cloud/agents/<slug>.json` (beside the context selection, so under
`CAYU_CLOUD_CONFIG`'s directory when that is set), and says so once in `warnings`; an Agent
deployed before this record existed gets it on its next successful deploy the same way. From
then on, a command above that changes that Agent is refused with `agent_organization_mismatch`
when the login or context in use acts in another Organization or on another Cloud, before
anything is changed. The command resolves its Agent reference (a slug, a name, or the slug of a
name) as it always does, and the Agent it resolves to is the one checked and then changed; an Agent that isn't
found where the command acts is explained by its record when this machine has one. The error
names both Organizations, and how to switch: `cayu cloud organizations use ORG_ID` (or `cayu
cloud login --organization ORG_ID`) for another Organization of the same Cloud, or `cayu cloud
login --api-url URL --organization ORG_ID` for another install. This keeps a deploy meant for
staging from going to production only because no `--context` was given.

To act on the Agent where you are anyway, name that Organization with `--organization ORG_ID`,
either right after `cayu cloud` or after the command. It must be the Organization the command
acts in: any other is refused with `organization_mismatch`, so `--organization` also works on its
own as a check that a command, a read or `doctor` included, runs where you mean it to. A `deploy` with it
moves the record to the new Organization and says so in `warnings`; the other commands act once
and leave the record as it was. The follow-up commands a command prints (`deployment status`,
`rollback`, `eval baseline` and so on) carry its `--organization`, like `--context`. A record
that can't be read (`agent_record_invalid`) names its file; delete it and the next deploy records
it again. An Organization name with control characters is never printed; its id names it
instead. Records are kept per machine and per user, so CI and a teammate's laptop keep their own.

## Cloud rollouts

`cayu cloud platform` says what Cayu Cloud itself runs, so "did Cloud change under me?" has an
answer from the terminal. For each component (the API, scheduler, deployment, evaluation and
studio workers, the sandbox broker and the Gateway) it prints the source `revision` it serves,
`serving_since` (when that revision rolled out), and its recent `rollouts` and `restarts`,
newest first with their times and pods. A restart is a start on the same revision: a pod
rescheduled or replaced, which still ends what that pod held (a sandbox broker restart is how
brokered sandboxes get handed over). Cloud keeps 90 days of starts. `cayu cloud doctor` includes
each component's revision and `serving_since`, or `null` from a Cloud that doesn't report them.

## Candidate releases

`cayu cloud deploy --no-promote` stages a candidate release: Cayu Cloud builds and smoke-tests
it and stops there, so nothing about the Agent changes. Use it for a change someone evaluated,
for example a Compound candidate a person approved, that a person should publish later.

> **Changed:** `--no-promote` used to skip only the CLI's own promote request. Cayu Cloud
> selects and publishes every smoke-tested release itself, so the release was published
> anyway. It now stages a candidate release, which needs a Cayu Cloud with candidate
> releases and names the release the change was evaluated against (`--evaluated-against`).
> An older Cayu Cloud refuses the request: the command exits `2` with category
> `candidate_unsupported`, and nothing is deployed. Deploy without `--no-promote` to publish
> a release.

```console
cayu cloud deploy . --no-promote --evaluated-against dep_live \
  --snapshot-root 5f0c...e1 --brief brief-monthend-cost --requirement req-1 \
  --receipt evaluation-7c1d=a3b9...04
cayu cloud deployment status DEPLOYMENT_ID --application my-agent
cayu cloud deployment promote DEPLOYMENT_ID --application my-agent
cayu cloud deployment discard DEPLOYMENT_ID --application my-agent
```

- `--evaluated-against` (required) names the release the change was evaluated against, or
  `none` when the Agent had no selected release. The CLI never fills it in, so Cloud records
  only an evaluation that happened. Cloud refuses the candidate, and later its promotion,
  while another release is selected, so a change is never published over a release it
  wasn't compared with.
- `--expires-in-seconds` (3600 to 2592000) bounds how long it can be promoted; Cloud's default
  is 604800 (seven days). Cloud then discards it.
- `--snapshot-root` (the evaluated Cayu `AgentSnapshot` root, lowercase hex SHA-256),
  `--brief`, `--requirement` and `--receipt RECEIPT_ID=FINGERPRINT` (both repeatable) record
  the evaluation. Cloud returns them with the release, also after promotion, and never deploys
  what they name: the release is built only from this source, so the evaluation's trial
  history and knowledge stay out of the Agent.

These flags are refused without `--no-promote`. The deploy waits until the candidate is
smoke-tested (or returns at once with `--no-wait`) and reports `result.deployment`, whose
`candidate` holds the statement and `expires_at`. Nothing it does publishes the candidate: it
never promotes it, never retries its publication and never starts its service, whatever
state Cloud reports.

Running it again with the same source and flags replays the same candidate, also after a
client timeout. When Cloud answers that the source already has a release, for example
because the acknowledgement or session flags differ, the deploy takes that release up only
if it is a candidate with the same baseline, evaluation and expiry; otherwise it exits `2`
with category `candidate_conflict` naming the release, and nothing is promoted. An
acknowledgement or session choice the candidate found again doesn't carry is reported in
`result.not_applied`; as for any deploy, a kept session choice more permissive than the one
given makes it exit `2` with category `session_choice_not_applied`. A plain `deploy` given
`--acknowledge-breaking` or a session flag, which looks for an earlier release of its source,
stops with `candidate_conflict` when that release is a candidate, with the `deployment
promote` and `deployment discard` commands for it, because only a person's promotion
publishes a candidate. Without those flags it reports Cloud's version conflict (HTTP 409),
as before.

The deploy exits `2` without promoting anything when the candidate can't be promoted:

- `candidate_baseline_changed`: the Agent no longer selects the release the candidate was
  evaluated against, whether Cloud refuses to stage it or the deploy finds it again.
- `candidate_discarded`: the candidate was discarded or expired, before this deploy or while
  it waited, so it can't be published or retried.
- `candidate_invalid`: Cloud rejected the statement under rules newer than this CLI's.
- `candidate_unsupported`: the Cayu Cloud predates candidate releases.

In each case, stage the change again from changed source or a new `version` in
`cayu-cloud.toml` once it can be promoted.

`deployment promote` publishes a candidate: Cloud's deployment worker checks the serving
release's [unfinished sessions](#unfinished-sessions), prepares the database, selects the
candidate and publishes it, so the command returns the Agent before its selection changes.
Follow it with `deployment timeline` or `service status`. A repeated promote does not publish
it twice. Promoting an expired candidate, or one whose evaluated-against release is no longer
selected, fails with HTTP 409, Cloud's reason and `error.code` `candidate_expired` or
`candidate_baseline_changed`. `deployment discard` cancels a candidate so
it is never published, and refuses any other release with category `not_a_candidate`. While
Cloud publishes a promoted candidate, discard fails with HTTP 409 and Cloud's reason; once
that publication fails or is refused, the candidate can be discarded.

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
release's unfinished Cayu sessions through the Agent's own control plane API. A session
that is mid-interaction, or paused on a tool approval, user input or another pending
action, may not resume on the new release, so it blocks the replacement. A session interrupted with
nothing pending can adopt the new release at its next interaction and doesn't block.
Agents without the Cayu control plane API are published as before.

An Agent can declare standing consent for exact approval pairs in its root
`cayu-cloud.toml` (schema version 2):

```toml
[[session_replacement.approvals]]
agent_name = "issue_filer"
tool_name = "file_issue"
reason = "The application can raise a new approval request if continuation is lost."
```

This accepts possible loss of continuation; it does not guarantee recovery or replay
an approval. Cloud uses the declaration admitted from the **serving release's source**.
A new release cannot authorize replacing the old release retroactively. The first
adoption may therefore need an explicit acknowledgement of its blocking sessions.
Removing a declaration takes effect once that release is serving.

The table accepts only `approvals`, with 1 to 20 entries containing exactly
`agent_name`, `tool_name`, and `reason`. Agent/tool pairs must be unique. Names are
exact, case-sensitive, printable strings of 1 to 128 characters, without surrounding
whitespace or `*`; they are not patterns. Reasons are printable strings of 1 to 512
characters without surrounding whitespace. Unicode is retained without normalization.
Invalid or unknown fields are rejected. `cloud init` declares no consent; `cloud init
--force` preserves an existing valid declaration in canonical pair order while
regenerating unrelated descriptor fields, even if those fields are invalid. Invalid
consent or unreadable TOML stops initialization before any files change; repair the
file or explicitly remove it to regenerate without preserving its consent.

Without explicit session flags, Cloud may apply this consent to matching
approval-paused sessions. Other pending action kinds, unmatched approvals, unreadable
session evidence, active interactions and held tasks retain their existing protection.
Consent does not bypass migration acknowledgements, worker stopping or storage safety.
Cloud records accepted replacement consent as `session_preflight.state = "consented"`
when it permits publication. An explicit `wait` or `block` overrides standing consent.
The declaration travels only in the root source file, never as a CLI-generated
`proceed` choice or wildcard acknowledgement; `--manifest` cannot select another file
containing it or hide a root declaration.

You choose what Cloud does about blocking sessions, per release:

- `wait` (the default) holds the publication until no session blocks, for at most
  `--session-wait-seconds` (60 to 3,600; Cloud's default is 900), then refuses. The
  previous release keeps serving while it waits.
- `block` refuses at once.
- `proceed` publishes once every blocking session is acknowledged with
  `--acknowledge-session SESSION_ID`. The flag can be repeated, and `'*'` acknowledges
  every unfinished session, including ones Cloud could not read. It implies
  `--session-policy proceed`.

The same check lists the tasks the serving release's worker has claimed or is running
(`session_preflight.tasks`). Replacing the worker interrupts them, so each blocks like a
running session: `wait` waits for it to finish, `block` refuses, and `proceed` publishes
only when `--acknowledge-session` names it by its task ID (or `'*'`). The release then
records the tasks it interrupted as `session_preflight.interrupted_tasks`. A worker that
drains ([above](#draining-the-worker-a-deploy-replaces)) finishes them instead.

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
`wait` choice with any applicable serving-source consent, and a retry or rollback keeps
the release's current choice. Every explicitly supplied choice, including
`--session-policy wait`, is sent to Cloud and included in the deploy's idempotency key.
Omission and explicit default wait have distinct identities because only omission
allows standing consent. Cloud's returned `session_policy.source = "explicit"` confirms
that a default wait choice came from an operation rather than the implicit default.

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
the blocking sessions), `held_tasks` (only tasks the worker holds block) or
`sessions_unreadable` (it could not read them all). For `held_tasks`, `error.commands.retry`
acknowledges the tasks by ID, and the message says to let them finish first.
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
the release once nothing blocks. A refused rollback or promotion that had already selected
the release (as `deploy`'s promote can) returns the Agent's selection to the serving
release. The release keeps `promoted`, and only its `publication_error` says it was
refused. When Cloud checked first, the selection never changed. Either way, the retry
publishes it: Cloud checks the sessions again, prepares the release and selects it.

`deployment retry` returns as soon as Cloud has queued the retry, before any of that,
and the release's status (`promoted`, say) doesn't mean the Agent's service runs it. So
the command writes to standard error, and returns as `message` and `commands.wait`, the
command that follows the release until the service runs it (for a failed release's
retry, the new attempt Cloud answered with):

```bash
cayu cloud deployment wait DEPLOYMENT_ID --application AGENT_SLUG --live
```

A staged release's retry that is still building (a failed `deploy --no-promote`
release's new attempt) is published only once someone promotes it, so for that one
`commands.wait` waits for the build without `--live` and `commands.promote` is the
promotion that publishes it.

Promoting the release again doesn't help: it was already promoted, and Cloud answers with
what applies instead.

Running `cayu cloud deploy` again with unchanged source and any of the three flags (even
`--session-policy wait` alone, which has its own create request and key) gives the
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
promote may race Cloud's own, `--no-wait` may stop the deploy first, or
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

<a id="agent-operator-credentials"></a>

## Agent Developer dashboard credentials

Cayu Cloud gives every Agent its own login for its `/cayu/` Developer dashboard and injects it
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

## Redeploying an Agent's service

A service keeps the bindings it was published with: the variables and credentials Cloud gave
its processes then. When the install changes what it binds, for example when it turns on its
Gateway after the Agent was deployed, `service status` shows the issue `gateway_key_unbound`
("The Agent runs without this install's Gateway"). `service restart` replays the old bindings,
and a `deploy` of unchanged source reuses the Release without publishing it again. Publish the
selected Release again instead:

```console
cayu cloud service redeploy --application my-agent
```

`service redeploy` asks Cloud for a forced publication of the Agent's selected Release
(`PUT /v1/applications/<agent>/service?force=true`), as the portal's **Redeploy service**
does. Cloud queues it and its deployment worker publishes it shortly after, with the bindings
it gives Agents now, so the command returns at once: `status` is `queued`, `result` is Cloud's
answer (the service as `deploying`), and `commands.status` is the `service status` command to
watch the service until it runs again, with the issue gone. Cloud replaces the service's
processes as it does for a deploy.

An Agent with no selected Release exits `2` with `no_selected_release`, one deleted meanwhile
with `application_not_found`, and Cloud's other refusals (such as for an archived Agent) are
reported as for any request. A Release that runs no web process, worker or schedule gets a
warning that nothing will run.

## Reading an Agent's state from the CLI

Read commands that inspect durable state take `--application NAME` to read an Agent on
Cayu Cloud instead of a local store, for example `cayu task list --application my-agent`
([inspecting tasks](project-workers.md#inspect-tasks)) and `cayu knowledge list
--application my-agent` ([kept knowledge](control-plane.md#kept-knowledge)). The CLI signs in as `cayu cloud`
does (`cayu cloud login`, `--context PATH`, or `CAYU_CLOUD_API_KEY`), asks Cloud for the
Agent's Cayu address and its [Developer dashboard login](#agent-developer-dashboard-credentials),
and reads the Agent's own Cayu API with that login, so the same people who can read the
login can use these commands. It reads `/cayu/api/` first and `/api/` when the Agent runs
`cayu serve`. A sleeping Agent is woken by the read, and the command waits for it: it says
once on stderr that the Agent is waking and for how long it waits (`--wake-timeout-seconds`,
default 180), then answers. `--no-wait` fails at once with `starting` instead (the wake still
starts, so a later command finds it awake); `cayu task wait --no-wait` fails too rather than
reading through the wake. The `cayu eval` commands that take `--application` take both
options as well. `--timeout-seconds` stays the bound on each request. Nothing is stored
locally.

A failure names its cause in `error.category`: `not_serving` when the Agent's service is
stopped or failing, `no_address` when it serves no Cayu address (an Agent without a `[web]`
process), `credentials_rejected` when the Agent doesn't accept its Developer dashboard
login (HTTP 401), `forbidden` when it accepts the login but refuses the request (HTTP 403,
with the route's own reason and, when it gives one, its code, such as `role_forbidden`),
`not_served` when it doesn't serve Cayu's API, and `unsupported` when its Cayu is older
than the command needs. A refused `cayu task` or `cayu knowledge` action prints the
route's reason too, and its JSON error carries `category`, `status_code`, `detail` and
`code`.

`cayu session list|show|interactions|usage|tools|events|transcript|output --application
my-agent` reads the Agent's sessions the same way; see
[Session inspection](session-inspection.md#sessions-of-a-cloud-agent). `cayu task wait
TASK_ID --application my-agent` waits for one of its tasks to end.

`cayu task show` and `cayu task wait` print one field of the task with `--field PATH`, a
dotted path into the task as `--json` shows it under `task` (a number picks a list item).
The value is printed as JSON; add `--raw` to print a text value as plain text, such as a
Markdown brief a task returned:

```console
cayu task show TASK_ID --application cayu-herald --field result.markdown --raw
```

A path the task doesn't have fails and names the first part that is missing.

### Reading a deployed Agent from the project's own commands

A project's own commands can read the deployed Agent through the same sign-in, with no
routes or client of their own. `cayu.storage.open_cloud_agent_readers(APPLICATION)` returns
read-only task and session readers for the Cloud Agent, and
`cayu.storage.open_store_agent_readers(target)` the same readers for a local store, so one
command reads either:

```python
from cayu.storage import TaskListRequest, open_cloud_agent_readers
from cayu.tasks.records import TaskStatus

async with open_cloud_agent_readers("cayu-herald") as agent:
    rows = await agent.tasks.list_tasks(
        TaskListRequest(type="digest", status=TaskStatus.COMPLETED, limit=1)
    )
    digest = await agent.tasks.load_task(rows[0]["id"]) if rows else None
```

`agent.tasks.list_tasks` lists newest-updated first without results (`TaskListRequest`
takes `type` for one task type or `types` for several), and `load_task` returns one task with its `result`, in the shape of the Cayu server's task API (what
`cayu task show --json` prints under `task`). `agent.sessions` reads sessions as `cayu session
--application` does, for example `query_events(EventQuery(...))` and
`query_transcript(TranscriptQuery(...))`. The open function takes `context=` for
`--context`, `timeout_seconds=` and `wake_timeout_seconds=`; a read that can't be answered
raises `cayu.storage.AgentReadError`, whose `category` is one of those above. The readers
only read.

## Reading an Agent's logs

`cayu cloud service logs --application AGENT` prints the output of the Agent's web process,
worker, schedules, database migrations and `/data` audit, newest lines last, each with its time
(UTC), process and release:

```text
$ cayu cloud service logs --application my-agent --process worker --since 2026-10-28T21:00-07:00 --grep Error
2026-10-29T04:06:00.123Z worker dep_6c1f0e2a9b8d4c7e8f0a1b2c3d4e5f60 AttributeError: 'list' object has no attribute 'items'
```

- `--process web|worker|migrate|storage-audit|schedule:NAME` keeps one process's lines; all of
  them by default. Lines of a Release deployed before Cloud named each process's logs have no
  process (`-`) and are left out by `--process`; deploying again names them.
- `--release DEPLOYMENT_ID` keeps one Release's lines, and with `--process` reads only that
  Release's process, which is quicker on a busy Agent.
- `--since` and `--until` take a span back from now (`90s`, `15m`, `1h`, `2d`) or an ISO 8601
  time; one without an offset is local. A read spans at most 7 days.
- `--grep TEXT` keeps lines containing the text, plainly and case-sensitively.
- `--event NAME` keeps JSON log messages whose top-level `event` field is exactly that string.
- `--field KEY=VALUE` matches a literal top-level JSON field. Repeat it to require every field;
  it also combines with `--event` and the other filters. Values are JSON scalars or plain text:
  `--field status=ok`, `--field count=2`, `--field ready=true`, `--field reason=null`.
  Use `--field 'count="2"'` for the string `"2"` instead of the number `2`. Numbers compare
  numerically (`2` equals `2.0`); booleans, strings and null stay distinct. A missing field
  does not match null. Keys containing dots are literal keys, not nested paths.
- `--limit N` prints up to N lines (100 by default, at most 1000).

For example, to inspect a worker's structured events during a deploy window:

```bash
cayu cloud service logs --application my-agent --process worker \
  --event worker.completed --field status=ok --since 1h --until 5m
```

Structured filters inspect the JSON object in each line's redacted `message`, locally in the
CLI. Plain text, malformed JSON, JSON values that are not objects, and truncated messages do
not match. Nonstandard JSON constants such as `NaN` and `Infinity` are rejected. Matching
lines retain their original message and metadata in both readable and `--json` output.

Without `--since` the command prints the newest lines, up to 7 days back. With it, it prints the
first lines after that time, oldest first, and when more follow it ends with
`More lines follow; go on with --cursor CURSOR`: run it again with that `--cursor` (and the same
filters) for the next lines. To read a failed worker task's traceback, find the time with
`--process worker --grep Traceback`, then read from there with `--since`.

A wide search, such as a `--grep` over a day or more of a busy Agent's lines, takes Cloud more
than one request: it answers each page with what it found in a few seconds and the span it
searched. While a page comes back with fewer lines than `--limit` and more to read, the command
follows its cursor itself, for up to `--search-for` (`2m` by default; `90s`, `10m`, or `0` for
one page), and prints the lines of every page it read. When it stops before the end of the
window, it ends with `Searched FROM to TO; more may follow. Go on with --cursor CURSOR`, so the
part it covered is never in doubt; Ctrl-C stops the search early the same way.

For `--event` or `--field` without `--since` or `--cursor`, the CLI scans forward through that
seven-day window and retains the newest N matches, oldest first. The window ends at the earlier
of `--until` and the time the command starts. This may require several requests even after N
matches have been found. `--search-for` bounds this scan too; when it stops early, the output
warns that newer matches may remain, and `--json` includes the warning alongside the available
`searched`, `more` and `next_cursor` evidence. Continue with that cursor and the same filters
to read subsequent matches, or increase `--search-for` to complete a new newest-match search.
With an explicit `--since` or `--cursor`, the command collects the first N matching lines,
continuing past nonmatching pages. A backend that cannot report its searched span retains
`searched: null`; the CLI does not manufacture coverage evidence.

`--follow` prints the lines, then asks Cloud every two seconds for the lines after the last one
printed, until Ctrl-C, which ends it without an error. It shows each line once. Cloud holds back
the last 10 seconds of lines, so a line that reaches it late is still shown. With `--until`, it
stops once that time has been read.

`--json` prints the usual envelope, whose `result.lines` carry `id`, `timestamp`, `process`,
`release`, `instance` (the ECS task or Kubernetes pod) and `message`, with `next_cursor`,
`more` and `searched` (`from` inclusive, `to` exclusive, of every page read; `null` for the
newest lines). With `--follow`, it prints one envelope per page that has lines, then one with the
cursor to go on from. A partial initial structured scan also prints its warning when it has no
matches. If interrupted before a structured follow read has obtained a cursor, the final result
can instead contain `next_since`; readable output gives the corresponding `--since` command.
This includes the transition after a completed initial snapshot: continuation starts one
millisecond after its inclusive end. The normal seven-day bound applies when resuming with
`--since`. Follow otherwise keeps the original `--until`, including a future time.

Cloud redacts credential-shaped text (`[redacted]`) and removes terminal control sequences
before a line leaves it, and cuts a line longer than 8,192 characters; `--grep` matches the
redacted text. A filter Cloud can't read is refused with its reason (`log_query_invalid`). A
Cloud from before line times prints the newest lines without times or processes, and refuses
any option rather than ignore it (`service_logs_filters_unsupported`).

## Checking an Agent's database

`cayu cloud service database --application AGENT` shows where the Agent's database time goes,
for anyone who may see the Agent. It reads Cayu Cloud's database health for the Agent and
changes nothing:

```text
$ cayu cloud service database --application my-agent --window 1h
Agent database: my-agent
  Observed: 2026-11-06 18:30Z
  Window: 2026-11-06 17:30Z to 2026-11-06 18:30Z

Round trips from Cloud to the instance (direct):
  Connect: 21.4 ms
  SELECT 1 x 10: median 0.62 ms, min 0.51 ms, max 1.9 ms

Connections of the Agent's role now:
  TOTAL  ACTIVE  IDLE  IDLE IN TX  OTHER  WAITING ON LOCKS
  7      1       6     0           0      0

PgBouncer for this Agent (120 samples from 2 tasks):
  Queries: 51120 (14.2/s), average 1.1 ms
  Transactions: average 3.4 ms
  Wait for a server connection: average 0.02 ms, longest 0.0 ms
  Most in a minute: 0 clients waiting, 4 server connections

Shared db.t4g.small (every Agent of the install):
  CPU: average 18.5%, max 61.0%
  CPU credits: now 140.2, lowest 132.0
  Connections: at most 88
```

- The round trips are measured by Cloud as it answers, from its API in the Agent's network
  straight to the instance: connecting, then ten `SELECT 1`. Agents reach the database through
  PgBouncer where the install runs it, so a round trip the Agent sees as much slower than this
  is spent in PgBouncer, its load balancer, or the Agent's own CPU.
- The connections are the Agent role's connections to its database now. Through PgBouncer they
  are its server connections; the Agent's direct `LISTEN` connections count too.
- PgBouncer's figures cover the window: how many queries ran and the average query and
  transaction time it measured for the Agent's database, how long clients waited for a server connection (on average and the
  longest), and the most waiting clients and server connections in any minute.
- The instance's figures cover the window too. Every Agent of the install shares the instance;
  CPU credits apply only to burstable classes (`db.t*`), whose CPU slows down when they run out.

`--window` takes `15m` (the default) up to `6h`. A part Cloud can't read is shown as unavailable
with the reason, for example `no PgBouncer lines for this Agent in the window` for an idle Agent.
`--json` prints the JSON envelope the other `cayu cloud` commands print. Its result keeps only
the fields above, each checked (numbers stay numbers), so it never carries a host or credential.

## Reading an Agent's /data files

`cayu cloud data ls|cat|get|stat --application AGENT PATH` reads the files the Agent keeps in
its [`/data`](#files-under-data), such as a run's record after the run failed:

```console
$ cayu cloud data ls --application cayu-scout scout/runs
$ cayu cloud data cat --application cayu-scout scout/runs/RUN/run.json
$ cayu cloud data stat --application cayu-scout scout/runs/RUN/run.json
$ cayu cloud data get --application cayu-scout scout/runs/RUN
Wrote RUN.tar.gz (48211 bytes).
```

- `ls [PATH]` lists a directory in name order, 200 entries a page (`--limit` up to 1000); when
  more follow it ends with `More entries follow; go on with --after NAME`. A symbolic link is
  listed as one, not followed.
- `stat PATH` shows an entry's type, size and modification time.
- `cat PATH` prints a file of at most 16 MiB. Piped or redirected, its bytes come out as they
  are; a terminal gets UTF-8 text only, with control characters shown as `\xNN`.
- `get PATH` downloads a file of at most 256 MiB, or a directory as one `NAME.tar.gz` archive,
  to the path's own name in the current directory (`--output FILE` to choose, `--force` to
  replace an existing file). An archive holds at most 10,000 entries and 64 MiB of files,
  compresses to at most 16 MiB and is read within 20 seconds; symbolic links and special files
  are left out and named. Get a larger directory's files one by one.
- `--json` (except for `cat`) prints the JSON envelope the other `cayu cloud` commands print.

PATH is relative to `/data`, for example `scout/runs/RUN/run.json`; `ls` without one lists
`/data` itself. Absolute paths, `.` and `..` names are refused, and so is a path that leads out
of `/data` through a symbolic link: links are followed only while they stay inside it. The
commands only read. There is no `rm` or `put`: changing `/data` stays with the Agent's own code
and the web app.

The Agent's own Cayu server serves these reads, the way `cayu task --application` reads its
state ([how](#reading-an-agents-state-from-the-cli)): the CLI asks Cloud for the Agent's
Developer dashboard login, which Cloud records as an `agent_operator_credentials_read` audit
event naming who asked, and the Agent's server writes one `cayu.data_volume.read` line per read
to its web log (`cayu cloud service logs --process web --grep cayu.data_volume.read`), naming the
path, the outcome and the bytes returned, never what a file holds. So the Agent must be serving:
a failed run is readable, but an Agent whose web process is down, or one without a `[web]`
process, isn't. It also needs a release built with a Cayu that serves these reads, deployed on a
Cloud that sets `CAYU_DATA_VOLUME`; an older one fails with `unsupported`, or with a message
saying to deploy the Agent again.

## Running an Agent's evals

`cayu cloud eval` runs a suite of a portable [eval corpus](evals.md#portable-corpus-documents)
on a deployed Agent and reads its eval runs. The run executes on the Agent's own eval worker,
in the release it runs now, with its secrets, sandboxes and stores, and lands in the Agent's
eval history beside the runs its Developer dashboard starts. The commands reach the Agent as
[reading an Agent's state](#reading-an-agents-state-from-the-cli) does, with its Developer
dashboard login, and fail with `evals_unavailable` when the Agent's Cayu serves no eval API
(it predates it, or runs without a durable store).

```console
cayu cloud eval run --application my-agent --corpus evals.json --suite refund-regressions \
  --output run.json
cayu cloud eval runs list --application my-agent
cayu cloud eval runs show --application my-agent RUN_ID
cayu cloud eval runs cancel --application my-agent RUN_ID
cayu cloud eval compare --application my-agent BASELINE_RUN_ID CURRENT_RUN_ID
cayu cloud eval baseline --application my-agent RUN_ID
```

`eval run` reads the corpus file (at most 8 MiB) and checks the suite against the eval target
the corpus names, as the Agent's current release publishes it (`GET /api/evals/targets`).
Before importing anything it refuses a target the release doesn't have, naming the ones it
has; a target whose execution profile isn't ready; and a suite the target can't run: more
trials per case, cases or seconds per trial than the target allows, or a `--max-concurrency`
(default 1) above the target's limit or the suite's trial policy. There is no `--trials`: the
trial count is part of the suite's content, so a different count is a second suite in the
corpus.

It then states the run's bound on standard error, and in the JSON result as `bound`: cases ×
trials per case = total trials, how many run at once, and each trial's model-step limit and
timeout. The trials of one run execute in one process on the Agent, so past about four at
once each trial slows down; see [how many trials one process runs
well](evals.md#how-many-trials-one-process-runs-well). `--max-cost AMOUNT` (with `--currency CODE`, `USD` by default) is a per-trial
candidate cost threshold, `bound.candidate_cost_threshold`: it interrupts a trial's agent once
its observed cost reaches the amount, and one model call can cross it before its cost is known.
A target that publishes no pricing refuses `--max-cost`. It doesn't bound AI judges: when the
suite has model-judge assertions, `bound.judges` lists each judge profile the suite uses, how
many evaluations the run asks of it, and the limits its profile publishes on each one (cost,
tokens and timeout, when the target's catalog states them), and the stated bound names them.
A judge's cost limit is a stop threshold too: one judge call can exceed it before its cost is
known.

The command imports the corpus (importing the same content again changes nothing) and admits
one run under an idempotency key: `--operation-id ID`, or a fresh one. Before admitting, it
prints a continuation command that repeats the admission exactly, with that key and
`--execution-profile-revision REV`, the eval target's execution-profile revision the request
names. The Agent binds the key to the whole request, so when the command stops before it names
the run, for example because the Agent's answer was lost (`outcome_unknown`), the continuation
admits nothing new and continues with the run the first one admitted, also after a deploy has
changed the target's current revision. A continuation skips the pre-admission checks and sends
the admission first; it imports the corpus, and admits again under the same key, only when the
Agent answers that it doesn't store the corpus, so a release that would refuse the corpus now
still replays a run it admitted before. Errors carry the continuation as `commands.continue`,
and the JSON result carries `execution_profile_revision`.

When the key already admitted a different request, because the corpus file or an option changed
since, the command fails with `operation_id_conflict`: that run still exists, and
`commands.list` lists the eval target's runs. When the Agent's current release refuses the
request and the key admitted nothing, the continuation fails with `eval_release_changed`: there
is no run to continue, so start a new one without `--operation-id`.

It waits for the run and prints its progress on standard error when it changes (`Run
eval-...: running, 3/10 trials finished`), riding out reads the Agent can't answer for a
while, as during a deploy. When the run completes it downloads the run's report and prints
its status, score and trial outcomes; `--output FILE` writes the report, the JSON report the
Developer dashboard downloads and a file `cayu eval compare` reads. A run that fails or is
cancelled exits `2` with `eval_run_failed` (and the run's `failure_code`) or
`eval_run_cancelled`. `--no-wait` returns once the run is admitted. A wait that reaches
`--wait-seconds` (six hours by default, read every `--poll-seconds`, 5 by default) exits `2`
with `eval_run_still_running`, the run id and the `runs show`, `runs cancel` and `continue`
commands; the run goes on in the Agent.

`runs list` lists one eval target's runs, newest first: the Agent's default target, or
`--target-key KEY`, filtered by `--status` and limited by `--limit` (20 by default, at most
200). `runs show` shows a run's status, trial progress and result, how the candidate's run
failed in the trials of its first five failing cases, and for a scenario run each trial's phase
and the approval checkpoint a trial waits on; `runs cancel` asks the Agent to stop it. `compare`
compares two completed runs of the same suite on the Agent and
lists the current run's regressions; for a pass/fail exit in CI, compare the downloaded
reports with `cayu eval compare`. `baseline` approves a completed run's result as its suite's
baseline, the result the Developer dashboard compares later runs with, over the baseline's
current generation; approving the run that is already the baseline changes nothing. When its
answer is lost, the error's `commands.continue` repeats the same decision with
`--operation-id ID --expected-generation N`: the Agent answers the approval it already applied,
or, when the baseline has moved past generation `N` since, refuses with
`eval_baseline_conflict` instead of overriding the later approval. Each command prints text, or
the JSON envelope with `--json`.

`cayu eval suites`, `cayu eval add-session`, `cayu eval calibrate`, `cayu eval scenario` and the
same `runs` and `baseline approve` commands take `--application NAME` too, to list and open the
Agent's suites, add its sessions to them, calibrate its AI judges, check, prepare, run and
approve its production scenarios and download its runs' reports (`cayu eval runs report`); see
[Curating evals from the CLI](evals.md#curating-evals-from-the-cli).

## Signing in to an Agent

`ProxyIdentityAuth` lets an Agent trust a sign-in proxy that authenticates people
and supplies a signed identity header. Declare Cloud sign-in and public paths in
[`cayu-cloud.toml`](#sign-in-and-home-page). The deployment proxy applies those
settings; Runtime verifies the identity supplied to it.

For Cloud sign-in, the router must authenticate the person and add an
`X-Cayu-Identity` header to protected requests over HTTP, SSE and WebSocket. The
header is a JWT of at most two minutes, signed by Cloud, naming the person (`sub`),
their Organization (`org`), their role for this Agent (`role`: `admin`, `developer`
or `member`) and the Agent's public `origin`, which the Agent compares with the
browser's `Origin` on writes. The token may also carry the person's `name` and
`email`; Cayu Cloud doesn't send them yet. The router must remove any
`X-Cayu-Identity` a client supplied. Public routes must not receive a signed
identity; their access policy belongs to the Agent and proxy configuration.

Cloud sets these on every Agent that uses Cloud sign-in, and an Agent can't override
them:

| Variable | Value |
| --- | --- |
| `CAYU_PROXY_IDENTITY_ISSUER` | Cloud's sign-in issuer |
| `CAYU_PROXY_IDENTITY_AUDIENCE` | `cayu-agent:<application id>` |
| `CAYU_PROXY_IDENTITY_JWKS_URL` | Cloud's published sign-in keys, reachable from the Agent |

With them set, `cayu.server.environment_auth:OPERATOR_BASIC_AUTH`, the
`[tool.cayu.serve].auth` target `cayu new` and `cayu cloud init` write, is a
`ProxyIdentityAuth` that verifies the header and falls back to the
[Developer dashboard login](#agent-operator-credentials) when there is none. So
the Developer dashboard login, webhooks, scripts and Cloud's own reader keep
working, and a header that doesn't verify is refused rather than ignored.

On the Developer dashboard, `admin` and `developer` can do everything. A `member`
can read sessions, tasks, artifacts, pending knowledge proposals and the inbox, and
decide: approve, deny or answer approvals and questions, settle inbox items, and
approve or reject knowledge proposals. Everything else answers 403. The
[runtime contracts](runtime-contracts.md#trusted-proxy-identity) list every route a
member may use.

Members decide the existing proposal: approval and question decisions cannot carry
replacement arguments, an approval revision, or execution overrides (`max_steps`,
`limits`, `budget_limits`, `retry_policy`, `structured_output`, or `thinking`).
Admin and developer access may still supply these fields. Protected host routes
outside the control plane require their own product policy; only the packaged
browser client and dashboard receive the built-in read exception.

The Agent's own app can get the person too. Build the same dependency from the
environment, read the person with `authenticate`, or limit a route to some roles
with `product_dependency`, which also serves `AuthenticatedProductAccess`:

```python
from typing import Annotated

from fastapi import Depends

from cayu.server import AuthContext, BasicAuth, ProductPrincipal, ProxyIdentityAuth

auth = ProxyIdentityAuth.from_environment(fallback=BasicAuth.from_environment())


@app.get("/me")
async def me(person: Annotated[AuthContext, Depends(auth.authenticate)]) -> dict:
    return {"id": person.subject, "role": person.claims["role"]}


admins = auth.product_dependency(roles=("admin",))


@app.post("/close-month")
async def close_month(principal: Annotated[ProductPrincipal, Depends(admins)]) -> None: ...
```

`principal.name` and `principal.email` are the person's name and email when Cloud
has them, for display only. Cloud's `org` is the Organization the Agent was created
in, not the person's own, so an app with customers of its own keeps their
membership in its own tables.

An app with a login of its own needs no change: it never reads the header. The
header's format and the server's checks are in the
[runtime contracts](runtime-contracts.md#trusted-proxy-identity).

## Runs and human touches

An Agent reports its runs and the decisions people made on them to Cayu Cloud
through the versioned `cayu.cloud-agent` contract, which Cloud reads with a
bearer token of its own. When the Agent's environment has
`CAYU_CLOUD_READER_TOKEN_SHA256`, the SHA-256 digest of that token, its web
process serves the contract at `{api}/cloud`: `/cayu/api/cloud` when the
application mounts the control plane with `mount_cayu`, `/api/cloud` under
`cayu serve`. The Agent holds only the digest.

The token is not the Developer dashboard login. It reads three routes and nothing else:

- `GET {api}/cloud/contract` names the contract, the versions the Agent
  serves and `store_id`, the identity of the Agent's session store.
- `GET {api}/cloud/v1/activity?cursor=...` returns the summaries of the runs that
  changed after the cursor and the human touches that settled in the same range,
  and `task_health`: how many tasks are fenced after their worker was lost, such
  as by a deploy during a task that can't be restarted, and wait for the Agent
  to reconcile them.
- `GET {api}/cloud/v1/schedule` returns when the Agent next has work it
  scheduled: when its next delayed task becomes claimable or expires, how many
  tasks are scheduled or due, the earliest deadline of an external-event wait,
  and how many delivered events still wait for the Agent to service them. A
  host reads it before putting the Agent to sleep, to wake it in time, and
  backs off when a wake leaves a signal unchanged.

A run summary has the run's status, including what it waits on (a person, a
subagent, or an outside event with its source and deadline), when it started
and ended, its estimated cost, the Release it ran on, the agent and who started
it. A touch is an approval, edit, denial, expiry, answer, recovery, interruption
or steering message, with who made it, the reason they gave and whether a person
stepped in. Neither carries conversation content: no messages, tool arguments,
questions or answers. The conversation stays in the Agent and opens in its
Developer dashboard.

A reader stores the cursor of each page it accepted and continues from it, so
neither its own restart nor the Agent's leaves a gap, and it keeps the latest
summary per run and one record per touch. A cursor names the store it came
from. When the Agent's database is replaced, or restored from a backup and
written past it, the Agent answers 410 with code `activity_cursor_stale`, and
the reader reads again from the beginning. After restoring an Agent's database,
also discard the reader's cursor yourself: a restore the Agent has not written
past yet cannot be detected. Measures leave out forked sessions, such as the
copies Compass makes, even though their `root_run_id` names the run they were
copied from, and eval sessions (`eval: true`): the trials of the eval cases the
Agent runs, and everything under them. A Runtime older than the contract answers 404 there, which a
reader reports as too old for the contract. The wire format, the cursor rules
and how Cloud's four measures are computed are in the
[runtime contracts](runtime-contracts.md#cayu-cloud-agent-contract).

To read it from a local server:

```bash
token=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
export CAYU_CLOUD_READER_TOKEN_SHA256=$(printf %s "$token" | shasum -a 256 | cut -d' ' -f1)
cayu serve
curl -H "Authorization: Bearer $token" http://127.0.0.1:8000/api/cloud/v1/activity
```

### Cloud's activity reader

`cayu cloud activity status --application NAME` shows where Cloud's reader for an Agent
stands, for anyone who may see the Agent:

```bash
cayu cloud activity status --application cayu-scout
cayu cloud activity status --application cayu-scout --json
```

It prints:

- `availability`: `ready`, `catching_up` (more pages wait), `stale` (no page for five
  minutes), `not_active` (no reader activated yet), or the last read's reason code, such as
  `runtime_unavailable` for an Agent that timed out or answered with an error;
- the event position the cursor names (`after event N`, or `before the first event`) and
  when an accepted page last moved it on. A page that arrives without new activity leaves
  it where it was, so `cursor advanced` can be older than `last page`;
- the last read: whether a page arrived, timed out or failed (with its reason code), how
  long it took, how many runs it asked for, and how many runs and touches it got;
- the last ten reads that ran out Cloud's ten-second deadline;
- how many runs the next page asks for, between 10 and 200. After a page read times out,
  Cloud asks for a quarter as many runs and grows the size again as pages arrive. The size
  is the one the Cloud process that read last left; a restarted one starts at 200 again.

It prints a table by default; `--json` prints the JSON envelope the other `cayu cloud`
commands print, with `schema_version` in its result and `reader: null` for an Agent Cloud
has no reader for yet. Nothing in it is a credential: it carries no token, digest or URL,
and only the cursor's event position, never the cursor. A Cloud without the route answers
with category `activity_status_unsupported`. The command reads
`GET /v1/applications/{id}/activity/reader`; `cayu.cli._cloud_activity.read_activity_reader_status`
reads it with any signed-in `CloudApiClient`.

## Local deploy check

Cayu Cloud starts a public service (a project with `[tool.cayu] service_factory`) with
`cayu serve`, which refuses to start a production service while `cayu check --deploy`
reports a `PUBLIC_SERVICE_*` finding: placeholder or development product or Developer
dashboard access, or non-durable identity, session or task storage. The web process would exit 1
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
| `SERVE_COMMAND_INVALID` | `cayu serve` would reject the web command's arguments, for example `--dev` with `--auth`, `--as` without `--identity-proxy`, an unknown option, or an unclosed quote. |
| `SERVE_RELOAD_MODE` | The web command passes development-only `--reload`. Remove it before deploying; Cloud must not run the file watcher or restart on source edits. |
| `SERVE_DEV_MODE` | The web command passes `--dev`. `cayu serve` refuses it on a non-loopback host such as `0.0.0.0`, and on loopback it is unauthenticated and unreachable from Cloud. |
| `SERVE_IDENTITY_PROXY_MODE` | The web command passes `--identity-proxy`, the local development sign-in. Cayu Cloud's router signs people in; `cayu serve` would also refuse the proxy identity variables Cloud sets. |
| `SERVE_IDENTITY_PROXY_INVALID` | `[tool.cayu.serve.identity_proxy]` is malformed. Every `cayu serve` reads it, with or without `--identity-proxy`, so the web process would exit at startup. |
| `SERVE_AUTH_MISSING` | Neither `--auth` nor `[tool.cayu.serve].auth` names a target and there is no `service_factory`, or `[tool.cayu.serve].auth` is not a non-empty string (which `cayu serve` rejects even with `--auth`). |
| `SERVE_AUTH_TARGET_UNRESOLVABLE` | The effective target's project module or attribute doesn't exist, it isn't callable, or it is the ready-made target and the declared runtime may not ship it: the `cayu` requirement allows 0.8.1 or older, or the `[tool.uv.sources]` entry for `cayu` isn't an uploaded path source inside the project that contains the module. The message names which. |
| `SERVE_AUTH_WITH_SERVICE_FACTORY` | A public service also sets `[tool.cayu.serve].auth` or `--auth`, which `cayu serve` rejects. |

Any callable auth dependency is accepted, not only Basic auth. The ready-made
`cayu.server.environment_auth:OPERATOR_BASIC_AUTH` target is accepted by name without
reading `CAYU_OPERATOR_USERNAME` or `CAYU_OPERATOR_PASSWORD`, because Cloud provides
them, but only when the declared `cayu` requirement excludes 0.8.1 and older, or a
`[tool.uv.sources]` path source inside the project, which the deploy uploads, ships the module. The
unmodified `server_auth.py` that `cayu cloud init` writes is accepted by name for the
same reason. Any other target is imported in the project context; if it can't be loaded here
for another reason, such as an environment variable or package only Cloud has, the
check reports `unavailable` with the reason instead of failing.

`cayu cloud init` runs the same checks. When it finds a problem, for example public
service access that is still a placeholder or a `uv.lock` to refresh after init added
the `server` extra, or can't run, the result includes `deploy_check` and a `warnings`
entry describing what to fix before deploying.

After initialization, `cayu check --deploy --fail-on warning --json` also reads the
project's `cayu-cloud.toml` and includes the same `SERVE_*` findings for its
`cayu serve` web command. Ordinary `cayu check`, worker-only manifests, and other
web commands do not run this serve preflight. Tag filters still apply. If the
manifest cannot be read, or auth cannot be checked and no blocking finding is
available, the standalone command returns an invocation error instead of claiming
success. If scaffold source checks block
imports, their findings take precedence and custom auth is not imported.

When the manifest's `[worker]` command is `cayu worker NAME`, `cayu check --deploy` also
resolves how many processes it would start, as `cayu worker` does: `--processes`, then
`CAYU_WORKER_PROCESSES` in `[env]`, then `[tool.cayu.worker_processes]`, then one per
CPU, which counts as several since the CPUs Cloud gives the worker aren't known locally.
`WORKER_SINGLE_PROCESS_REQUIRED` is an error when the worker's entrypoint passes
`reconcile_abandoned_after_s` and the command doesn't run one process: the worker would
exit at startup and keep restarting. Add `--processes 1` to the command or set
`NAME = 1` under `[tool.cayu.worker_processes]`. `WORKER_COMMAND_INVALID` is an error
for a command `cayu worker` would refuse, such as an undeclared worker name. The check
reads the entrypoint's source without importing it; see
`cayu guide diagnostics#worker-single-process-required` for what it sees.
`cayu cloud deploy`'s own check doesn't run these yet.

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

Each `env set` or `env unset` call is one change, and a change restarts a live Agent.
To set several values with one restart, give them in one call. Plain values stay
`NAME=value`; each secret is a bare `NAME`, and with `--secret` every secret `NAME`
takes its own `--value-file`, in the same order as the names (at most one of them may be
`-`, standard input). Give every assignment first and the options after them; a name after
an option is refused as an unrecognized argument:

```console
cayu cloud env set \
  MODEL_ENDPOINT=https://models.example.test/v1 \
  MODEL_PROVIDER=openai \
  MODEL_API_KEY \
  --secret --value-file /private/path/model-key \
  --application my-agent
```

Cloud applies all of the values or none of them. The CLI reads every value file first,
so a missing or empty one stops the command before Cloud is asked to change anything.
Setting several values needs a Cayu Cloud with the batch environment route; an older
Cloud refuses it with `environment_change_unsupported` and nothing changes. Set the values
one per call there instead. The CLI doesn't fall back to that on its own, since it would
restart the Agent once per value and could leave only some of them set.

List or remove configuration with:

```console
cayu cloud env list --application my-agent
cayu cloud env unset VAPI_API_KEY --application my-agent
cayu cloud env unset OLD_ENDPOINT OLD_KEY --application my-agent
```

Several names in one `env unset` are likewise one change and one restart. Set and unset
responses include the Agent service rollout status when a live service is being updated.
With several names, `configuration` lists the variables set (never a secret's value) and
`unset` the names removed.

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

## Organization API keys

CI, Compass and other automated callers reach Cayu Cloud with an Organization API key instead
of a person's login. A signed-in Organization admin creates, lists and revokes them; Cloud
refuses an API key, whatever its scopes, so sign in with `cayu cloud login` first:

```console
cayu cloud api-keys create --name github-actions --purpose ci --key-file ~/.config/cayu/ci-key
cayu cloud api-keys list
cayu cloud api-keys revoke key_0123456789abcdef
```

`create` generates the key locally and writes it to `--key-file`, a new file only you can
read; it refuses a path that already exists. Cloud receives only the key's SHA-256 digest, and
neither the key nor the digest is printed. `--purpose` gives the key the narrowest scopes for
its use: `verifier` (read and report, for Compass and evaluation runners), `ci` (read and
deploy candidates), `release` (also promote, roll back and operate the service) or `developer`
(a developer's whole access). `--scope read|report|deploy|operate|developer`, repeatable, names
the scopes instead. If Cloud answers that it refused the key, the file is removed. If the
answer is lost, unreadable or a server error, Cloud may have kept the key, so the file stays:
run the same command with `--retry`, which sends the key in `--key-file` again and gets the key
Cloud kept (or keeps it now), or check with `cayu cloud api-keys list`.

To rotate a key, create its replacement, move its holder to the new key file, then revoke the
old key. A revoked key stops working at once and stays listed as inactive. The results report
each key's `id`, `name`, `scopes`, `active` and `created_at`, never the key.

## Connections

A connection is one of the Organization's model credentials or app connectors, such as an
Anthropic API key or a Snowflake key pair, under a name the Organization picks. Agents use it
through Cayu Cloud and never hold the credential. `cayu cloud connectors` lists what this
install can connect to.

```console
cayu cloud connections add anthropic --name team-anthropic --key-file ~/.config/cayu/anthropic-key
cayu cloud connections add snowflake --name warehouse --agent support-desk \
  --credential account=acme --credential user=cayu --credential private_key=@~/.keys/sf.p8
cayu cloud connections list
cayu cloud connections bind primary team-anthropic --application support-desk
cayu cloud connections bound --application support-desk
cayu cloud connections unbind primary --application support-desk
cayu cloud connections agents warehouse --all
cayu cloud connections rotate team-anthropic --key-file ~/.config/cayu/anthropic-key-2
cayu cloud connections remove warehouse
```

Credentials come from files: `--key-file` for an API key and `--credential FIELD=@PATH` for
any other field; `--credential FIELD=VALUE` is refused except for the fields that aren't secret
(`account`, `user`). `--agent` and `--application` take an Agent's id or name. No result
prints a credential. Developers and admins add connections and bind them to an Agent's slots; a
developer names the Agents a new connection is for with `--agent`, and an admin who names none
makes it available to every Agent. Only an admin changes which Agents may use one (`agents`),
rotates its credential or removes it, and Cloud refuses removing one while any Agent is bound to
it. Members see only the connections the Agents they work with use.

SAP S/4HANA comes as three connectors to one system: `sap` signs in with OAuth client
credentials, `sap-certificate` with a client certificate, and `sap-sandbox` reaches SAP's public
sandbox with an API key. Their plain fields are `base_url`, `token_url`, `client_id`,
`oauth_scope` and `sap_client`; `client_secret`, `client_certificate`, `client_key`, `ca_bundle`
and the sandbox's `api_key` come from files. Each scope is one OData service an Agent may read,
such as `API_SUPPLIERINVOICE_PROCESS_SRV.read`.

```console
cayu cloud connections add sap --name erp --scope API_SUPPLIERINVOICE_PROCESS_SRV.read \
  --credential base_url=https://my-s4.example.com --credential token_url=https://my-s4.example.com/oauth/token \
  --credential client_id=CAYU_READER --credential client_secret=@~/.keys/sap-secret
cayu cloud connections add sap-sandbox --name sap-demo --key-file ~/.keys/sap-sandbox-key
cayu cloud connections check erp
```

`connections check NAME` (admins) asks Cloud to test the connection now: that its credential
signs in, and one small read of each service it holds. It prints the connection's status, then
each service's result (`ok`, `not_authorised`, `not_found` or `failed`) with its detail; `--json`
prints the JSON envelope. A service that answers `not_authorised` usually lacks its communication
arrangement or role in SAP. When the check itself fails, it prints why in words
(`check_message`) with its code (`check_error`). Only an unreachable SAP host or token URL
(`host_unreachable`, `token_url_unreachable`) leaves the connection's status unchanged, to try
again later; a rejected credential or certificate (`auth_failed`, `certificate_rejected`) or a
host this install doesn't allow (`host_not_allowed`) sets it to `error` until an admin fixes it.

A GitHub App comes as `github-app`: its plain `app_id` (or client id, `Iv...`) and its
unencrypted `private_key` from a file, the `.pem` GitHub generated. Cloud reads the App with
the key when it is added, and its scopes are the App's repository permissions
(`issues:write`, `contents:read`, ...). A binding names the repositories the slot may call with
`--repository OWNER/NAME`, or `OWNER/*` for every repository of an owner the App is installed
on; it needs at least one, and no other connector takes them. `check` reads the App with the
kept key again.

```console
cayu cloud connections add github-app --name triage-app \
  --credential app_id=123456 --credential private_key=@~/.keys/triage-app.pem
cayu cloud connections bind github triage-app --application support-desk \
  --scope issues:write --scope pull_requests:write --scope contents:read \
  --repository acme/widgets --repository 'acme-labs/*'
cayu cloud connections check triage-app
```

A connector that signs in, such as Slack where the install has its Slack app, takes no
credential: an admin adds it, and the command prints the link where the connection is approved.

```console
cayu cloud connections add slack --name eng-slack --scope chat:write --scope channels:read
cayu cloud connections authorize eng-slack
```

Open the link in a browser where you're signed in to the Cayu Cloud portal as the same person, in
the same Organization, that started it (the admin who ran the command), and approve it; the
portal then finishes connecting it. Anyone else's portal refuses it (`sign_in_not_yours`). The connection is `pending`
until then, and `error` if the sign-in fails, with the reason in `error` (for example
`access_denied` or `state_expired`). A link works once, within 10 minutes: `authorize` prints a
new one, to finish a pending connection, retry one in error or reconnect a ready one. `--json`
prints the JSON envelope instead, with the link as the result's `authorization_url`. A connection
that signed in shows the account it connected, for Slack the workspace, as `account` (`id` and
`name`) in `list` and `show`.

An install with connections turned off doesn't serve their routes. When `cayu cloud
connectors`, `cayu cloud connections` or `cayu cloud models` gets a plain HTTP 404, it asks
Cloud's `GET /v1/connections-status` whether connections are on: when they are off, the
command fails with `connections_disabled` and says "Connections are turned off in this Cayu
Cloud installation." A Cayu Cloud without that route is older; the command then fails with
`connections_unavailable` and says connections are turned off there or it predates them.
`cayu cloud platform` shows which revision the install runs.

### Model slots

An Agent declares the models it needs as named model slots, each with a default, and the
Organization binds each slot to one of its approved models without a redeploy:

```python
app.require_model("primary", default="bedrock/claude-sonnet-4-5", label="Primary model")
app.register_agent(support_spec, model_slot="primary")
```

```console
cayu cloud models list --application support-desk
cayu cloud connections bind primary --model bedrock/claude-opus-4-1 --application support-desk
cayu cloud connections unbind primary --model --application support-desk
```

`models list` prints each slot's model, whether it is `bound`, uses its `default` or is
`unbound`, whether the model is approved and whether the running Release declares the slot, then
the approved models; `--json` prints the JSON envelope. Developers bind the slots of Agents they
can change; Cloud refuses a model that isn't approved (`model_not_approved`). The Agent reads its
bindings in the background and resolves its slot when a session starts, so a binding changes new
sessions only; resumed sessions keep their model. A slot nobody bound uses its declared default.
If the bound model is later no longer approved, new sessions are refused rather than moved to
another model. `session.started` records the slot and where its model came from (`bound`,
`default`, or `unread` when the bindings could not be read since the process started). See
[ADR 0003](adr/0003-connections-runtime-contract.md#models).

## The install's system Agents

System Agents are a Cayu Cloud install's own services, which Cloud calls directly, such as
Compound, Compass and Arena's case service. They stay on: the install's default idle timeout
never applies to their web processes. The install names each as `<organization id>/<agent
id>` in `CAYU_SYSTEM_AGENTS`; the deprecated `CAYU_AGENT_ALWAYS_ON` may still name some by a
bare Agent id. An ECS install sets both in its API's task definition, and an EKS install in the
Helm chart's `agents.systemAgents` and `agents.alwaysOn` values. A Cayu platform administrator
lists them, as the running API read them, without AWS or Kubernetes access:

```console
cayu cloud admin system-agents
cayu cloud admin system-agents --json
```

Each entry in `result.items` has its `entry`, `organization_id` (null for a bare id),
`agent_id`, `setting` (`CAYU_SYSTEM_AGENTS` or `CAYU_AGENT_ALWAYS_ON`) and `deprecated`, and as
`agent` the Agent it matches today: its `organization_id`, `lifecycle` and `recorded_system`.
`agent` is null when no Agent matches, as for a mistyped entry; a bare id matches only an active
Agent, in whichever Organization owns it. Cloud refuses anyone else, an Organization admin or an
API key included, and the command exits `2` with category `platform_admin_required`. A Cloud
that doesn't list them yet gives `system_agents_unsupported`.

An Agent's service takes a changed list at its next publish or restart. `service status`
reports `system` from the install's current setting and `recorded_system` from the service's
last publish or restart, which is what its web process's idle timeout follows. When the two
differ, the output adds a `warnings` entry saying so and naming the `service restart` command
that applies the current setting. `admin system-agents` says `not system until restarted` for
such an Agent. A Cloud from before `recorded_system` leaves it out and reports the recorded
value as `system`, and no warning is given.

## Arena case enrollments

Arena's case service, which serves an install's Evals > Cases, keeps its enrollments in its own
ledger. A Cayu platform administrator lists every one of them, across Organizations, with the
eval corpus schema each enrolled Agent declares:

```console
cayu cloud admin arena-enrollments
cayu cloud admin arena-enrollments --below-schema 6
cayu cloud admin arena-enrollments --json
```

Each row is one enrolled Arena agent: its Organization, the Cloud Agent it is bound to
(`unbound` when none), the Git commit it was enrolled from when Arena recorded one, its
enrollment (`#sequence` and fingerprint), the lowest corpus schema it declares, and where that
schema came from:

- `recorded`: Arena recorded it when the Agent enrolled;
- `current_release`: the enrollment predates that, so Cloud read the corpora the bound Agent's
  current Release declares under `[tool.cayu.improvement]` and each file's `schema_version`;
- `unknown`: neither, with the reason in brackets, such as `not_bound` or
  `source_not_uploaded` (a Release built from a connected repository).

`--below-schema N` keeps the rows whose shown schema is below `N`, with their source, and says
how many rows show no schema (unknown, or no corpus declared) and so aren't listed. Cayu 0.16.3
and later read only corpus schema 6, so `--below-schema 6` lists the Agents an Arena on a newer
Cayu would refuse with `declaration_invalid` at their next enrollment, until they re-export their
corpora. With `--json`, `result.items` holds Cloud's rows (`agent_id`, `organization_id`,
`application_id`, `bound`, `enrollment`, `arena_error` and `corpus_schema` with `source`,
`version`, `corpora`, `release` and `reason`); with `--below-schema`, `result` also has
`below_schema` and `without_schema`, the count left out.

Cloud reads them with the install's Arena inventory token, which can do nothing else in Arena.
Without it the command exits `2` with `arena_inventory_not_configured`, and the message names
the setting; `arena_inventory_unconfigured` means the Arena service has no token of its own, and
`arena_inventory_refused` that the two differ. Cloud refuses anyone but a platform administrator
(`platform_admin_required`), and a Cloud that doesn't list them yet gives
`arena_enrollments_unsupported`.

## An Agent's runtime record

Cayu Cloud keeps a runtime record for each deployed Agent. Among other things it notes which
Release last ran the Agent's own `[database] migrate` (`app_database_migration`) and which
release image last ran `cayu storage migrate` on its database (`migrated_image_digest`), so a
later publication doesn't run them again and a rollback never runs an older migration against
newer tables. When the Agent's database is restored from a backup, those notes describe tables
that are gone. A Cayu platform administrator reads the record and forgets a note, so the Agent's
next deploy runs that migration again, without AWS, Kubernetes or database access:

```console
cayu cloud admin runtime-record show my-agent --organization org_123
cayu cloud admin runtime-record show my-agent --organization org_123 --json
cayu cloud admin runtime-record forget-migration my-agent --organization org_123 --revision 17
cayu cloud admin runtime-record forget-migration my-agent --organization org_123 --revision 17 \
  --note migrated_image_digest
```

`show` prints the record's `revision`, where Cloud keeps it (`location`), the Release it runs and
its migration notes, then the `forget-migration` command for that revision, with the same
`--context` and `--api-key-file` (`result.commands.forget_migration`); `--json` prints the whole
record in `result.record`. The record names secrets by reference and holds no secret
values. `--organization` names the Organization that owns the Agent and is required.

`forget-migration` drops each `--note` (`app_database_migration` by default; repeat it for both)
and prints the record's new revision and the notes it held (`result.forgotten`); a note the
record doesn't hold changes nothing. Cloud changes the record only while it is still at
`--revision`, no publication or other service change of the Agent runs and no migration of it may
still be running, and only for an active Agent. Otherwise the command exits `2` with the code
Cloud gave, having changed nothing: `runtime_record_changed` (read the record again and decide
again), `service_busy`, `database_migration_running` or `application_archived`. An Agent never
published has no record (`runtime_record_not_found`). Deploy the Agent afterwards to run the
migration. Cloud refuses anyone but a platform administrator (`platform_admin_required`), and a
Cloud without these commands gives `runtime_records_unsupported`.

## Agent settings

Managed settings change how an Agent runs without a new deploy. The Release gives
each setting its value, and a value set in Cloud (from the portal, the API or this
command) wins until it is unset. Each setting is in one of two namespaces:
`platform` settings are how Cloud runs the Agent, such as `web.sleep` or
`web.resources`; `agent` settings are the ones the Agent's own parts declare, and
Cloud knows them once the Release has started.

```console
cayu cloud settings list --application my-agent
cayu cloud settings set web.sleep=always_on --application my-agent
cayu cloud settings set web.sleep=3600 agent.reviewer.max_steps=60 --application my-agent
cayu cloud settings unset web.sleep --application my-agent
```

`--application`, here and in every `cayu cloud` command, takes the Agent's id or its
name. The id is the `application` slug in its `cayu-cloud.toml`, and Cloud gives it to the
Agent's processes as `CAYU_CLOUD_APPLICATION_ID`, so an app that shows the command to change
one of its settings can print `--application` with that value and the command works as
shown.

`list` shows each setting's effective value, its source (`release` or `cloud`) and
its status (`applied`, `applying`, `conflict` or `refused`). A value too long for the
table is shortened between items (`owner/a, ... (+38 more)`) and printed whole
below it. For an Agent setting it
also shows how many of the Agent's processes adopted it, and lists the reasons a
process refused one. Values Cloud keeps but can't apply to the selected Release, for
example a setting the Release no longer declares, are listed as conflicts.

`set` takes one or more `KEY=VALUE` assignments and sends them as one change: either
all of them apply or none does. A value is read as JSON when it parses as JSON and as
a string otherwise, so `60` is a number, `true` a boolean,
`'{"cpu_millis": 512, "memory_mb": 1024}'` an object and `always_on` a string; write
`'"60"'` to send the string `60`. A list or map setting (see
[managed settings](managed-settings.md#lists-and-maps)) takes its text:
`app.repositories=owner/a,owner/b` or `app.merge_modes=owner/a=approve`. When
Cloud's settings carry the descriptor's `format`, `set` also takes a JSON list or
object for one, and checks every item before it writes anything, naming any item
that doesn't fit. Otherwise the Agent refuses a bad value when it reads it, and
`list` shows the reason. The command reads the current settings first to
find each key's namespace and the revision to write against. A key that both
namespaces have needs a prefix, `platform:KEY` or `agent:KEY`, and a prefixed key is
sent as written, which also reaches an Agent setting before the Release has
published its settings.

Each change carries an operation id, which the output names. If the answer to a
change is lost, run the same command with `--operation-id ID`: Cloud answers a
repeated id with its first answer instead of changing anything again.

`unset KEY` clears the value set in Cloud, so the Release's value applies again. It
also clears a value listed as a conflict.

Cloud refuses a change, and changes nothing, when:

- someone else changed the Agent's settings after the command read them (`code`
  `settings_revision_conflict`, with the current `revision`). Check them with `list`
  and run the command again;
- a value isn't valid for its setting (`code` `invalid_setting`). `errors` lists each
  setting's `namespace`, `key`, `reason` and, when Cloud offers them, the `allowed`
  values;
- the setting needs an organization admin, such as `web.sleep=always_on` (`code`
  `organization_admin_required`, with the `key`).

Unlike the other `cayu cloud` commands, `settings` prints a table, and its errors go
to standard error. Add `--json` for the same JSON envelope the other commands print:
`result` is Cloud's settings document, and `set` and `unset` add the keys they
changed and, for `set`, the `operation_id`.

These commands need a Cayu Cloud release with the settings API; an older one answers
`settings_unsupported`.

## Sandboxes for an Agent's tools

Where the install runs a sandbox broker, Cloud gives the Agent's processes
`CAYU_CLOUD_API_URL`, `CAYU_CLOUD_APPLICATION_ID` and `CAYU_CLOUD_SANDBOX_TOKEN`.
The Agent has no cloud-provider permission of its own, so its tools run code in
a sandbox Cloud allocates, leases and destroys for it:
`BrokeredSandboxCodingEnvironmentFactory` in `cayu.sandbox_broker`
([coding in a sandbox brokered by Cayu Cloud](environment-factories.md#coding-in-a-sandbox-brokered-by-cayu-cloud)).
A project made with `cayu new --preset coding` selects it automatically when
those variables are set, and Docker when they aren't
([choosing a coding backend](environment-factories.md#choosing-a-coding-backend)).
Set `CAYU_CODING_BACKEND=cayu-cloud` on the Agent to make a missing sandbox
credential fail at startup instead.

Cloud limits how many sandboxes run at once, per Agent, per Organization and
optionally per install. `SandboxBrokerClient.capacity()` reads what is free
under each limit right now, from Cloud's `GET .../sandboxes/capacity`. A create
with `queue_seconds` waits in Cloud's queue instead of being refused, and
`SandboxBrokerClient.queue_position()` says where it stands. The coding factory
takes `queue_seconds` as well and reports a waiting session's queue position in
its lifecycle progress events
([limits and capacity](environment-factories.md#coding-in-a-sandbox-brokered-by-cayu-cloud)).

### Reading an Agent's sandboxes

`cayu cloud sandboxes` reads the sandboxes Cloud brokers for an Agent, with
the `cayu cloud login` (anyone who may see the Agent) or a context's API key
with the `read` scope. It changes nothing; only the Agent creates, renews and
destroys its sandboxes.

```sh
cayu cloud sandboxes list --application my-agent                 # newest first
cayu cloud sandboxes list --application my-agent --status ready --limit 200
cayu cloud sandboxes list --application my-agent --session ses_41  # one session's
cayu cloud sandboxes list --application my-agent --task task_7 --since 2h
cayu cloud sandboxes list --application my-agent --label mission=m-17 --label role=builder
cayu cloud sandboxes show --application my-agent sbx_7f2c...      # with its events
cayu cloud sandboxes archives --application my-agent             # copy-in archives kept
```

`list` starts with the Agent's capacity now (how many places are held under
its limit, and whether a create would wait or be refused) and shows each
sandbox's status, when its lease ends, why and when it ended, and the Release,
session and task it was created for. A sandbox that still holds a place for a
Release the Agent no longer selects is marked `*`: a redeploy's leftover keeps
its place until its lease runs out. `show` adds its audit events: who asked,
each command's program and outcome, renewals, egress hosts and its end. Both
print a table, or the JSON envelope with `--json`. A sandbox created by an
older Runtime, or without provenance, shows `-` for its Release, session and
task.

`--session`, `--task`, `--label KEY=VALUE` (repeat it; all must match) and
`--since` (`15m`, `2h`, `2d` or an ISO 8601 time) list only the sandboxes
created for that session or task, with those labels, or since then. Cloud
filters and pages them, so finding one session's sandbox doesn't read the
Agent's whole history. Both commands also say what each sandbox did: how many
commands it ran, how many failed (a non-zero exit, a timeout or a command
Cloud couldn't run), how many were cancelled or still run, their run time
summed, and its file transfers in (archive uploads and staged copies in, with
their bytes and files) and out (archive downloads and staged copies out).
`list` ends with the same summed over every sandbox that matches, not only
the ones it shows. In `--json` that's each sandbox's `activity`, `totals`
(`{"sandboxes", "activity"}`) and the `filters` asked for. The filters and
the activity need a Cayu Cloud with cayu-tech/cayu-cloud#766. An older Cloud
would ignore the filters, so `list` refuses them there
(`sandbox_filters_unsupported`) instead of listing every sandbox; without
filters it lists as before and reports no activity. The commands need a Cayu Cloud with cayu-tech/cayu-cloud#679; an older
one refuses the login on these routes.

`archives` lists the copy-in archives Cloud keeps for the Agent. Once a staged
copy in is extracted, Cloud keeps its archive for a while, and the Agent's next
copy in of the same archive (same size and SHA-256) into any of its sandboxes
starts from it without an upload (the copy says `reused`). The command shows
each kept archive's id, status (`ready`, or `storing` while the broker writes
it), size, SHA-256, when it was kept, its last use and when it lapses, and the
install's setting (how long after the last use, how many per Agent, how big).
It never shows where an archive is stored, and it changes nothing: Cloud lets
archives lapse on its own, and nothing gives one up on request. It reads up to
`--limit` (default 50), newest first; `--cursor` starts from an earlier list's
`next_cursor`. It needs a Cayu Cloud with cayu-tech/cayu-cloud#970; an older
one answers `sandbox_archives_unsupported`.

Cloud also logs every create it refuses for a full limit, with the limit, how
many held a place and when the soonest lease runs out (`sandbox_quota_refused`
in its API log).

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

A Python process runs the runtime's own work for all of its sessions on one CPU
core, however many vCPUs its task has, and past a few concurrent sessions each
of them slows down. So `cayu worker NAME` starts one process per vCPU of its
`cpu_millis`, rounded down: two for `cpu_millis = 2000`, one for 1000 or less.
Set another count with `--processes N` in the `[worker]` command, or with the
`CAYU_WORKER_PROCESSES` setting (`N` or `auto`); `--processes 1` runs one. The
processes share the Agent's PostgreSQL database, whose claims keep them from
taking the same work. Sessions a web process runs itself are bound
the same way, so move them to the worker. See [one process is one CPU
core](project-workers.md#one-process-is-one-cpu-core); a worker logs a warning
when its process stays saturated.

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

## Compass

Compass watches how an Agent runs and where people step in, and proposes
findings: an agent change a Compound mission can make, or an autonomy grant. A
person accepts or dismisses each one; Compass changes nothing on the Agent by
itself. It judges every finding by the objectives the Agent's Release declares
under `[tool.cayu.improvement]`, so declare them first (`cayu guide
evals-production#objectives-for-compass`, checked with `cayu objectives
validate`). Cloud refuses an Agent whose Release declares none with
`no_objectives`.

```console
cayu cloud compass enroll my-agent --agent my-agent
cayu cloud compass verify my-agent --wait
cayu cloud compass show my-agent
cayu cloud compass pass my-agent --wait
cayu cloud compass findings list my-agent --status proposed --limit 20
cayu cloud compass findings list my-agent --status proposed --limit 20 --cursor CURSOR
cayu cloud compass findings list my-agent --summary
cayu cloud compass findings accept my-agent FINDING --note "Worth trying"
cayu cloud compass findings dismiss my-agent FINDING --note "Expected behavior"
cayu cloud compass findings dismiss my-agent --rule repeated_tool --note "Paging, not repeats"
cayu cloud compass findings sessions my-agent FINDING
cayu cloud compass passes list my-agent --all
cayu cloud compass missions my-agent
cayu cloud compass disable my-agent
```

The commands call Cloud's `/v1/compass/agents/...` routes, and its paged
`/v1/compass/findings` and `/v1/compass/passes`, with your `cayu cloud
login` (or the selected context's API key), and Cloud answers only for your
Organization's Agents. They need a developer or organization administrator; a
member is refused with `code` `developer_required`. They mirror Cloud's operator
`compass` commands, except that waits take `--poll-seconds` and
`--wait-seconds` like the rest of `cayu cloud`.

- **A finding's sessions.** A finding cites at most 10 sessions. `findings show`
  says how many it cites of how many (`sessions_cited`, from `session_count`), and
  `first_seen_at`/`last_seen_at` with `last_seen_release_id` show whether it still
  happens. `findings sessions` lists every session behind it across every pass,
  latest first, with when each last showed it and its Release; `--limit N` keeps the
  latest N. A Cloud or Compass from before this answers
  `finding_sessions_unavailable`.
- **Enrolling.** `--agent` names a Cayu agent inside the Agent that Compass
  watches; the Agent must register it, or it won't start with Compass's fork
  controller. Repeat it to watch several (1 to 16). Each `enroll` sets the whole
  enrollment: options left out take Cloud's defaults (12 forks, no model runs,
  5 runs per finding, a 14-day window, $20 per mission and $100 a month), and an
  agent left out is no longer watched. `--allow-model-runs` needs
  `--fork-budget-usd`. `--release` picks the Release whose objectives judge
  findings (the selected one by default), and `--api-path /cayu/api` is for an
  Agent that mounts Cayu with `mount_cayu` instead of running `cayu serve`.
- **Several agents and older Cloud releases.** `enroll` always sends the agents
  as `agents`. A Cloud release from before several agents refuses that field, so
  with one `--agent` the command sends it again as `agent` under the same
  idempotency key, and with several it fails saying that Cloud watches one agent
  per Agent. Once an Agent watches several agents, Cloud refuses a single
  `agent` with `agents_required`.
- **Idempotency.** `enroll`, `verify`, `pass`, `findings accept` and
  `findings dismiss` take `--idempotency-key`. Without one the command chooses a
  key and prints it as `idempotency_key`, on an error too, so a write whose
  answer was lost (`compass_outcome_unknown`) can be repeated with it.
- **Jobs.** `verify` and `pass` print the first job Compass queued, with one per
  watched agent in `jobs`; `--agent` queues only that agent's. With `--wait`
  they wait until every job succeeds or fails, retrying brief API outages, and
  exit `2` with `wait_timeout` and a `compass job` command once
  `--wait-seconds` pass.
- **Reading findings.** `findings list` prints a table (`--json` prints the
  JSON envelope) of the findings, strongest first. `--agent NAME`, `--rule`,
  `--dimension`, `--min-score N` and `--tool NAME` narrow them, and
  `--limit N` keeps the N strongest (correctness, then cost, time and
  attention; the highest score first). A finding's tool is the tool its
  ledger row is about: its calls, or the approval gate on it. `--summary`
  counts the matching findings by rule and tool, with how many of each are
  proposed, accepted and dismissed. Cloud filters the findings when it can; a
  Cloud release from before that answers every finding, and the CLI filters
  them itself.
- **Paging findings.** Narrowed by no more than `--status` and `--kind`
  (`agent_change`, `autonomy_grant` or `platform_issue`), `--limit N` reads
  Cloud's paged listing: the same N strongest, with `total` and a
  `next_cursor`. The table ends with "More findings follow: pass --cursor
  CURSOR"; pass it with the same options for the next N. `--cursor` without
  `--limit` lists every finding after it. Cloud pages findings by status and
  kind only, so `--cursor` and `--kind` can't be combined with `--agent`,
  `--rule`, `--dimension`, `--min-score` or `--tool`, nor `--cursor` with
  `--summary`; with those filters `--limit N` keeps the N strongest that match,
  with no cursor. A Cloud release from before paged findings carried their
  `score` leaves the SCORE column `-` for them. Without `--limit`, `--cursor` or `--kind` the command
  reads the Agent's findings as before. A cursor Cloud doesn't recognize is
  refused with `invalid_cursor`, and a Cloud release from before paging answers
  `compass_paging_unavailable` to `--cursor` and `--kind` (a plain `--limit`
  still works there).
- **Passes.** `compass show` gives an Agent's recent passes; `passes list`
  lists every one, newest first: when it ran, its window, the sessions it read,
  the findings it proposed and left waiting, and the mission it made ready or
  why it made none. `--limit N` lists N (20 by default), `--cursor` goes on
  from an earlier listing's `next_cursor`, and `--all` follows the cursors to
  the oldest pass. The table ends with how many it shows of Cloud's `total`
  ("20 of 57 passes."), which `--json` prints as `total`; a Cloud release from
  before it counted passes sends none, and `total` is null. `--json` prints the
  JSON envelope. A Cloud release from before paging answers
  `compass_paging_unavailable`.
- **Dismissing findings in bulk.** `findings dismiss AGENT --rule RULE --note
  TEXT` dismisses every proposed finding of the rule, narrowed by the same
  options as `findings list`. It shows what it will dismiss and asks first;
  `--yes` skips the question, and without a terminal it refuses with
  `confirmation_required` unless `--yes` is given. Accepted and already
  dismissed findings are left as they are, so repeating it dismisses nothing
  more, and it reports how many it dismissed and how many it skipped. It
  sends the findings in batches of 100 under one idempotency key, from which
  Cloud derives one key per finding; a Cloud release without the bulk route
  gets them one at a time, and the command says so. If it stops partway, the
  error says how many it dismissed; repeating it with the same
  `--idempotency-key` dismisses the rest. `findings list` and `findings
  dismiss` print text; pass `--json` for the JSON envelope with
  `idempotency_key`.
- **Missions.** `missions` lists the Compound missions Compass made ready and
  why Cloud starts none now, if it doesn't. Cloud starts them on its own, and
  each waits for a person to approve its plan before it spends anything.

A refusal exits `2` with Cloud's `code` and message, for example
`no_objectives`, `invalid_settings`, `not_enrolled`, `fork_controller_conflict`,
`enrollment_incomplete` (repeat the same enroll, or disable), `agents_required`,
`agent_not_watched`, `compass_disabled` or `compass_unavailable`.

## Tools and knowledge registration

When Cloud binds the Agent's `CAYU_CLOUD_REGISTRY_TOKEN`, the Runtime registers its
specialists' tools, their package or MCP-server source, and the active shared
knowledge the application created, then reports how often they are used. Model-written
memories and other conversation-derived knowledge are never sent. See
[Tools and knowledge registered with Cloud](cloud-inventory.md) for the privacy
boundary, routes and retry behavior.

## Sign-in and home page

An Agent says in `cayu-cloud.toml` how people sign in to its pages and which page they
land on when they open it from Cayu Cloud:

```toml
home = "/close"

[sign_in]
mode = "cloud"
public_paths = ["/vendor/", "/webhooks/"]
```

- `home` is optional and sits at the top level, before any table. It is a path on the
  Agent's own host. Without it, people land on the Agent's address as before.
- `[sign_in]` is optional. `mode = "cloud"` (recommended) has Cayu Cloud sign people in
  before they reach the Agent's pages. `mode = "own"` leaves sign-in to the Agent's own
  login. An Agent without a `[sign_in]` table keeps its own sign-in, exactly as before.
- `public_paths` is optional and allowed only with `mode = "cloud"`. Each entry is a path
  prefix that anyone can open without signing in, such as static assets or webhooks. It
  holds at most 50 different prefixes.

`home` and each `public_paths` entry start with `/` and are at most 1024 printable ASCII
characters. They may not contain a scheme or host (including a leading `//`), a query
(`?`), a fragment (`#`), a space, a backslash, or a `..` segment, also when
percent-encoded; percent-encode other characters. `mode` is required in `[sign_in]` and
must be `"cloud"` or `"own"`. Other keys in `[sign_in]` are rejected like those in
`[web]`. All of this requires `schema_version = 2`. `cayu cloud deploy` checks it before
uploading and fails with `manifest_invalid`.

These settings are not sent in the deployment request. Cayu Cloud reads them from the
uploaded `cayu-cloud.toml`, so `home` and `mode = "cloud"` must be in the deployment
root's file: a custom `--manifest` path that declares them is rejected with
`manifest_invalid` before upload. A custom `--manifest` deploy is also rejected when the
uploaded root `cayu-cloud.toml` declares `home` or a `[sign_in]` other than
`mode = "own"`, or cannot be read as TOML, because that deploy would get the root file's
settings rather than the ones in the selected manifest. Cayu Cloud acts on them once its router sign-in ships;
until then they are accepted and have no effect.

`cayu cloud init` writes `[sign_in]` with `mode = "cloud"` when `cayu serve` uses the
ready-made `cayu.server.environment_auth:OPERATOR_BASIC_AUTH` or `OPERATOR_ONLY_AUTH`
target, and for a worker.
It writes `mode = "own"` for a project with a `service_factory`, whose product and
Developer dashboard access are its own, and for any other auth target; see
[Initialize an Agent project](#initialize-an-agent-project). It writes no `home`,
because init cannot know a page of the project's own. Add `home` yourself when the
Agent serves one.

## Webhooks to a sleeping Agent

An Agent's web process sleeps when it is idle. An Agent that receives webhooks lists their
path prefixes, and Cayu Cloud keeps a `POST`, `PUT`, `PATCH` or `DELETE` to one of them
while the Agent sleeps, answers the sender `202`, and delivers the request once the Agent
is ready:

```toml
[webhooks]
paths = ["/webhooks/"]
```

- `[webhooks]` is optional. Without it nothing changes: a request to a sleeping Agent
  wakes it and gets `503`.
- `paths` is required in `[webhooks]` and holds 1 to 20 different path prefixes. They
  match like `public_paths`: one ending in `/` covers everything under it, any other
  covers that path and everything below it. Each follows the same rules as `home` and
  `public_paths` (see [Sign-in and home page](#sign-in-and-home-page)).
- Other keys in `[webhooks]` are rejected like those in `[web]`. It requires
  `schema_version = 2`. `cayu cloud deploy` checks it before uploading and fails with
  `manifest_invalid`.
- With `[sign_in] mode = "cloud"`, webhook paths are also public: no one is sent to sign
  in and no identity is given, live or on delivery, so you don't list them again in
  `public_paths`. With the Agent's own sign-in they reach the Agent like any request.
- A delivered request is the kept one, unchanged, plus `X-Cayu-Cloud-Replayed: 1` and
  `X-Cayu-Cloud-Received-At`. Delivery is at least once, so verify signatures and
  deduplicate by the sender's delivery ID, as for
  [external event waits](external-event-waits.md).

Like `home` and `[sign_in]`, `[webhooks]` is not sent in the deployment request: Cayu
Cloud reads it from the deployment root's uploaded `cayu-cloud.toml`. A custom
`--manifest` path that declares it is rejected with `manifest_invalid`, and so is a custom
`--manifest` deploy whose root `cayu-cloud.toml` declares it. A Cayu Cloud that doesn't
keep webhooks yet ignores the table. `cayu cloud init` writes no `[webhooks]`, because it
cannot know which paths receive them.

## Operator check

`cayu cloud service status` says whether the Agent's processes are up, not whether the Agent
works. An Agent can declare one route that checks that itself (its AI gateway answers, one live
workflow completes), and `cayu cloud service check` has Cayu Cloud call it after a deploy, with
the Agent's operator login, whatever the Agent's own sign-in:

```toml
[operator_check]
path = "/operator-check"   # required
method = "POST"            # optional: "POST" (the default) or "GET"
timeout_seconds = 30       # optional: 1 to 45
```

- `path` follows the same rules as `home` (see [Sign-in and home page](#sign-in-and-home-page)),
  and can't be under `/.cayu/`, which Cayu Cloud's Agent router serves itself, or under one of
  the `[webhooks] paths`, whose requests Cloud keeps while the Agent sleeps. Other keys are
  rejected like those in `[web]`. It requires `schema_version = 2`, and `cayu cloud deploy`
  checks it before uploading and fails with `manifest_invalid`.
- Like `[webhooks]`, it is not sent in the deployment request: Cayu Cloud reads it from the
  deployment root's uploaded `cayu-cloud.toml`, so a custom `--manifest` path that declares it,
  or a custom `--manifest` deploy whose root file declares it, is rejected.

Cayu Cloud calls exactly the declared route of the serving release and nothing else, through the
Agent router, with HTTP Basic `CAYU_OPERATOR_USERNAME` and `CAYU_OPERATOR_PASSWORD` (the login it
injects into every Agent process). The router passes that request on unchanged with either
`[sign_in]` mode. The app serves the route with `cayu.server.operator_check`:

```python
from cayu.server.operator_check import CheckFailed, OperatorCheck

check = OperatorCheck()


@check.step("ai_gateway")
async def ai_gateway() -> str:
    reply = await gateway.test_connection()
    if not reply.ok:
        raise CheckFailed(f"The gateway answered {reply.status}.")
    return f"Answered in {reply.elapsed_ms} ms"


app.include_router(check.router())  # POST /operator-check, operator login only
```

Steps run in order, plain or `async`, and a failed step doesn't stop the next. A step fails when
it raises: `CheckFailed` shows its message, any other exception only its type (the traceback
goes to the app's log). The route answers `200` with `{"ok": true, "summary", "checks": [{"name",
"ok", "detail", "duration_ms"}]}` when every step passes and `503` with `"ok": false` otherwise.
`router()` reads the operator login when it is called and raises `AuthConfigurationError` without
it; pass `path=`, `method=` and `auth=` (any server auth dependency) to serve it elsewhere or
protect it another way, and call `await check.run()` for the same report without a route. An app
that serves the route itself checks the same Basic login and answers the same JSON.

```bash
cayu cloud service check --application my-agent
cayu cloud service check --application my-agent --json
```

The command prints the verdict, the HTTP status, how long the check took, the summary and each
step, and a hint when the answer shows the route isn't set up for Cloud (for example `401`: the
route refused the operator login). `--json` prints the JSON envelope with `result.outcome`
(`passed` or `failed`), `http_status`, `duration_ms`, `deployment_id` and a bounded `body`. A
failed check exits `2` with `operator_check_failed` and the same result in the error. Cloud's
refusals exit `2` with their `code`: `operator_check_not_declared`, `service_not_running`,
`no_web_service`, `operator_check_timeout` (the check may still be running: read `service logs`
before running it again) or `agent_unreachable`.

A sleeping Agent wakes for the check. While Cloud reports `agent_starting` (the check wasn't
sent), the command asks again after Cloud's `Retry-After`, for up to `--wait-seconds` (300 by
default). It never sends a check twice otherwise, because a check can run a live workflow. Cloud
records who ran each check, on which Agent and release, when, and the result.

## Application database migration

An Agent that keeps its own tables beside Cayu's, such as a project generated with
`cayu new --with app-database`, declares the command that migrates them:

```toml
[database]
migrate = "alembic upgrade head"
```

- `[database]` is optional. Without it nothing changes: Cloud migrates only Cayu's own
  storage, and the deployment request is the same as before.
- `migrate` is required in `[database]`. It is a command, run from the project root
  without a shell, of at most 2048 characters. It reads the database from
  `CAYU_DATABASE_URL`, or from `CAYU_DATABASE_DIRECT_URL` when that is set.
  Other keys in `[database]` are rejected like those in `[web]`. It requires
  `schema_version = 2`, and `cayu cloud deploy` checks it before uploading and fails
  with `manifest_invalid`.
- The command is sent in the deployment request, so a custom `--manifest` path can
  declare it.
- A Cayu Cloud that doesn't run application migrations rejects the declaration.
  `cayu cloud deploy` then exits `2` with category `database_migration_unsupported`,
  and nothing is deployed. Deploy to a Cayu Cloud that runs them, or remove `[database]`
  and migrate the application's tables yourself.

`cayu cloud init` writes `migrate = "alembic upgrade head"` for a project generated with
`--with app-database` (`app-database` in `[tool.cayu.scaffold].capabilities`), and
reports it as `result.database.migrate`.
