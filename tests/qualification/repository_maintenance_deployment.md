# Single-host PostgreSQL deployment

These are deployment inputs, not a completed production qualification. Native
Compose validation, image builds, PostgreSQL readiness, live coding, shutdown,
restart and coordinated backup/restore must pass on the intended host before
admission. Structural source tests do not establish these outcomes.

## Prerequisites and trust boundary

Use a Linux host with Docker Engine and Compose 2.30 or later (raw env files),
the default `/var/run/docker.sock`, and reviewed, locally loaded immutable
images. Do not override `DOCKER_HOST` or use a different Docker context inside
the services. All five application roles inspect the same coding image and
construct the same application graph. Run one instance of each role.

Application containers are trusted host services: Docker socket access is
host-equivalent authority. Per-role environment separation is not isolation
against a compromised host service. The native coding runner creates separate
restricted, no-network guest containers; do not give those guests host secrets,
the Docker socket, or application-state mounts. Compose does not own those
guest lifetimes. No service in this file merges pull requests.

Provision three distinct absolute host directories deliberately: the admitted
fixed-seed repository, persistent runtime/artifact state, and Git broker state.
Source must match the application's fixed base, not an arbitrary checkout.
Make the required paths accessible to UID/GID 10001 without recursively changing
unrelated user data. Runtime and broker directories must be writable; only the
coding role receives a writable source mount. Missing bind paths are errors.
Keep private files outside these directories and outside all source packages.

## Build and record immutable inputs

Emit this project using the reviewed **installed Cayu wheel**. Confirm its
generated `pyproject.toml` version agrees with that wheel; source-checkout
emission using a different installed version is not release evidence. Retain
the wheel hashes, lock, emitted source revision and image digests privately.

1. Build the native coding image with this project's `build_coding_image.py`
   and its documented arguments. Retain the resulting immutable
   `docker-coding-image.json`; make that image available in the same host daemon.
2. Select pinned Python 3.12 Debian and Docker CLI images and exact Debian Git
   and ripgrep package versions. Build the host-tools image below. Package
   downloads require separate authorization; do not run this merely to inspect
   the project.
3. Supply `deployment/requirements.lock` with exact versions and hashes for
   **all transitive dependencies** required by the generated `pyproject.toml`,
   and matching Linux/Python wheels in `deployment/wheels/`. Include the reviewed
   Cayu wheel and server/PostgreSQL extras. The application build is offline and
   rejects missing wheels or hashes; it does not resolve new dependencies.

```sh
docker build -f Dockerfile.host-tools \
  --build-arg PYTHON_IMAGE="$PINNED_PYTHON_IMAGE" \
  --build-arg DOCKER_CLI_IMAGE="$PINNED_DOCKER_CLI_IMAGE" \
  --build-arg GIT_PACKAGE_VERSION="$REVIEWED_GIT_VERSION" \
  --build-arg RIPGREP_PACKAGE_VERSION="$REVIEWED_RIPGREP_VERSION" \
  -t maintenance-host-tools:reviewed .
docker image inspect maintenance-host-tools:reviewed --format '{{.Id}}'
docker build -f Dockerfile.application \
  --build-arg CAYU_HOST_TOOLS_IMAGE="$PINNED_HOST_TOOLS_IMAGE" \
  -t maintenance-application:reviewed .
docker image inspect maintenance-application:reviewed --format '{{.Id}}'
```

Set `PINNED_HOST_TOOLS_IMAGE=maintenance-host-tools:reviewed`, the exact local
tag just inspected. Docker's `FROM` resolver treats a bare local `sha256:` image
ID as a registry image name on some BuildKit hosts, so passing that ID directly
can unexpectedly attempt a network pull. The tag is safe here because automatic
pulls are disabled and it is verified immediately before the offline build;
record the inspected image ID alongside the resulting application digest.
Dockerfile-specific context allow-lists exclude runtime state and deployment
credentials; never place credentials in copied Python packages. Host-tools needs
no local build payload.

## Private configuration

Create these required UTF-8 env files with mode 0600. Use raw `NAME=value` lines,
without shell quoting or expansion; each JSON value occupies one line. Do not
commit them or retain expanded `docker compose config` output in public evidence.

| File | Contents |
| --- | --- |
| `deployment/common.env` | Shared database URL, provider/model and provider key, public-authority alias keyring, reviewed budget JSON, and non-secret Git/GitHub authority configuration |
| `deployment/api.env` | `CAYU_MAINTENANCE_ACCESS_JSON` with separate product and operator credentials |
| `deployment/git.env` | `CAYU_MAINTENANCE_GIT_USER` and `CAYU_MAINTENANCE_GIT_TOKEN` only |
| `deployment/github.env` | `CAYU_MAINTENANCE_GITHUB_TOKEN` only |
| `deployment/postgres.env` | `POSTGRES_USER=cayu`, `POSTGRES_DB=cayu`, and a private `POSTGRES_PASSWORD` |

For password-free CLI arguments, use
`CAYU_DATABASE_URL=postgresql://cayu@postgres/cayu` and matching `PGPASSWORD` in
common.env. This is a dedicated trusted database, not a least-privilege DB-role
recipe. Preserve the same alias keys, provider, pinned model and budget/pricing
across every role and restart. A missing provider key selects a different graph,
not a supported credential-free deployment. Review all attributable model costs
against the total authorized trial cap; no prices are supplied by this guide.

Use the exact application configuration contracts for
`CAYU_MAINTENANCE_BUDGET_JSON`, `CAYU_MAINTENANCE_GIT_JSON`,
`CAYU_MAINTENANCE_GITHUB_JSON`, and the two corresponding `*_HOST_JSON` values.
Git host `broker_root` is `/delivery`, `git_executable` is `/usr/bin/git`, and
`remote_url` is the approved remote. Configure the explicit HTTPS
`CAYU_MAINTENANCE_GITHUB_WEB_ORIGIN` used for final result links. Preserve original
request timestamps and authority tuples on retry; do not mint replacement IDs
to evade a blocked operation. Delivery tokens never belong in common.env.

Export the following **non-secret** Compose interpolation variables: immutable
`CAYU_MAINTENANCE_APP_IMAGE`, a reviewed pgvector/PostgreSQL 16 digest as
`CAYU_MAINTENANCE_POSTGRES_IMAGE`, absolute `CAYU_MAINTENANCE_SOURCE`,
`CAYU_MAINTENANCE_STATE`, `CAYU_MAINTENANCE_BROKER`, and the actual socket group
as `CAYU_MAINTENANCE_DOCKER_GID`. Images must already be loaded; automatic pulls
are disabled. Keep a stable Compose project name across restarts so that the
PostgreSQL volume is not silently replaced by an empty project volume.

## Explicit schema preparation and startup

Run commands from the emitted project. Only one schema operator may run; keep
all application writers stopped. Validate configuration without printing secrets:

```sh
docker compose -f compose.yaml config --quiet
docker compose -f compose.yaml up -d postgres
docker compose -f compose.yaml ps
docker compose -f compose.yaml run --rm --no-deps api cayu storage status \
  --postgres postgresql://cayu@postgres/cayu
```

Wait for PostgreSQL health before running schema commands. Before migration,
capture a coherent backup as below. Pass its SHA-256 to the native migration
command and explicitly acknowledge each breaking revision reported by status.
Do not automatically waive backup, including for a fresh database.

```sh
docker compose -f compose.yaml run --rm --no-deps api cayu storage migrate \
  --postgres postgresql://cayu@postgres/cayu --backup-sha256 "$BACKUP_SHA256"
# Add --acknowledge-breaking for each actual required revision, if any.
docker compose -f compose.yaml run --rm --no-deps api \
  python -m operations.maintenance_schema initialize
docker compose -f compose.yaml run --rm --no-deps api \
  python -m operations.maintenance_schema check
docker compose -f compose.yaml up -d api coding git_preparation git_delivery github_delivery
```

Runtime migration and application reservation bootstrap have different owners.
Neither occurs automatically on startup. Partial/conflicting reservation schemas
are refused; investigate rather than deleting their tables. Each application
role validates startup schemas before accepting work. The API's loopback health
endpoint indicates availability after startup, **not periodic database readiness**.
Remote access needs separately authorized TLS termination and access controls.

## Shutdown, backup, restart and rollback

Stop API intake first, then workers; preserve the database until owned work has
positively settled. Inspect exit status, operator task/delivery views, cleanup
evidence and owned Docker guests. A 30-second worker grace or 45-second Compose
stop grace bounds waiting, not cleanup. Exit 124, forced kill, missing receipts,
or an unknown remote effect is not proof that a retry is safe.

```sh
docker compose -f compose.yaml stop api
docker compose -f compose.yaml stop coding git_preparation git_delivery github_delivery
docker compose -f compose.yaml ps -a
```

After verified quiescence, capture a restricted `pg_dump -Fc` using
`docker compose exec -T postgres pg_dump -U cayu -d cayu -Fc`, together with source,
runtime artifacts, broker state, alias keys/private configuration, and exact
application/coding/database image manifests. Record hashes and test a coordinated
restore in an isolated target. A database dump alone is not a complete backup.
Only then stop PostgreSQL. Never use `down -v`, global prune, or task resets as
incident recovery. Retain uncertain guests and durable identities for exact-owner
investigation; Compose termination does not reclaim native sibling containers.

Restart the same image/configuration and persistent paths only after examining
unfinished work through the native owners. Lease expiry alone is not permission
to repeat a Git push, GitHub write or coding effect. Use a stopped, homogeneous
upgrade rather than mixed-version rolling writers. Breaking rollback requires a
matching coordinated backup restore, not a down-migration. Remote commits/PRs
can survive local restore and must be reconciled before any new dispatch.

## Required deployment evidence

Use `operations/maintenance-incidents.md` for the operator exercise and its
explicit recovery limits. A stopped process or recorded terminal state alone
does not satisfy the required cleanup evidence.

Record native config validation, offline image build, fresh and reopened schema
checks, refusal of unprepared schema, admitted real Docker coding, exact approved
delivery, restart/acknowledgement-loss behavior, bounded stop with positive cleanup
or explicit uncertainty, and an isolated backup/restore exercise. Keep secrets
out of logs and public artifacts. These checks remain required even when the
source-level Compose contract tests pass.
