# Session-store targets

Cayu storage-aware CLI commands resolve one durable session store without
importing or constructing the application factory. The shared resolver applies
this precedence:

1. explicit `--sqlite PATH` or `--postgres DSN` options;
2. `CAYU_DATABASE_URL`;
3. `[tool.cayu.session_store]` in the nearest applicable `pyproject.toml`;
4. the exact `data/cayu.db` file under that project root, when it exists;
5. an actionable missing-configuration error.

`--sqlite` and `--postgres` are mutually exclusive. An explicit option always
wins over the environment and project configuration, which keeps scripts and
cross-project inspection deterministic.

## Project configuration

Configure SQLite with an explicit typed table:

```toml
[tool.cayu.session_store]
backend = "sqlite"
path = "data/cayu.db"
```

Relative paths resolve from the directory containing `pyproject.toml`, not the
caller's current working directory.

Without a `session_store` table, Cayu recognizes only the exact
`<project>/data/cayu.db` convention. It does not search for alternate filenames
or create the file during resolution.

Configure Postgres by naming an environment variable:

```toml
[tool.cayu.session_store]
backend = "postgres"
env = "CAYU_DATABASE_URL"
```

Do not place a production DSN in `pyproject.toml`. Cayu reads the named variable
when the command runs. `postgres://` and `postgresql://` URLs are accepted.

## CLI workflow contract

Storage-aware commands that adopt this resolver can use a configured project
without a repeated selector. Their explicit `--sqlite` and `--postgres` options
inspect another store or override project settings without guessing from an
untyped value.

`CAYU_DATABASE_URL` can also select a store without changing project
configuration. It accepts a Postgres URL or an absolute SQLite URL such as
`sqlite:///srv/cayu/cayu.db`.

## Application stores

Application factories select their stores from the same variable with
`open_application_stores(configured_database_url(), sqlite_path=...)`, which parses
URLs with this resolver's implementation (`cayu.storage.targets`). Generated
projects keep `[tool.cayu.session_store]` pointed at the local SQLite file that
the factory uses without `CAYU_DATABASE_URL`, so the CLI and the app select the
same store for every combination of the variable and that section. Explicit
`--sqlite` and `--postgres` options still override both for one command.

| Variable | Read by | Effect |
| --- | --- | --- |
| `CAYU_DATABASE_URL` | App factories and the CLI | Postgres URL, or absolute SQLite URL. Unset selects the project's local SQLite file. A set but blank value is an error. |
| `CAYU_DATABASE_POOL_MAX` | `open_application_stores` and the project Evals store | Maximum connections per pool (default 5). The application's PostgreSQL session, task, knowledge, and (with `product_operations=True`) product operation stores share one pool. |
| `CAYU_DATABASE_DIRECT_URL` | `open_application_stores` | Optional Postgres URL for the task-admission `LISTEN` connection. `LISTEN` does not survive transaction pooling, so set it to a direct server address when `CAYU_DATABASE_URL` points at PgBouncer or a similar proxy. Pooled store traffic keeps using `CAYU_DATABASE_URL`. |
| `CAYU_REQUIRE_POSTGRES` | Every Cayu SQLite store | `1` makes SQLite stores raise at construction; `0` or unset changes nothing. Deployments set it so a missing `CAYU_DATABASE_URL` fails at startup. |

A PostgreSQL-backed application process therefore opens at most
`CAYU_DATABASE_POOL_MAX` pooled connections, one dedicated listener connection,
and, when `cayu serve` or `cayu check` assembles the project, the Evals store's
pool. The PostgreSQL application stores validate the schema; apply migrations
with `cayu storage migrate` as a deploy step.

Target resolution only identifies and validates the requested backend. It does
not search arbitrary directories, import an app factory, create a database, or
run migrations. Read-only commands open the resolved target under their own
non-mutating backend contract.

## Storage commands

`cayu storage status`, `cayu storage migrate`, and `cayu storage export` use
this resolver. Without `--sqlite` or `--postgres` they select the store from
`CAYU_DATABASE_URL`, then `[tool.cayu.session_store]`, then the existing
`data/cayu.db` convention. An explicit option always wins, and `--postgres`
still accepts a libpq key/value DSN as well as a URL. The JSON output names
the selection in `target_source` (for example `explicit`,
`environment:CAYU_DATABASE_URL`, or `project:APP_DATABASE_URL`).

Output and errors never contain the connection string. The Postgres `target`
field and the migration receipt carry only the scheme, host, port, and
database name; credentials and query parameters are removed, including from
driver errors after a failed connection.

### Direct connection for migrations

Migrations hold session-level PostgreSQL advisory locks while they build
indexes concurrently. A transaction-pooling proxy such as PgBouncer can run
consecutive statements of one session on different server connections, so it
cannot preserve those locks. When the application's `CAYU_DATABASE_URL` points
at such a pooler, also set `CAYU_DATABASE_DIRECT_URL` to an unpooled URL for
the same database:

- `cayu storage migrate` and `cayu storage status` connect through
  `CAYU_DATABASE_DIRECT_URL` whenever it is set and the target was resolved
  rather than passed with `--sqlite` or `--postgres`;
- the resolved store must be Postgres, and the variable must contain a
  Postgres URL; otherwise the command fails without connecting;
- `cayu storage export` and the running application keep using the pooled
  `CAYU_DATABASE_URL`.

## Deployment migration step

Postgres stores validate their schema at startup and never run DDL
(ADR 0001). Apply migrations as a separate deployment step:

1. Run `cayu storage migrate` as a one-off task from the release image, with
   the service's exact environment and secrets, before the new service
   version starts. The migration preflight reads the same configuration the
   application reads, including the public-authority alias keyring
   (`CAYU_PUBLIC_AUTHORITY_ALIAS_*`), so a different environment can pass or
   fail preflight differently from the service.
2. Keep the connection string in `CAYU_DATABASE_URL` (or the variable named
   by `[tool.cayu.session_store]`) instead of the command line, where task
   definitions, process listings, and logs would expose it.
3. Start the service only after the step exits `0`. A failed migration then
   fails the deployment instead of crash-looping application processes.

A platform can ask first whether a release needs the step:

```console
cayu storage status --json
```

| Exit | `migration.need` | Meaning |
| --- | --- | --- |
| `0` | `up_to_date` | The database is at this build's latest revision. |
| `3` | `forward_migration` | A forward migration is available and its read-only preflight passes with the given inputs. |
| `4` | `incompatible` | This build cannot migrate the database with the given inputs. |
| `1` | — | General error, such as an unresolvable target or a failed connection. |
| `2` | — | Command-line usage error. |

The `migration` object reports:

- `input_revision` — the database's current revision (`0` when it has no Cayu
  schema) and `input_empty`, which is true only when the database contains no
  Cayu relations at all;
- `target_revision` — this build's latest revision;
- `breaking_boundary` and `breaking_revisions` — whether breaking revisions
  lie between input and target, and which ones `migrate` must acknowledge. An
  empty database has none. Both are `null` when the input state cannot be
  interpreted;
- `missing_acknowledgements` — breaking revisions not covered by the
  `--acknowledge-breaking` options passed to `status`;
- `resuming` — whether an interrupted Postgres migration left a durable
  receipt that `migrate` will resume;
- `reason` and `detail` — for `incompatible`: `database_newer_than_build`,
  `invalid_schema_state`, `breaking_acknowledgement_required`, or
  `migration_preflight_rejected` (for example a missing alias keyring or
  schema privileges).

`status` accepts the same `--acknowledge-breaking REVISION` and
`--reset-empty-recall-state` inputs as `migrate` and runs the same read-only
preflight, so exit `3` means the matching `migrate` invocation would pass
preflight. `status` does not take backup options; `migrate` still requires
backup authority for a database that holds Cayu data (see below). `status`
never writes to the database. When `migration.resuming` is true, `migrate`
resumes the recorded operation without repeating preflight, and `status`
likewise skips it.

### Backup authority

A Postgres migration of a database that already holds Cayu data records its
backup authority in the migration receipt. Pass exactly one of:

- `--backup-managed rds-snapshot:<snapshot identifier>` for an RDS snapshot
  taken before the migration. The identifier starts with a letter, contains
  letters, digits, and single hyphens, does not end with a hyphen, and is at
  most 255 characters. An automated snapshot may keep its `rds:` prefix.
- `--backup-managed rds-pitr:<timestamp>` for an RDS point-in-time recovery
  target. The timestamp is RFC 3339 in UTC, for example
  `2026-09-29T12:00:00Z` (`+00:00` is also accepted).
- `--backup-sha256 SHA256` for an application-consistent backup you hold.
- `--waive-backup` for an explicit operator waiver.

The receipt records a managed backup as
`{"mode": "provider_managed", "reference": "<REF>", ...}`, distinct from a
waiver (`"mode": "waived"`). `--backup-managed` is valid only for Postgres.

When the database has no Cayu schema yet, the first migration needs no backup
authority: the receipt records `input_revision: 0`, `input_empty: true`, and
`backup.mode: "not_required"`. A backup option passed for an empty database is
accepted and recorded as given, so a platform may always pass one.

A receipt binds its exact invocation. If a migration is interrupted, rerun the
same command with the same backup option; a new reference (for example a new
`rds-pitr` timestamp) is rejected as a different migration invocation.
