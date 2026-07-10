# ADR 0001 — Database storage and schema migrations

- **Status:** Accepted
- **Date:** 2026-06-16
- **Supersedes:** the ad-hoc `CREATE TABLE IF NOT EXISTS`-on-first-use scheme in
  `src/cayu/storage/*`.
- **Source:** `cayu-database-storage-plan.html` (storage architecture plan), plus
  review of the Postgres `SessionStore`/`TaskStore` backend (cayu#16) and findings
  from running lane-agent on it (cayu#32).

> **Greenfield assumption.** Nothing is released yet; every schema, table name, and
> default is freely modifiable. The only consumer (lane-agent) is a disposable test
> deployment whose database can be recreated at will. We therefore design for the
> *correct* end state, not for backward compatibility — there is no legacy to carry.
> Past incidents (e.g. the `processing_start_at` crash) are cited only as
> motivation for the design, not as compatibility constraints.

## Context

Cayu's sessions, events, transcripts, checkpoints, and tasks are **runtime state**,
not just logs: the runtime needs filtered reads, durable cursors, atomic status
transitions, and concurrent-worker safety. A relational database is the right
primary store, behind the existing `SessionStore` / `TaskStore` contracts.

The two backends are at very different maturity, and the problem is **specific to
Postgres**. On Postgres, schema is created by **`CREATE TABLE IF NOT EXISTS` run
lazily on first store use**, with:

- **no revision record** — nothing says which schema version a Postgres DB is at;
- **no fail-fast** — an app binary expecting a column the DB lacks just crashes at
  query time (this hit lane-agent: a new `processing_start_at` column never got
  added to the existing prod table → `UndefinedColumn` crash-loop);
- **no concurrency coordination** — two stores sharing a pool
  (`PostgresSessionStore` + `PostgresTaskStore`, the production pattern) each run
  the full DDL; Postgres `CREATE … IF NOT EXISTS` is **not** atomic against a
  concurrent creator and can raise a duplicate-object error at startup;
- **unprefixed table names** (`sessions`, `events`, `tasks`, `checkpoints`,
  `transcript_messages`) that can **collide** with an app's own tables in a shared
  database (lane-agent's `sessions`-like app tables live in the same DB).

The **SQLite** backend is further along: it already stamps `PRAGMA user_version`
(currently `SCHEMA_VERSION = 6`) and fails fast on a too-new or unrecognized
version. But it only *rejects* a mismatch (it does not migrate), and the scheme is
SQLite-specific. This ADR generalizes versioning across backends: SQLite's
`user_version` becomes one backend's realization of the shared revision recorded in
`cayu_schema_migrations`, rebaselined (greenfield) to a common revision.

## Decisions

1. **Database stays the canonical runtime store.** Keep SQLite for local
   durability; keep Postgres for production. JSONL is an **export/replay/backup**
   format only — never the hot path.
2. **Keep the store contracts as the abstraction; no ORM, no migration
   framework.** Schema, SQL, pooling, and migrations live *below*
   `SessionStore`/`TaskStore`, so a new backend is "implement the contract," not a
   rewrite. We deliberately **do not** adopt SQLAlchemy (ORM), Alembic, or any
   migration library — see Decision 8 for the precise trigger and rationale.
3. **Add a Cayu-owned schema-version + migration system** with:
   - a `cayu_schema_migrations` table (revision id, applied_at, checksum);
   - monotonic, **forward-only** revision ids;
   - per-backend migration definitions (SQLite and Postgres DDL differ);
   - **validate-at-startup**: fail fast with an actionable error when the DB is
     too old (needs `migrate`) or *incompatibly* new (a breaking revision the
     binary doesn't support — see Decision 7's compatibility floor);
   - idempotent empty-schema initialization recorded at the baseline revision.
4. **Coordinate all schema work under a backend lock.** On Postgres, wrap
   create/migrate in a transaction-scoped advisory lock (`pg_advisory_xact_lock`)
   so concurrent creators/instances serialize: one runs the DDL, the rest wait and
   then validate. (Fixes the shared-pool race above.)
5. **Prefix Cayu tables with a configurable prefix, default `cayu_`.**
   `cayu_sessions`, `cayu_events`, `cayu_transcript_messages`, `cayu_checkpoints`,
   `cayu_tasks`, `cayu_schema_migrations`. No hyphens (they force quoted
   identifiers). Postgres-schema namespacing (`cayu.sessions`) may come later; the
   cross-backend default is the prefix.
6. **Migrations are explicit, never silent-on-import.** A separate
   `cayu storage migrate` / app-owned deploy step applies pending migrations;
   running app instances use `validate` at startup. Auto-`create` is allowed only
   for empty databases and only when explicitly enabled.
7. **Revision compatibility model.** Every revision is **additive**
   (forward-compatible — only adds tables/columns/indexes; older binaries keep
   working because the store selects explicit columns) or **breaking** (rename /
   drop / retype / semantic change). `cayu_schema_migrations` records a per-revision
   `compatible_from` floor: the oldest app revision that can operate against the DB
   at that revision (additive inherits the prior floor; breaking sets it to itself).
   **validate** passes iff `app.latest >= db.compatible_from` (binary new enough for
   the DB) **and** `db.revision >= app.min_supported` (DB not ancient); otherwise
   fail fast. `migrate` is the standard deploy step (a no-op when nothing is
   pending); startup never runs DDL.
8. **Postgres is the production standard; no ORM / no migration framework until a
   concrete trigger.** The "what if we add MySQL / SQL Server / Oracle?" question
   resolves to: **Postgres is the one supported production SQL backend** (SQLite is
   dev/local; in-memory is tests). Managed Postgres exists on every cloud, and
   "support any SQL DB for agent state" is a far larger promise than it looks —
   JSON semantics, upsert syntax (`ON CONFLICT` vs `ON DUPLICATE KEY` vs `MERGE`),
   identity/sequence generation, and especially the **migration lock primitive**
   (`pg_advisory_xact_lock` vs MySQL `GET_LOCK` vs MSSQL `sp_getapplock` vs Oracle
   `DBMS_LOCK`) all diverge per dialect. Therefore:
   - **No migration framework (Alembic).** It pulls in SQLAlchemy, is built for a
     branching/down-migration model we rejected (Q1), and its autogenerate is
     unreliable. Our `migrations.py` core (~200 lines) already does the chosen
     policy. No library abstracts the lock coordination — the genuinely hard part —
     so a framework does not make multi-backend migrations cheaper.
   - **No ORM.** The stores map rows ↔ pydantic explicitly and depend on hand-tuned
     concurrency SQL (`SELECT … FOR UPDATE`, `MAX(session_order)+1`, CAS
     transitions); an ORM would fight all of it.
   - **Trigger to adopt SQLAlchemy _Core_ (dialect builder only, not the ORM, not
     Alembic):** a *third committed production SQL dialect* (e.g. a real
     customer-required Oracle/MSSQL). Core slots in *below* the contracts, so it is
     an internal swap that changes zero callers; the migration model and CLI stay.
     Even then the per-dialect lock code stays hand-written.
   - **Cheaper intermediate step if a 3rd backend lands before that trigger:** a
     small internal "dialect" struct (type names + a few SQL fragments for
     upsert/lock/identity) captures ~80% of the divergence with no heavy dependency.
     Reach for SQLAlchemy Core only if *that* becomes painful.

## Policy decisions

| # | Question | Decision | Why |
|---|---|---|---|
| Q1 | Down-migrations & app rollback | **Forward-only**, no down-migrations. App rollback is safe **only across additive revisions** — the prior binary still satisfies the DB's `compatible_from` floor (Decision 7), so it validates. Across a **breaking** revision, an app-only rollback fails validation ("DB too new") and requires a DB **restore/export** to the pre-migration state. | Down-migrations double the test surface and are rarely safe on live data; the compatibility floor makes "can I roll back?" a precise, checkable rule. |
| Q2 | Compatibility window | Each package declares `[min_supported_revision, latest_revision]`; startup validates per Decision 7. | Lets one binary span additive revisions; makes too-old / incompatibly-new explicit. |
| Q3 | What may a **patch** release change? | **Additive revisions only**; **breaking** changes require a minor/major. Patch migrations apply via the **standard deploy `migrate` step** (no patch-specific manual step), and because they're additive, rolling deploys and rollbacks stay safe. | Keeps patch upgrades safe and reversible without a special path, and consistent with explicit-migrate (Decisions 6–7) — no silent startup DDL. |
| Q4 | Auto-create schema in **production**? | **No by default for Postgres.** Default startup mode = `validate`; `create`/`migrate` are explicit. Keep auto-`create` default for SQLite/in-memory (dev/test). | Silent DDL-on-import is how the `processing_start_at` crash happened; prod schema should be a deliberate step. lane-agent (greenfield/test) builds in the explicit step from the start. |
| Q5 | Auto-migrate in production? | **Opt-in only**, never on import; always under the backend lock. | Prevents N app instances racing to migrate. |
| Q6 | Multi-instance coordination | Advisory lock during create/migrate; losers wait then validate. | Decision 4. |
| Q7 | App vs Cayu table separation | The `cayu_` prefix (Decision 5), configurable. | Avoids collisions in app-owned DBs. |
| Q8 | ~~Existing unprefixed deployments~~ | **N/A — greenfield.** The baseline revision creates every table with the `cayu_` prefix; there is no unprefixed legacy to migrate. The disposable lane-agent test DB is recreated on the new schema. | No legacy to carry. |
| Q9 | Other SQL backends (MySQL / SQL Server / Oracle) & ORM/migration libraries | **Postgres is the production standard; no ORM, no Alembic.** Add SQLAlchemy **Core** (dialect builder, below the contracts) only on a *third committed production dialect*; a lightweight internal dialect struct is the cheaper step before that. See Decision 8. | Avoids a speculative N-dialect tax (YAGNI); the store contracts already make a new backend a self-contained PR; lock coordination is dialect-specific in every framework, so no library makes it free. |

## Consequences

- **Clean baseline, no migration debt.** Because everything is greenfield, the
  first schema revision simply *is* the correct schema: all tables `cayu_`-prefixed,
  created at the baseline revision. There is no rename migration, no `table_prefix=""`
  back-compat escape hatch, and no live data to preserve.
- **lane-agent gets the clean design from the start.** Its disposable test DB is
  recreated on the `cayu_` schema, and its deploy runs the explicit
  `migrate`/`create` step (Q4) rather than relying on lazy auto-create. Cheap to do
  now precisely because there's no legacy.
- **The advisory-lock init lands first** (Phase-1 step 1) — it fixes the real
  shared-pool race and is the coordination primitive the migrate path also uses.
- **Set the release policy now, while it's free.** Forward-only (Q1), additive-only
  patch releases (Q3), validate-at-startup (Q4), and opt-in migrate (Q5) cost
  nothing to adopt today but are expensive to retrofit after the first real release
  — so they go into the baseline contract.
- Relation to cayu#32: this ADR addresses **schema** evolution. Before 1.0,
  correctness-critical `SessionStore` operations can be required abstract methods
  so incomplete stores fail at instantiation instead of later in a recovery path;
  optional lifecycle additions can still use explicit default implementations.

## Roadmap (incremental)

- **Phase 1 — versioning foundation** ✅ implemented
  1. ✅ Advisory-lock-protected, idempotent schema init (fixes the race now).
  2. ✅ `cayu_schema_migrations` table; record baseline revision.
  3. ✅ `validate`-at-startup with fail-fast on out-of-range revision.
  4. ✅ `cayu_`-prefixed tables in the baseline schema (both backends, revision 1).
- **Phase 2 — migration tooling** ✅ implemented
  1. ✅ `cayu storage migrate` + `cayu storage status`.
  2. ✅ Forward migration framework per backend (`_MIGRATION_STEPS` hook), migrate
     under the lock. The history currently holds only the baseline revision, so
     `pending()` is empty until the first real schema change appends a `Revision`.
  3. ✅ Migrator behavior tests (validate fail-fast, create-records-baseline,
     idempotent migrate, shared-pool single baseline) on a real Postgres + SQLite.
- **Phase 3 — portability**
  1. ✅ JSONL export (`cayu storage export`, backend-agnostic via the store
     contracts); import/replay is future work.
  2. ⏳ Migration dry-run/reporting.
  3. ⏳ SQLAlchemy **Core** (not the ORM, not Alembic) — deferred until the
     Decision 8 trigger (a third committed production SQL dialect). Not on the
     near-term roadmap; Postgres is the production standard.

## Notes

Originally drafted on the local `feat/postgres-session-store` branch. Phases 1–3
(core) are implemented on that branch and proposed for merge into `main` via PR.
Accepted on the greenfield basis above; revisit if a real release introduces
external consumers.
