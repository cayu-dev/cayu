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

Target resolution only identifies and validates the requested backend. It does
not search arbitrary directories, import an app factory, create a database, or
run migrations. Read-only commands open the resolved target under their own
non-mutating backend contract.
