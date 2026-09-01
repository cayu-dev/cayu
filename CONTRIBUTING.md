# Contributing to Cayu

Thank you for contributing! This guide covers what we're looking for, how to set up a
development environment, and how to get an issue or pull request accepted.

---

## Contribution priorities

Our current maintainer priorities are:

1. **Security fixes and hardening** — exploitable boundary failures are the
   highest-priority bugs and must be coordinated through
   [private reporting](SECURITY.md). Preventative hardening that does not disclose a
   suspected vulnerability can use the normal public contribution process.
2. **Bug fixes** — crashes, incorrect behavior, data loss. Among ordinary bugs,
   durability and recovery correctness (resume, replay, budgets, ledgers) comes first.
3. **Conformance gaps** — cases where a built-in provider, runner, or workspace adapter
   diverges from the shared conformance suites, or where the suites miss a real behavior.
4. **Performance and robustness** — retry classification, graceful degradation, bounded
   queries, recovery paths.
5. **Documentation and examples** — fixes, clarifications, runnable examples.
6. **New integrations** — usually belong outside this repo; see
   [placement policy](#placement-policy-what-lands-in-tree) before writing code.

## Before you start: search first

- Search **open and closed** issues and PRs for your symptom or idea — duplicates are
  common and the tracker can lag the code:

  ```bash
  gh search issues --repo cayu-dev/cayu "<your terms>"
  gh search prs --repo cayu-dev/cayu "<your terms>"   # searches open and closed
  ```

- Search the source too. Many requested capabilities already exist in-tree.
- If an open PR already addresses your problem, review or improve it instead of opening
  a competing one.
- For larger work, comment on the issue first so the approach is agreed before you build.

## Issue taxonomy

Every open issue has exactly one **type**, one **state**, and one **priority**. Types
describe what the work is; states describe whether the project has accepted and
scheduled it.

| Type | Meaning |
| --- | --- |
| `Bug` | A reproduced violation of an existing contract. |
| `Hardening` | Makes existing behavior safer or more reliable under failure, concurrency, recovery, or hostile input. |
| `New Feature` | Introduces a genuinely new caller-visible capability. |
| `Maintenance` | Improves architecture, verification, CI, documentation, examples, releases, or repository health without adding a capability. |

| State | Meaning |
| --- | --- |
| `Proposed` | Reported or suggested, but not yet accepted or scheduled. |
| `Backlog` | Accepted work, but not selected for the upcoming queue. |
| `Ready` | Selected upcoming work that is fully specified and unblocked. |
| `In Progress` | Actively being implemented. |
| `Blocked` | Selected work that cannot currently proceed. |

Priorities are `P0` through `P3`. `P0` is a stop-the-line exception: do it above
everything else. It should be rare, assigned deliberately, and removed as soon as the
emergency is resolved. Priority does not replace state: a high-priority proposal is
still `Proposed` until accepted, and a high-priority accepted issue is not `Ready`
until it is sufficiently specified and selected for implementation.

## Placement policy: what lands in-tree

Cayu's backend extension points — including model providers, runners, workspaces,
vaults, artifact stores, and knowledge stores — are deliberate seams. The in-tree set
of adapters behind each seam is **curated**: it exists to prove the contract and cover
the foundational backends, not to absorb every vendor SDK.

- **Bug fixes to existing in-tree adapters are always welcome.** Provider, workspace,
  and runner changes should run the corresponding repository suite:

  ```bash
  uv run pytest tests/providers/test_provider_conformance.py -q
  uv run pytest tests/workspaces/test_workspace_conformance.py -q
  uv run pytest tests/runners/test_runner_conformance.py -q
  ```

- **New third-party integrations ship as standalone packages.** Implement the public
  contract and test it in that package. The repository suites above are deterministic
  behavioral references for Cayu's built-in adapters; they do not expose an external
  registration API, certify third-party packages, or ship in the installed wheel or
  source distribution. Vaults, artifact stores, and knowledge stores do not yet have a
  shared conformance suite. If a reusable external conformance harness would unblock an
  integration, open an issue so that contract can be designed explicitly.

  A standalone adapter does not need in-tree placement to be a first-class integration;
  in-tree placement adds maintenance burden without adding capability for its users.
- **If your integration needs a capability the seam doesn't expose**, that's a feature
  request to widen the contract — never a special case for one adapter in core.
- A well-built integration PR can be technically flawless and still be redirected to a
  standalone package. That's a placement decision, not a verdict on the code.

## Development setup

Prerequisites: [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11 (CI runs 3.14; uv
installs it on demand). Docker is optional. It is required for container-runner and
container-egress tests and for the default testcontainers-managed Postgres setup; the
Postgres tier can instead use a disposable external database.

```bash
git clone https://github.com/cayu-dev/cayu.git && cd cayu
uv sync --extra dev --extra server
```

### Tests

```bash
uv run pytest                                  # full suite, serial
uv run pytest -n 3 --dist loadfile             # full suite, conservative parallelism
uv run pytest tests/core -q                    # focused runs are fine while iterating
uv run pytest --store-durations --clean-durations  # refresh CI's .test_durations snapshot
```

For deterministic pre-commit, acknowledgement-loss, and ownership-handoff
coverage around durable session operations, use the repository-private
[Session operation publication fault harness](docs/session-operation-fault-harness.md).
It preserves each backend's real transaction boundary and must not be copied into
production code or the installed testing API.

The parallel command is safe with the default testcontainers-managed Postgres setup,
where each worker gets an isolated container. When `CAYU_TEST_POSTGRES_DSN` points at one
shared database, run the suite serially because store tests intentionally clear their
disposable schema between cases.

Postgres tests **skip automatically** when no disposable database is available. Two
ways to run them:

- Have Docker running — the suite provisions a disposable pgvector Postgres via
  testcontainers.
- Or point `CAYU_TEST_POSTGRES_DSN` at a **disposable** database (tests drop tables).

Docker runner and egress tests require a running Docker daemon and skip automatically
when it is unavailable.

CI sets `CAYU_REQUIRE_POSTGRES=1` (and the Docker equivalents) so a lost tier fails
loudly instead of hiding behind skips. You don't need those flags locally, but a PR that
touches stores, runners, or egress should be run at least once with the relevant tier
active.

### Lint and types

CI enforces all three — run them before pushing:

```bash
uv run ruff check src/ tests/ examples/ scripts/ maintenance/
uv run ruff format --check src/ tests/ examples/ scripts/ maintenance/
uv run ty check src/cayu examples maintenance
```

Note: `ruff format --check` is a CI gate, not a suggestion. `ty` is the type oracle for
this repo; don't submit `mypy`/`pyright` suppressions.

### Dashboard

Dashboard changes require Node.js ≥ 22.18.0 and npm. Run the complete dashboard job
from the `dashboard/` directory:

```bash
npm ci
npm run lint
npm run test
npm run typecheck
npm run check:api:repo
npm run build:package
cd ..
uv run python scripts/build_dashboard_source_bundle.py
```

`build:package` regenerates `src/cayu/server/dashboard/`. Commit those packaged assets
when dashboard source changes. The bundle command then regenerates the deterministic editable
source shipped in wheel and source distributions. Commit both outputs; CI fails when the compiled
assets, API baseline, license inventory, release metadata, or source bundle is stale. If a reviewed
dependency license changes, update `dashboard/THIRD_PARTY_LICENSES.md` from the finalized build
before regenerating the source bundle.

### Pull request verification

Every pull request runs the repository-wide static checks and the complete Python 3.14
test suite without coverage instrumentation. A small fail-open path selector additionally
runs the retained SQLite cancellation, package/sidecar, and dashboard gates whenever their
contracts can change. `main` runs every retained gate.
Duration-balanced general tests run across six single-process jobs, while stress and process
tests share an isolated lane and Postgres conformance runs in two duration-balanced jobs. CI
rejects a duration snapshot after more than 5% of collected tests lack timings; refresh it
with the command above before it can materially unbalance the shards.

Release tags rerun every retained gate, add release-only freshness and tag/version checks,
and may publish the exact artifact validated by that release path.

Run that same retained CI contract locally from a clean tracked checkout:

```bash
python3 scripts/run_ci.py --base origin/main --head HEAD
```

The command never fetches, pushes, comments, publishes, or changes remote state. It resolves the
exact checked-out head, rejects tracked worktree or index changes, and executes that commit in a
fresh detached worktree. Source untracked paths are recorded but never copied into the execution
checkout, so they cannot affect imports, test collection, or results attributed to the commit. The
runner revalidates the detached head and tracked tree before issuing successful proof. It runs the
hosted lanes with two concurrent general-shard workers and writes
`/tmp/cayu-local-ci-proof-<sha>.md` with commands, results, durations, pass/skip counts, platform,
and limitations. Use `--all` to retain every optional PR lane regardless of changed paths. Set
`--jobs 1` for the lowest local resource usage or increase it up to `--jobs 6` on a dedicated
machine; specialist, database-conformance, SQLite, dashboard, and package lanes remain serial to
avoid compounding their resource contention.

Publication policy is intentionally separate from ordinary PR readiness. Only a tagged release
checkout with exactly one `v*` tag at `HEAD` may add `--publishing`; that mode includes immutable
tagged-note, source-version, and release-freshness validation. GitHub Actions invokes the same
canonical lane implementations from `scripts/run_ci.py`. Package commands live in the canonical
`scripts/package_ci_steps.yml` manifest consumed by that runner; ordinary PR verification records
the publication-only steps as explicit skips.

## Pull request process

### Keep PRs focused

One logical change per PR. Don't mix a bug fix with a refactor with a feature. Large
mechanical changes (renames, formatting) go in their own PR so review diffs stay honest.

### Requirements

- **Tests pass locally** (`uv run pytest`) along with ruff and ty (commands above).
- **Bug fixes require a regression test** that fails without the fix.
- **Features require tests and doc updates** (README, `docs/`, or docstrings — wherever
  the contract lives).
- **Contract changes require updating `docs/runtime-contracts.md`** when they touch a
  documented runtime boundary.
- **New dependencies need justification** in the PR description. Version constraints
  must reflect the compatibility range Cayu verifies and must not assume compatibility
  across breaking releases. Runtime (non-dev) dependencies need maintainer sign-off
  before you build on them.

### Commit messages

Imperative subject line, ≤ 72 characters, body explains *why* rather than *what*.
Conventional-commit prefixes (`fix(runtime): …`, `feat(workspaces): …`) are welcome and
match much of the existing history, but the imperative subject is the hard rule:

```
Fix budget reservation liveness and window accounting
feat(workspaces): bound SyncBinding tar transfers
```

### Branch naming

```
fix/short-description
feat/short-description
docs/short-description
```

### Maintainer releases

PyPI publishing requires external trusted-publisher, environment-protection,
and immutable-tag configuration in addition to the checked-in workflow. Follow
[the release runbook](docs/releasing.md) exactly; in particular, do not push a
release tag until every one-time security prerequisite is active.

## Reporting issues

Use the issue templates. The short version:

- **Bugs**: cayu version, Python version, OS, which provider/runner/store you were
  using, a minimal reproduction, and the full traceback. A failing test is the best bug
  report there is.
- **Features**: describe the problem before the solution, and check the
  [placement policy](#placement-policy-what-lands-in-tree) if it's an integration.
- **Security vulnerabilities**: never open a public issue, pull request, discussion, or
  Discord thread — use [private reporting](SECURITY.md).

## Community

- **Discord**: [discord.gg/jWa3kKJ7R8](https://discord.gg/jWa3kKJ7R8) — questions,
  show-and-tell, and discussion that isn't an actionable bug or feature yet.
- **GitHub issues** — actionable bugs and concrete feature proposals.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](LICENSE), the same license that covers the project.
