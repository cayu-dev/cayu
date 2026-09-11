# Runtime stability qualification

The registered local suite runs deterministic Runtime contract and capacity
fixtures against an **installed Cayu wheel**. It is release qualification, not an
agent-quality benchmark. The repository supplies the versioned fixture registry;
fixtures and their private fault harness are not installed as Runtime APIs.

From a checkout matching the candidate build:

```sh
uv build --wheel
uv venv /tmp/cayu-qualification-env
uv pip install --python /tmp/cayu-qualification-env/bin/python 'dist/cayu-0.5.2-py3-none-any.whl[dev]'
python scripts/run_runtime_qualification.py \
  --python /tmp/cayu-qualification-env/bin/python \
  --report runtime-qualification.json
```

The runner copies repository fixtures to a temporary directory **without `src`**,
clears inherited Python and pytest path/options overrides, and verifies the imported
package against installed distribution metadata. Each scenario checks the same
Runtime build fingerprint. Editable source imports are rejected. An
unavailable identity remains explicitly unavailable. The report also binds the
registry manifest digest and fixture-content digest; use the matching checkout to
interpret hashed test IDs. `fixtures_recipe: cayu.qualification-fixtures.v2`
hashes relative paths and content digests for every staged test file, including
non-Python guide and corpus assets, plus the staged project configuration and
runner script. Temporary-directory names are not part of that identity.

The default profile repeats every scenario twice. Each scenario process has a
five-minute wall-clock bound (ten minutes for stress), followed by a bounded cleanup grace period. No provider credentials, paid calls, Docker, or
PostgreSQL are required. POSIX process groups and real SIGKILL are required for
fresh-process recovery; unsupported hosts produce prerequisite failure, not a pass.

The default `docker-allocation` scenario uses simulated Docker calls to qualify
immutable attachment lifetimes without a daemon. For the real Docker continuation
and fresh-process controls, explicitly enable the local Docker lane with an existing
coding image (no image build or pull is performed):

```sh
CAYU_DOCKER_CODING_IMAGE=<existing-coding-image> \
python scripts/run_runtime_qualification.py \
  --python /tmp/cayu-qualification-env/bin/python --docker \
  --scenario docker-allocation-live --repeat 1 \
  --report runtime-qualification-docker.json
```

This lane requires Docker availability rather than accepting skipped fixtures.
It verifies the three recreation controls, model-step interrupted handoff and
continuation without a duplicate effect, concurrent fresh-process reconstruction,
exact container disposal, and immutable reference counts.

The explicit stress profile adds 100 concurrent durable sessions/task workers,
100 concurrent environment operations, a single 100-call model-authored round,
and 200 empty workers:

```sh
python scripts/run_runtime_qualification.py \
  --python /tmp/cayu-qualification-env/bin/python --profile stress \
  --report runtime-qualification-stress.json
```

Stress requires at least 512 file descriptors and recommends four CPUs and 4 GiB
available RAM. These requirements are included in the report. Stress is never
selected by the default command or automatically added to CI.

For PostgreSQL authority and fresh-process parity, provision a **disposable** database,
set `CAYU_TEST_POSTGRES_DSN`, and add `--postgres`. The configured account must
have database creation privileges. The outer runner owns each scenario database
before dispatching its creation and drops that exact database after its subprocesses
settle, including when pytest exits abruptly or is forcibly terminated. Provisioning
and deletion use the selected interpreter with bounded connection, statement, and
process deadlines. The runner verifies deletion and includes its result even when
pytest writes no report. Repeated runs cannot inherit damaged test rows. Database
cleanup failures fail qualification. Missing PostgreSQL prerequisites cannot silently downgrade to SQLite.
The SQLite report does not claim PostgreSQL qualification.

`--scenario NAME` and `--repeat 1` support focused diagnosis. Focused reports are
marked `scope: focused`; they are not full qualification evidence. A release should
retain both full profile reports and the PostgreSQL result when that backend is
supported. There is no claim of live-provider capability conformance.

## Coverage and interpretation

`tests/qualification/registry.py` names the invariant and first durable boundary
for every scenario. It composes existing focused tests with dedicated concurrent
trajectories rather than redefining Runtime orchestration in a second harness:

- Persistent SQLite sessions and workers run alternating sequential/parallel rounds;
  a barrier proves simultaneous execution. The dynamic trajectory repeatedly uses a
  discovered tool reference. The separate long-loop and 12-large-result compaction
  fixtures check request bounds and atomic call/result grouping.
- The `SessionOperationFaultHarness` supplies pre-transform, pre-commit,
  commit-then-raise and barrier schedules. SQLite and PostgreSQL retain their real
  transactional implementation.
- The completed-session contract replay proves no-effect replay. It does not
  convert interrupted or unsupported trajectories into completed replay evidence.
- Provider-native history/redaction, attachment retention, semantic and absolute
  deadlines, reconnect state, invalid completions and ambiguous dispatch recovery
  use deterministic provider fixtures.
- Real SIGKILL tests reconstruct the registered application against persistent stores
  across model dispatch, tool effects, approval and task-claim/attachment boundaries.
  Checkpoint crash fixtures cover publication, and environment fixtures cover
  contention, cancellation, retained cleanup and exact retries.
- Worker authority and operator-recovery fixtures verify stale-owner rejection,
  bounded recovery, and queued-input preservation through interrupted handoff.
- Seeded negative controls disable semantic deadlines, introduce stale worker
  authority, duplicate a tool effect, retain active tool work, and hot-poll. Each
  must fail the same invariant assertions used by positive fixtures.

The version 1 JSON report contains only registry text, build identity, hashed case
identities, phase dispositions, elapsed times and allowlisted integer counters.
Prompts, tool arguments/results, raw assertion output, exception messages, local
paths and credentials are not included. Counter evidence is scenario-specific:
absence of a counter means **not measured**, not zero. Terminal session/task rows
are expected durable history; they are not resource leaks. Owned subprocess groups
are checked even after their leaders exit. A surviving descendant fails the scenario;
cleanup drains the group with a bounded wait and reports any groups or tracked
processes still remaining afterward. Launches also append owned group IDs to a
private journal held by the outer runner. After forced pytest termination, the
runner drains those groups even when pytest never reaches its reporting hook.
A launch whose group cannot be registered is terminated before its handle is
returned to the fixture. Repetitions use fresh
stores and verify the same completion/claim/tool cleanup invariants each time.

A skipped selected test, collection failure, timeout, wrong build, missing result,
or failed assertion fails the scenario. Exit status is 0 for a passing selected
scope, 1 for failed qualification, and 2 for unavailable prerequisites/infrastructure.
For local diagnosis, run the named registry selector with pytest in a controlled
checkout; its ordinary failure output may contain fixture payloads and is not the
content-safe qualification artifact.
