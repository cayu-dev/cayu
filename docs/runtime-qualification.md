# Runtime stability qualification

The registered local suite runs deterministic Runtime contract and capacity
fixtures against an **installed Cayu wheel**. It is release qualification, not an
agent-quality benchmark. The repository supplies the versioned fixture registry;
fixtures and their private fault harness are not installed as Runtime APIs.

## Choose the evidence for your question

Start with the small acceptance plan, then the application integration tests.
Neither is the broader registered release qualification suite below.

| Evaluator question | Existing command / owning evidence | Expected observation and limits |
| --- | --- | --- |
| Does a model-requested tool receive the intended arguments and return its result? | [Deterministic acceptance](#deterministic-acceptance): `tool_roundtrip` in [runtime_acceptance.py](../src/cayu/evals/internal/runtime_acceptance.py); [Evals reference](evals.md#first-party-runtime-acceptance-suite). | Seven cases pass; inspect the JSON case and assertion results. Scripted providers establish runtime mechanics, not model quality or live-provider conformance. |
| Does durable state survive a fresh process and support continuation? | [Support integration](#support-integration): `test_restart_approval_repeat_and_independent_verification` in [test_order_support.py](../tests/examples/test_order_support.py); [canonical journey](../src/cayu/guides/order-support.md). For abrupt death, run [focused crash qualification](#focused-crash-and-release-qualification). | Separate CLI processes reopen SQLite sessions and service records after normal exit. Transcripts, pending approval, and effects survive. The example does not simulate a crash; the `fresh-process` registry scenario adds real SIGKILL boundaries. |
| Does approval prevent effects before resolution and after denial? | The same support test, parameterized for approve and deny. | Independent service-database queries find no replacement before delivery or after denial, one after approval, and no additional effect or events after repeated receipt delivery. |
| Is execution bound to the reviewed proposal? | Support integration: `test_invalid_receipts_and_changed_version_never_execute`; [approval responsibilities](../src/cayu/guides/order-support.md). | Altered signatures, unknown signed receipts, wrong conversation, changed execution version, and mismatched native call identity are rejected without effects. Application code authenticates and binds the receipt; Runtime owns pending approval and execution. The local signing fixture is not a production authentication service. |

## Pin a clean installed artifact and matching fixtures

Requirements: Git, uv, Python 3.11+, a writable temporary directory, and a POSIX
host for the crash qualification. Dependency installation and building may need
network access; the selected scenarios require no credentials or network calls.
Docker, PostgreSQL, and live-provider lanes are separate and are not exercised by
these commands. The integration example uses SQLite; the acceptance plan uses
its own temporary stores/workspaces.

Use a full commit SHA for `REV` (or resolve a release tag to its commit first).
Build from a clean checkout of that revision so the wheel and repository-owned
fixtures match. A package version alone cannot identify an unreleased build.
Do not pair an arbitrary published wheel with current-main fixtures. Fixtures,
pytest tests, and qualification scripts do **not** ship in the installed package.

```sh
REV=<full-cayu-commit-sha>
RUN=$(mktemp -d)
git clone https://github.com/cayu-tech/cayu.git "$RUN/checkout"
git -C "$RUN/checkout" checkout --detach "$REV"
cd "$RUN/checkout"
git rev-parse HEAD > "$RUN/fixture-commit.txt"
uv build --out-dir "$RUN/dist"
# A new output directory ensures there is exactly one candidate wheel.
set -- "$RUN"/dist/cayu-*.whl
test "$#" -eq 1 && test -f "$1"
WHEEL=$1
uv venv "$RUN/env"
PY="$RUN/env/bin/python"
uv pip install --python "$PY" "$WHEEL[dev]"
uv pip freeze --python "$PY" > "$RUN/installed-requirements.txt"
"$PY" -c 'import hashlib, pathlib, sys; p=pathlib.Path(sys.argv[1]); print(hashlib.sha256(p.read_bytes()).hexdigest(), p.name)' "$WHEEL" > "$RUN/wheel-sha256.txt"
mkdir "$RUN/evidence"
cd "$RUN/evidence"
unset PYTHONPATH PYTEST_ADDOPTS
export PYTHONNOUSERSITE=1
"$PY" - <<'PYTHON' > "$RUN/import-identity.txt"
import importlib.metadata
from pathlib import Path
import cayu
from cayu.build_provenance import current_runtime_build_provenance
package = Path(cayu.__file__).resolve()
dist = importlib.metadata.distribution("cayu")
assert package == Path(dist.locate_file("cayu/__init__.py")).resolve()
build = current_runtime_build_provenance()
assert build.origin.value != "development_source_tree"
print("version:", dist.version)
print("package:", package)
print("build:", build)
PYTHON
```

Keep the wheel, fixture commit, wheel SHA-256, dependency freeze, import identity,
and reports together. The freeze records resolved dependencies; reuse those
versions when repeating this environment. The qualification report independently
records installed build and fixture fingerprints.

### Deterministic acceptance

From the empty evidence directory, use the installed executable explicitly:

```sh
"$RUN/env/bin/cayu" eval run cayu.evals.internal.runtime_acceptance:build \
  --case-timeout-seconds 30 --output "$RUN/evidence/runtime-acceptance.json"
```

Expect seven passing cases. Inspect each case's status and assertions in the
existing eval JSON report; this plan does not cover multi-phase approval resume,
SIGKILL, live providers, or full release gating. See [Evals](
evals.md#first-party-runtime-acceptance-suite) for the case inventory.

### Support integration

Stage the matching repository tests without `src`, then run pytest with the
installed interpreter. `CAYU_EXAMPLE_PYTHON` alone selects only the child CLI
processes: the parent test process also imports Cayu to inspect durable records.
Running ordinary checkout pytest can import checkout source via `pythonpath`.
This staging recipe makes both parent and children use the wheel.

```sh
cp -R "$RUN/checkout/tests" "$RUN/evidence/tests"
cp "$RUN/checkout/pyproject.toml" "$RUN/evidence/pyproject.toml"
cd "$RUN/evidence"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CAYU_EXAMPLE_PYTHON="$PY" \
  "$PY" -m pytest -q tests/examples/test_order_support.py \
  --junitxml="$RUN/evidence/order-support.xml"
"$RUN/env/bin/cayu" guide order-support > "$RUN/evidence/order-support-guide.md"
"$PY" -m cayu.examples.order_support --help
```

Expect three passed tests, zero skips, and exit status 0. The JUnit report records
individual test results; pytest assertion output diagnoses failures. The tests
inspect session events/transcripts and service SQLite rows independently of CLI
prose. For inspectable application state, follow the installed guide's commands
in a fresh directory; it identifies the databases and review/receipt files.
Normal process exits between commands are the restart boundary, with a 90-second
subprocess timeout. This route does not claim PostgreSQL or Windows verification.

### Focused crash and release qualification

For a bounded first crash check using existing report semantics:

```sh
"$PY" "$RUN/checkout/scripts/run_runtime_qualification.py" \
  --python "$PY" --scenario fresh-process --repeat 1 \
  --report "$RUN/evidence/runtime-qualification-focused.json"
```

Expect `status: passed`, `scope: focused`, and successful selected case results.
The [registry](../tests/qualification/registry.py) selects
[real SIGKILL recovery tests](../tests/recovery/test_sigkill_recovery.py) at model
dispatch, tool effect, approval, task-claim, and attachment boundaries, plus the
pending-approval terminal-publication invariant. Read the report's case/phase
dispositions; a skip, failure, or unavailable prerequisite is not evidence of success.

For the full default release profile, omit the scenario and repeat overrides:

```sh
"$PY" "$RUN/checkout/scripts/run_runtime_qualification.py" \
  --python "$PY" --report "$RUN/evidence/runtime-qualification.json"
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
five-minute wall-clock bound (ten minutes for stress), except capacity, which gets
fifteen minutes (twenty minutes for stress) to allow for disk contention. Each
scenario has a bounded cleanup grace period. No provider credentials, paid calls, Docker, or
PostgreSQL are required. POSIX process groups and real SIGKILL are required for
fresh-process recovery; unsupported hosts produce prerequisite failure, not a pass.

The default `docker-allocation` scenario uses simulated Docker calls to qualify
immutable attachment lifetimes without a daemon. For the real Docker continuation
and fresh-process controls, explicitly enable the local Docker lane with an existing
coding image (no image build or pull is performed):

```sh
CAYU_DOCKER_CODING_IMAGE=<existing-coding-image> \
"$PY" "$RUN/checkout/scripts/run_runtime_qualification.py" \
  --python "$PY" --docker \
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
"$PY" "$RUN/checkout/scripts/run_runtime_qualification.py" \
  --python "$PY" --profile stress \
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
