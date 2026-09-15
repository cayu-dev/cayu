# Native benchmark campaigns

A benchmark campaign is an immutable admission receipt over the existing
authored-suite catalog, `EvalRunSpec` records, trial checkpoints, and native
Evals coordinator. It adds no scheduler, provider adapter, or infrastructure
provisioner. See [benchmark packages](benchmark-packages.md) for the portable
suite, scenario, input-file, and scorer contracts.

## Installed command path

Install `cayu[server,files]` for native durable execution and image/file inputs.
The server extra supplies the existing coordinator and registry; the CLI does
not start an HTTP listener. Package creation/validation and offline inspection
also work with the base wheel.

These commands use only the installed synthetic provider and local SQLite and
artifact stores. They make no external model, tool, or judge calls:

```sh
cayu eval package init package
cayu eval package validate package
cayu eval run cayu.evals.benchmark_synthetic:build_synthetic_benchmark_plan \
  --package package --campaign-directory campaign \
  --trials 2 --max-concurrency 2 --output result.json --html-output result.html
cayu eval status campaign --json --sessions
cayu eval failures campaign
cayu eval report campaign --output report.html
cayu eval export campaign --output evidence.zip
cayu eval status evidence.zip --json
```

The full synthetic package deliberately includes a wrong answer: its completed
launch exits **1**. Select `--case echo --case attachment` for an all-passing
cohort. Repeated `--case` flags select an exact cohort without editing the
package. Exit **0** means a passed execution or successful inspection;
**1** means a completed, failed evaluation; **2** means invalid admission,
incompatible comparison, or incomplete/cancelled/error execution. Static
rescoring exits 0 when all requested assertions were evaluated, including
assertions that failed, and 2 when required facts were unavailable.

Campaign status applies the admitted minimum-pass threshold per case; individual
failed trial rows remain visible even when the case passes.

`--admit-only` saves the input snapshot, catalog, resolved configuration, and
idempotent native run requests without dispatch. Start it with:

```sh
cayu eval resume campaign cayu.evals.benchmark_synthetic:build_synthetic_benchmark_plan
cayu eval cancel campaign
```

Use an application `module:factory` or the existing project's eval target in
place of the synthetic factory. It must return an `EvalPlan` with a
`corpus_target` or `workflow_target`. The package target key must match that
trusted target. The plan's `execution_profile_policy` selects its application
reset/isolation contract and ceilings; the native registry resolves and binds
the exact execution profile. An undeclared policy permits one trial and one
concurrent execution. Repeated or concurrent trials require the existing
application-managed reset contract and stable isolation revision.

The launcher supports `--provider` plus `--model`, `--environment`, `--trials`,
`--minimum-passed-trials`, `--max-concurrency`, `--case-timeout-seconds`,
`--max-steps`, `--max-total-tokens`, `--max-tool-calls`, and
`--max-estimated-cost` with optional `--currency`. A model override selects an
already registered provider. Environment selection never provisions an
environment. Bounds can narrow existing target authority, not broaden it.
Unsupported providers, missing tools/files, excessive trials/concurrency,
missing pricing, and ambiguous flag combinations fail preflight.

Before dispatch, stderr identifies the campaign, package/cohort revisions,
admission path, canonical inspection command, settings, resolved execution
profile revisions, and maximum admitted work. `campaign.json` retains the exact
native specifications, model/environment/profile snapshots, trial policy,
candidate and judge exposure, scorer identity, and recovery/retry policy.

## Concurrency and budgets

`--max-concurrency` is the launch total, not a multiplier per worker. Native
authored-suite lanes serialize groups when the launch has more native runs than
available lanes. Within a run the existing trial scheduler and execution
capacity own concurrency. These workers claim only the campaign's launch
revision, so another queued launch sharing the target is not dispatched.

Package checks, target/profile validation, and finite work admission happen
before candidate dispatch. Runtime step/tool/token limits and priced stop
policies remain per trial. Observed-token and estimated-cost thresholds can
overshoot during in-flight work; they are not strict provider billing caps.
Priced limits require the target's trusted `PriceBook`. Missing usage or pricing
is unavailable, not zero. Candidate and judge allowances remain distinct.

Each authorized execution attempt can consume another full trial allowance.
For `N` original trials, `E` permitted execution attempts and `R` reserved
selective retries per original trial, the conservative global candidate
allowance is `N * E * (1 + R)`. The CLI prints this value. The same multipliers
apply to the admitted step/token/judge work ceilings; they do not turn an
observed threshold into a strict billing cap. Recovering a completed checkpoint
consumes no new candidate or judge work.

## Restart and selective retry

The default `checkpoint_only` recovery policy permits one execution attempt.
After process loss, a new native lease preserves every completed checkpoint
and publishes each uncheckpointed slot as `recovery_reexecution_blocked`, with
unavailable assertions and no invented output. Even a slot that may not yet
have started is conservatively blocked once ownership was lost. Provisional
session observations remain separately labeled and cannot replace an exact
published-trial link.

`--max-execution-attempts N` explicitly authorizes up to 10 full execution
attempts for uncheckpointed work, including its possible repeated effects and
spend. It must be selected at original admission. Claim epochs enforce the
finite bound across process restarts; concurrent resume uses native ownership
fences. Old owners cannot publish or renew a reclaimed claim. Native pending
approvals are not silently approved, and this campaign recovery path does not
infer safe replay from a partially completed scenario. If replay was not
authorized, those ambiguous slots remain blocked. A changed or process-local
execution identity can reject resume before dispatch.

Selective retry is opt-in and operator-selected:

```sh
cayu eval package init failure-package --failure-modes
cayu eval run cayu.evals.benchmark_synthetic:build_synthetic_benchmark_plan \
  --package failure-package --campaign-directory failure-campaign \
  --max-retry-attempts 1 --retry-backoff-seconds 0
cayu eval retry failure-campaign \
  cayu.evals.benchmark_synthetic:build_synthetic_benchmark_plan \
  --case provider-failure --trial 1
```

The synthetic provider fails once, then its selected successor passes. The
original result stays failed. The separate malformed synthetic judge remains
a scoring failure. The default eligible categories are `timeout`,
`provider_failure`, `environment_failure`, and `recovery_blocked`.
`--retry-category` explicitly replaces that list; answer mismatch, ordinary
execution failure, scoring/capture failure, cancellation, and unavailable
evidence are never implicitly opted in.

`--max-retry-attempts` reserves 0–3 successor attempts **per original trial**.
`--attempt` selects a reserved ordinal; attempts after the first require a
terminal, non-passing predecessor. Backoff is measured from the preceding
attempt's durable finish time, is bounded at admission to 0–3600 seconds, and
returns the next eligible time rather than running a retry loop. A target must
have the same admitted execution identity and an application-managed reset
contract, or the retry caller must explicitly pass `--allow-reexecution`.
Unknown/effectful work is not declared idempotent by Cayu.

Successor directories and run IDs are deterministic per original revision,
case, trial, and reserved attempt. A repeated or competing command reuses that
single admission. Admission is staged privately and published atomically only after
its catalog, receipt, and native run admissions are complete. Interruption before
publication leaves the retry slot available; no staged work is executed. Successors
cannot allocate another retry tree. Native run
invocations and campaign receipts retain the original run/trial revision,
attempt ordinal, failure category, and replay decision. Inspecting the original
campaign lists bounded successor summaries with their observed usage and cost
availability. Exhausted allowances reject without dispatch; earlier results
and spend are preserved.

## Evidence and retention

Keep the campaign directory and its linked application SessionStore and
ArtifactStore for the inspection lifetime. The campaign records the target's
evidence policy and capture bounds before launch; it does not enlarge those
bounds or reconstruct expired evidence. Input files are bounded snapshots
copied into the campaign and materialized through the target's existing static
environment ArtifactStore.

Campaign admissions opt into native private checkpoint retention. Existing
checkpoint item/byte bounds continue to apply. Terminal publication/cancellation
retains redacted checkpoint records for exact session linkage; raw candidate
output is prohibited in these checkpoint records. Default non-campaign runs
continue clearing checkpoints at terminal settlement. Retention lasts as long
as the operator retains the private EvalStore; no background cleanup policy or
provider result retention is implied.

CLI inspection combines native results and exact checkpoint links with bounded
process observations. It distinguishes published, provisional, and unavailable
results, and keeps execution failure, answer mismatch, capture/scoring failure,
cancellation, timeout, and missing evidence separate. New agent trial records
carry an optional typed `execution_failure_category`; `usage_evidence_state`
distinguishes complete observed usage, a partial observed prefix, and missing
usage. Legacy records remain readable. Known estimates retain their currency
and priced/unpriced step counts. Scorer diagnostics keep separate judge
accounting. None of these observations promises complete provider billing.

`--sessions`, `--max-sessions`, and `--max-diagnostics` follow available exact
SQLite links through the existing bounded session-inspection API. Messages,
tool arguments/results, child executions, workspace/artifact references, and
failed-workflow record references remain governed by the existing capture,
redaction, and retention contracts. Missing stores, incomplete traces, and
truncated inspection are explicit limitations. Provisional prior-session
references are diagnostic leads, not authoritative successful trajectories.

Attach the same EvalStore and application target to the existing dashboard:

```python
from cayu import SQLiteEvalStore
from cayu.server import EvalsConfig
from cayu.storage.migrations import SchemaMode

evals = EvalsConfig(
    target=plan.corpus_target,
    store=SQLiteEvalStore("campaign/evals.sqlite3", schema_mode=SchemaMode.VALIDATE),
    execution_profile_policy=plan.execution_profile_policy,
)
# Pass evals to the application's existing protected ServerConfig.
```

Open the run's `/cayu/evals?tab=runs&target=...&run=...` link. Evals shows its
native outcome, assertions, output preview, dimensions, timing, usage/cost,
execution-attempt bound, and retry lineage. **Open trial session** follows the
private checkpoint link to the existing session inspector. The authenticated
result response verifies that every link matches the published source trial
revision. Private session IDs remain absent from portable result JSON/HTML.
HTML campaign links assume a dashboard serving the same EvalStore.

The ZIP export contains exactly `inspection.json` and a byte-count/SHA-256
manifest. It is a private bounded inspection snapshot, not a portable corpus or
a sweep of stores/logs. It excludes reference answers, judge rubrics, tool JSON,
credentials, and package truth; output previews and private session locators
retain their existing redaction boundary. It includes original identities and
bounded successor summaries. `cayu eval status evidence.zip --json` validates
the envelope and opens no recorded store path. Offline `--sessions` is rejected;
the export does not promise embedded execution traces. A separate interchange
format is unnecessary for this workflow.

## Comparison and saved-output rescoring

```sh
cayu eval compare campaign campaign --output comparison.json
cayu eval package init scorer-v2 --scorer-version 2
cayu eval rescore campaign --package scorer-v2 --output rescored.json
```

Campaign comparison binds package/version/revision, exact selected cohort,
scorer definitions, model/agent/environment/tool and reset identities, budgets,
trial settings, and recovery/retry policy. Mismatches are **incompatible**;
unpublished or unscored trials make an otherwise compatible comparison
**unknown**. Neither produces score deltas. Comparable campaigns retain paired
source trial revisions and honor the ordinary `--score-tolerance`. Different
cohorts are not compared as if their scores meant the same thing.

Static rescoring requires a new scorer identity/version and the exact original
selected stimuli, scenarios, and file inputs. It uses the existing detached
assertion evaluator over complete retained redacted output and recorded root
status facts. Truncated output and unsupported/missing facts remain unavailable.
The new receipt binds the source campaign, source trial and output digests,
scorer package revision, and each new assertion revision. Source results and
their execution status are unchanged; failed execution is not promoted into a
successful run. The operation imports no application target and calls no
provider, workflow, tool, or judge. New model-backed judging is rejected before
scoring; it requires its own explicit, attributed, budgeted native evaluation
authority and is outside this static command.

## Supported modes and acceptance map

| Capability | Native implementation | Acceptance evidence |
| --- | --- | --- |
| Versioned packages, exact cohorts, real file inputs | `benchmark_package`, authored suites, scenario V2 | `test_benchmark_package.py`; wheel package validation and attachment execution |
| Installed launch, models/profiles, finite work admission | `benchmark_campaign`, CLI adapter, suite preflight and native coordinator | `test_benchmark_campaign.py`, `test_benchmark_cli.py`; wheel 2-trial/2-concurrency launch |
| Failed workflow activity/capture | Existing `workflow_failure` and runner projection | `test_workflow_failure_projection.py`: before/after work, reopen and bounded evidence |
| Failure mapping, selective retry and finite backoff/allowances | Result contracts, `benchmark_retry`, native invocation lineage | Typed failure/retry tests; wheel provider failure, malformed judge, no wrong-answer retry |
| Restart, stale/competing owners, checkpoints, cancellation | Existing native claims and checkpoints, recovery policy | Competing resume and exhaustion tests; store conformance; wheel SIGKILL during pending provider work |
| CLI/dashboard evidence and private links | Existing session inspection and result APIs, retained native checkpoints | `test_server_benchmark_campaign.py`; exact link rejection; wheel inspect/report/export/offline reopen |
| Comparability and static rescoring | Profile comparison identities and existing detached assertions | Campaign comparison/rescore tests; installed wheel source-preservation and call counters |

The packaged CLI currently owns a new local SQLite campaign directory. The
underlying retention/claim contracts are shared with in-memory and PostgreSQL
stores; their conformance is tested separately. This CLI does not expose remote
campaign storage, automatically provision providers/environments, or reinterpret
the existing process-run mode as resumable. File/lifecycle scenarios require a
native agent `CorpusTarget`; text-only packages also accept a
`WorkflowEvalTarget`. Those unsupported combinations reject before execution.

Run the credential-free installed-wheel gate from a checkout after building the
candidate wheel:

```sh
uv build --wheel
CAYU_BENCHMARK_WHEEL=/absolute/path/to/dist/cayu-0.5.2-py3-none-any.whl \
  uv run pytest tests/qualification/test_benchmark_campaign_wheel.py -q
```

The server package CI lane runs this gate against its built wheel. The gate
installs the actual wheel into a new virtual environment, uses only
installed CLI commands, and verifies candidate/judge call counters across
launch, selective retry, inspection/export, comparison, static rescoring,
cancellation, and real process loss/restart. The process-loss slice is POSIX.
It establishes implementation behavior, not agent accuracy or performance
superiority over another benchmark framework.
