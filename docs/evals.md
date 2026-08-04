# Runtime-Native Evals

Cayu evals are designed to test agent behavior through the Cayu runtime, not
only final model text.

The goal is two-part:

- provide stable abstractions so applications can bring their own eval stack
- provide a simple local/CI default for teams that want something built in

The built-in runner evaluates normal `CayuApp.run(...)` sessions and then
asserts over the durable runtime state Cayu already owns: sessions, events,
transcripts, tool calls, usage, workspaces, and artifacts.

## Minimal Example

```python
from cayu import (
    AgentSpec,
    CayuApp,
    EvalCase,
    EvalSuite,
    FinalOutputContains,
    Message,
    RunRequest,
    ScriptedModelProvider,
    SessionCompleted,
    run_eval_suite,
)
from cayu.providers import ModelStreamEvent


async def main():
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="agent", model="fake-model"))

    suite = EvalSuite(
        id="basic",
        cases=[
            EvalCase(
                id="says-done",
                request=RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "say done")],
                    max_steps=1,
                ),
                assertions=[
                    SessionCompleted(),
                    FinalOutputContains("done"),
                ],
            )
        ],
    )

    result = await run_eval_suite(app, suite)
    assert result.status == "passed"
```

## CLI

`cayu eval run [module:attribute]` loads a Python target. The target should
return one of:

- `EvalPlan(app=app, suite=suite)`
- `(app, suite)`
- an object or dict with `app` and `suite`

Projects can declare one default alongside their application factory:

```toml
[tool.cayu]
factory = "app:build_app"
eval_target = "evals.assistant:build_eval"
```

Running `cayu eval run` without a target searches upward for the nearest Cayu
project, changes to that project root, and resolves `eval_target` with the root
first on Python's import path. Relative stores, fixtures, and output paths
therefore resolve consistently when the command starts in a nested directory.
The original working directory, import path, and project modules are restored
when the command finishes.

An explicit target overrides project configuration and retains the caller's
current directory as its root. Use this form for additional suites or for evals
outside a configured Cayu project. Targets may be synchronous or awaitable; the
application `factory` is not used as an eval target because it does not identify
an `EvalSuite`.

Example:

```bash
cayu eval run --output results.json
cayu eval run evals.reviewer:build_eval --output reviewer-results.json
cayu eval report results.json --format html --output eval-report.html
cayu eval compare baseline.json results.json --output comparison.json
```

The command exits with `0` when all cases pass and `1` when the run fails,
errors, or a comparison detects regressions. `eval report` and `eval compare`
operate only on the paths supplied to them and do not perform project
discovery.

## Portable corpus documents

`EvalCorpusDocument` is Cayu's bounded, JSON-portable definition format for
reusable eval suites and cases. A document describes exactly one trusted
`target_key`. It contains only user-role text input, bounded trial settings,
diagnostic source/pricing identities, an explicit evidence policy, and a closed
set of structural assertion specifications. It cannot contain a `CayuApp`,
provider/model/environment selection, import path, callback, raw session ID, or
runtime event payload.

The portable assertion kinds in schema version 1 cover root and child terminal
status, final-output equality/containment, tool presence/order/count, model-step
and token limits, recorded usage, and estimated-cost limits. Cost assertions
require a `PricingProfileIdentityV1`; the identity fingerprints trusted pricing
used elsewhere and never embeds or authorizes a `PriceBook`.

Corpus documents are definitions, not executable application configuration.
Parsing one never imports project code or invokes a provider, tool, environment,
hook, or runtime. A trusted caller resolves `target_key` to local application
bootstrap code and separately verifies the diagnostic source identity before
execution.

Use the `.create(...)` factories for `EvalSuiteSpec`, `EvalCaseSpec`, and
`EvalCorpusDocument`. They validate and canonicalize their inputs and compute
`sha256:` content revisions. Suite revisions cover reusable suite settings;
cases reference suites by `suite_id`, so cases from independent corpus fragments
can be merged without rewriting suite membership. Case, suite, evidence-policy,
and corpus revisions change whenever their covered content changes;
`assertion_spec_revision(...)` provides the same identity for one assertion.

`eval_corpus_to_json(...)`, `eval_corpus_from_json(...)`, and
`load_eval_corpus(...)` enforce schema version 1 and Cayu's durable-JSON rules,
including duplicate-key, non-finite-number, integer-range, Unicode, and nesting
validation. Input is rejected before an unbounded read or decode. The hard
document limit is 8 MiB, with at most 64 suites, 1,000 cases, 64 assertions per
case, 16 messages per case, 65,536 characters per message, 262,144 input
characters per case, 100 sequential trials, and a 3,600-second per-trial timeout.
Each suite may expand to at most 10,000 published assertion results across all
of its cases and trials, so every accepted suite fits the complete 32 MiB public
result graph instead of failing only after execution.
Unknown fields and assertion kinds fail closed; schema version 1 has no legacy
compatibility loader.

Portable assertions consume one immutable `AssertionEvidenceView`, produced by
`project_assertion_evidence_view(...)` from a validated `Trajectory`. The view
contains only terminal statuses, bounded redacted final output, requested tool
names and counts, model-step/token counts, and optional currency-local cost
totals. It carries no session, event, interaction, provider, model, agent,
environment, payload, tool argument/result, or cost line-item identity. Child,
output, tool, model-step, and usage completeness are explicit; unavailable or
limit-exceeded evidence is never represented as a complete observation. The
public projector derives `root_evidence_available` from the durable root session,
so detached usage fields cannot make a rootless replay conclusive. The compiled
`EvalAssertion` adapter also preserves the existing explicit complete-synthetic
context contract used by direct assertions.

When cost evidence is requested, callers inject a trusted local `PriceBook`.
`pricing_profile_identity(...)` canonicalizes and fingerprints the validated
book together with Cayu's pricing-semantics version. The evidence and corpus
carry only that identity and bounded aggregate costs, never the book or its
provider/model line items. Reordering behavior-equivalent top-level price-book
entries does not change the identity; changing pricing content or Cayu's pricing
semantics does. Every corpus cost assertion must use one of the pricing
profile's advertised currencies, keeping the complete per-trial cost projection
inside the profile's 32-currency evidence bound.

`evaluate_assertion_spec(...)` and `evaluate_assertion_specs(...)` are the pure
portable evaluation boundary: they consume only an `AssertionEvidenceView` and
return the existing four-state `EvalAssertionResult`. `compile_assertion_spec(...)`
adapts one allowlisted spec to the existing `EvalAssertion` runner interface.
Callers inject the `CayuApp` redaction boundary, evidence policy, and optional
trusted local `PriceBook`; the runner projects one immutable evidence view and
shares it across every compiled assertion in the trial. Cost assertions may
share one trusted price-book source; the runner validates each distinct source
once per trial and rejects any change from its compilation-time fingerprint.
During a fresh run, the executing `CayuApp` is the authoritative redaction
boundary; the compiler-supplied
app is used for offline replay. Existing built-in
assertions and compiled specs share the same decision functions, so a missing or
bounded-away observation is `unavailable`, while a complete observed negative is
`failed`. Tool-order assertions use model-requested transcript order;
tool-presence/count assertions use calls that actually started.

## First-party runtime acceptance suite

Cayu ships an importable, hermetic target for exercising runtime-native evals
without relying on the caller's current working directory:

```bash
uv run cayu eval run cayu.evals.internal.runtime_acceptance:build \
  --case-timeout-seconds 30 \
  --output /tmp/cayu-internal-runtime-acceptance.json
```

The target builds the stable `cayu-internal-runtime-acceptance-v1` suite. Its
cases are:

- `tool_roundtrip` — tool arguments, result propagation, and final use;
- `workspace_roundtrip` — isolated write/read calls, results, and file content;
- `context_observability` — deterministic pressure estimates, provider token
  counts, and both reconciliation event families;
- `knowledge_tool_roundtrip` — search and read of one deterministic fact that
  is seeded pending and approved before the suite runs;
- `subagent_roundtrip` — a foreground direct child with an independent model
  script, completed-child evidence, and parent result use;
- `usage_accounting` — explicit positive model usage under a token ceiling;
- `budget_interrupt` — priced caller limits interrupt before a queued external
  side effect can run, with the complete interruption evidence scored directly;

Every independent workflow has its own named `ScriptedModelProvider`, selected
through `AgentSpec.provider_name`. An environment factory retained by the app
creates a separate temporary `LocalWorkspace` for every session, so a file from
one case cannot satisfy another case. The suite uses structural assertions only:
it reads no provider credentials, makes no network calls, consumes no external
sandbox quota, and does not use `LLMJudge`.

This first slice does not cover multi-phase approval resume, live-provider
promotion, browser behavior, `SIGKILL` recovery, provider billing
reconciliation, LLM-judged quality, or baseline release gating. Those require
different dependencies or release policy and should not be inferred from a
passing hermetic report.

Interrupted sessions remain outside the production terminal-evidence contract
used by promotion. A direct Python eval can nevertheless score an interruption
that it created itself: the runner must drain that fresh execution, and an
opted-in store must atomically match every emitted root event's durable
sequence and type. Before payload hydration, built-in SQL stores apply bounded
counts and backend-specific working-set guards; evidence that passes those
guards is then checked against the same exact canonical record and total byte
limits used by every store.
Interrupted descendants are accepted only when their direct parent is proven
inside that fresh execution tree. Any mismatch remains `unavailable`;
arbitrary historical interrupted sessions never enter through this path.

## Built-In Assertion Areas

Current assertions cover:

- session status
- final output text
- transcript text
- event occurrence and absence
- tool call counts
- exact transcript tool-call order
- tool arguments
- tool result text
- model step and token ceilings
- estimated-cost ceilings with a supplied price book
- workspace file existence/content
- artifact creation

`MaxEstimatedCost` fails closed: if even one observed model step has no matching
price, its outcome is `unavailable` and the retained cost summary reports both
priced and unpriced coverage instead of treating the missing price as zero.

## Workspace isolation

Cases in a suite run against the **same** `CayuApp`. Each case is a separate session, but it
shares the app's workspace unless you register the environment with an **environment factory**
(`register_environment_factory(...)`), which provisions a fresh environment — and a fresh
workspace — per session.

Because of this, `WorkspaceFileExists` / `WorkspaceFileContains` assert *"the file is present in
the workspace when the case finished"*, **not** *"this case created it"*. With a single shared
environment, a file written by an earlier case will satisfy a later case's workspace assertion.
To isolate per case, register an environment factory (or clean up the workspace yourself between
cases).

## Eval modes

The same suite/assertion surface supports several modes:

- **Deterministic** — drive the run with a `ScriptedModelProvider` (and fake tools). Hermetic,
  fast, and provider-free — ideal for CI. This is what the Minimal Example and most tests use.
- **Integration** — run against real providers, tools, runners, and environments to check whether
  the agent actually solves the task and to capture real cost/latency/tool usage. Slower and
  optionally gated behind credentials.
- **Replay / regression** — persist a run's `Trajectory`, then re-run the assertions against it
  later to catch regressions; compare a baseline to the current run with `compare_eval_runs` (or
  `cayu eval compare`). See [Trajectories & Replay](#trajectories--replay).
- **Offline** — evaluate a *captured* trajectory (`load_trajectory` → `evaluate_assertions`) with
  no live runtime, on any machine, from a saved JSON file.
- **Production replay** — promote a completed or failed durable session tree with
  `trajectory_from_session(...)`, then score or export the resulting `Trajectory` without running
  the application again. Corpus management and fresh re-execution are planned follow-ups.

## Results and repeated trials

`run_eval_case(..., trials=N)` executes trials sequentially with a fresh concrete
session ID each time. `EvalCaseResult.trials` is an ordered tuple of
`EvalTrialResult` values; every trial retains its own status, session ID, final
output, assertion outcomes, exact-snapshot usage, assertion-specific cost
summary, duration, diagnostic, evidence-completeness flag, and optional
trajectory. Case and run aggregates are reproducible from those retained tuples.
There is no representative or implicit “last trial.”

Assertion outcomes are `passed`, `failed`, `unavailable`, or `error`. Cases and
runs add `skipped` for a direct Python case with no assertions. Aggregate status
precedence is `error` → `unavailable` → `failed` → `skipped` → `passed`.
Unavailable and error results have `score = null`; aggregation never converts
them to zero or drops them from an average. A score is emitted only when every
contributing result is scored.

## Capturing terminal session evidence

The built-in in-memory, SQLite, and PostgreSQL session stores expose
`load_terminal_session_evidence(session_id, limits=...)` as the safe input
boundary used by production-session trajectory promotion. The operation accepts
only a coherent completed or failed session and returns one detached, bounded snapshot:
the session, its durable event prefix through the matching terminal event, its
attributed transcript, publication-marker state, and exact boundary/count/byte
metadata, including the complete canonical returned size. It excludes later
event telemetry and fails with a typed error instead of truncating incomplete,
contradictory, or oversized evidence.

After a fresh completed or failed eval drains `CayuApp.run(...)` completely
(including runtime hooks), the runner uses this operation to build the root
trajectory and derive usage from the same terminal-bounded event prefix. A
runner-owned fresh interruption uses the narrower reconciliation described
above without expanding production-session eligibility. A typed incomplete,
contradictory, unsupported, or oversized snapshot becomes an `unavailable`
trial with no score; an unexpected execution/evaluation failure becomes
`error`. Neither can pass.

This operation does not itself create an eval case or corpus and does not add a
control-plane route. Those product steps build on this storage guarantee. See
[Runtime Contracts](runtime-contracts.md#sessionstore) for the snapshot and
resource-limit contract.

## Promoting a durable session trajectory

`trajectory_from_session(...)` turns one already-finished production session
tree into the same `Trajectory` assertion substrate used by fresh eval runs:

```python
from cayu import (
    SessionCompleted,
    ToolsCalledInOrder,
    evaluate_assertions,
    trajectory_from_session,
    write_trajectory_json,
)

trajectory = await trajectory_from_session(app, session_id)
outcomes = await evaluate_assertions(
    trajectory,
    [SessionCompleted(), ToolsCalledInOrder(["search", "read"])],
)
write_trajectory_json(trajectory, "production-session.json")
```

This is historical evaluation, not re-execution. The operation only reads the
configured session store. It does not call providers, tools, environments,
hooks, recovery code, or other application behavior, and it does not write to
the store. Production probe data is therefore marked unavailable rather than
read from the app's current environment; assertions that need an uncaptured
workspace or artifact report `unavailable`.

The root and every admitted descendant must independently be a coherent
`completed` or `failed` terminal snapshot. A child belongs to the trajectory
when its first durable `session.started` or `session.forked` event is no later
than its direct parent's terminal event. A background child that began before
that boundary remains included even if it finished later; a fork created after
the boundary is excluded. Every retained origin must also carry the
runtime-owned `parent_session_id` matching the session record; a
`session.forked` origin must carry the same `source_session_id`. This validation
also applies when a non-root session is promoted directly. Matching
caller-authored text is not authority: built-in stores discard untrusted origin
linkage at ingestion, and custom terminal-evidence readers must preserve or
reconstruct equivalent runtime provenance. The same admission rule is applied
at every level.

Capture fails closed if an admitted node is active, interrupted, incomplete,
contradictory, oversized, or changes while the tree is being read. One
`SessionTrajectoryBounds` budget applies across the whole retained tree, with
default limits of 100 sessions and 32 tree levels, plus the terminal-evidence
count and byte limits documented under
[SessionStore](runtime-contracts.md#sessionstore). The caller may raise the
session limit to 500 and lower the depth limit when needed; Cayu's hard depth
ceiling is 32. The explicit depth ceiling keeps every accepted tree within the
serialization and replay envelope of the public recursive `Trajectory` model;
historical promotion never truncates at that boundary. Stable
`SessionTrajectoryErrorCode` values let callers classify rejection without
parsing error text. Repeating the read against unchanged durable state produces
an equal trajectory. The retained-session limit does not count children excluded
by the terminal boundary. Lineage discovery has a separate non-configurable
hard ceiling of 500 unique child candidates across the capture, so extreme
fan-out remains bounded before admission.

The built-in memory, SQLite, and PostgreSQL stores provide every required read.
A custom store must implement and advertise exact terminal evidence and bounded
session lineage. The lineage projection contains only pre-hydration-bounded
structural identity and payload-free origin-event fingerprints; Cayu does not
fall back to full topology objects or payload-bearing event reads.

`ToolsCalledInOrder([...])` requires an exact sequence: reordered, missing, or
additional calls fail. It reads model-requested `ToolCallPart` values in durable
transcript order rather than scheduler event timing, so parallel tool execution
does not change the result.

Trajectories may contain prompts, model output, tool arguments/results, and
session metadata. Export and retention policy remains the application's
responsibility; Cayu does not automatically publish or retain promoted data.

## LLM Judges

For *subjective* quality — "is this answer helpful / accurate / on-tone?" — a deterministic
check isn't enough. `LLMJudge` is a graded assertion: a model scores the run's output on a
continuous 0..1 scale against a rubric, and that score flows into the case/run score via the
score-first format (`score >= threshold` decides pass/fail).

```python
from cayu import AgentSpec, AnthropicProvider, CayuApp, EvalCase, LLMJudge

# A judging runtime — typically a stronger / different model than the agent under test.
judge_app = CayuApp()
judge_app.register_provider(AnthropicProvider(), default=True)
judge_app.register_agent(AgentSpec(name="judge", model="claude-opus-4-8"))

case = EvalCase(
    id="helpfulness",
    request=...,
    assertions=[
        LLMJudge(
            judge_app,
            agent_name="judge",
            rubric="Score how helpful and accurate the answer is.",
            threshold=0.7,
        ),
    ],
)
```

The judge runs **its own** agent (you configure the provider/model on `judge_app`), so judging is
an explicit, separate dependency rather than reaching into the run under test — and it is
deterministically testable by injecting a scripted provider. Every judgment is **auditable**:
`metadata` records the judge's provider/model, the `rubric` (and optional `rubric_version`), the
exact `prompt`, the raw `judge_output`, and the parsed `score`/`rationale`. Pass
`include_transcript=True` to give the judge the full transcript, not just the final output.

## Trajectories & Replay

A **`Trajectory`** is the serializable *record* of one run — its session, events,
transcript, usage, a captured probe snapshot (the workspace files and artifacts the
case's assertions need), and any sub-agent runs as nested children. It is also the
assertion substrate: assertions evaluate against a `Trajectory` (via the `EvalContext`
their `evaluate()` receives — `EvalContext` is the assertion's *view* of a trajectory plus
the case identity).

Because it is serializable, a run can be saved and re-checked later **without a live
runtime** — the replay loop:

```python
from cayu import (
    run_eval_case, write_trajectory_json, load_trajectory, evaluate_assertions,
)

# 1. Run, asking the runner to retain the probe-complete trajectory it built.
result = await run_eval_case(app, case, suite_id="suite", retain_trajectory=True)
trial = result.trials[0]
assert trial.trajectory is not None  # populated because retain_trajectory=True

# 2. Persist it (opt-in; a plain JSON file you manage — no automatic retention).
write_trajectory_json(trial.trajectory, "run.json")

# 3. Later / elsewhere: reload and re-run the same assertions offline.
restored = load_trajectory("run.json")
results = await evaluate_assertions(restored, case.assertions)
assert all(r.passed for r in results)
```

`retain_trajectory` defaults to `False`, so a normal run does not retain trajectory
payloads after each trial result is built. When enabled, every trial retains its own
trajectory. Trajectories are **excluded from saved `EvalRun` JSON** and remain separate,
opt-in exports.

Saved `EvalRun` baselines use schema version `4`. Version 4 preserves the complete
ordered trial graph and explicit outcome/null-score contract. It retains version 3's
identity-free aggregate usage, canonical large counters, and durable-JSON validation.
`load_eval_run(...)` rejects missing versions and versions 1–3; regenerate those
baselines with the current Cayu version. No compatibility loader or migration is used.

Standalone exports use a versioned document envelope. The current trajectory
schema version is `1`; `load_trajectory(...)` rejects files without that version
or with an unsupported version before validating the trajectory payload. This is an
intentional clean break from Cayu's earlier unversioned preview exports: they
are not migrated and must be regenerated. The trajectory schema version is
independent from `EvalRun.schema_version`. Version 1 applies Cayu's durable-JSON
contract before writing and while decoding: nonportable text or numbers,
duplicate object keys, and excessive nesting are rejected before an existing
export can be overwritten or an imported trajectory can be replayed.

Replay is faithful for the assertions the run captured: event / transcript / usage / output /
tool assertions always re-check correctly, and a workspace or artifact assertion replays as long
as it was part of the original run (its probe was captured then). Replaying with a *new*
workspace/artifact assertion whose path or scope the original run did **not** probe reports
`unavailable` with no score rather than inventing a negative observation. Probe
capture retains successful and unavailable workspace paths and artifact scopes
separately. A confirmed missing file or an empty successfully listed artifact
scope is negative evidence and can fail; a backend error cannot. Workspace file
content is captured through a byte ceiling. Finding the expected text in that
window passes, while not finding it in a truncated file is `unavailable` because
the unseen suffix could still contain the text. A truncated artifact listing is
also `unavailable`; it is never treated as a complete empty scope.

## Interop

The default result format is JSON. It is intentionally simple so downstream
systems can consume it in CI or adapt it to external eval platforms.

Cayu should own the runtime-native view. External tools can own broader
experiment management, hosted dashboards, human review queues, and organization
level workflows.
