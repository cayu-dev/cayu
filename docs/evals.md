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
- `EvalPlan(corpus_target=CorpusTarget(...))`
- `(app, suite)`
- an object or dict with either `app` and `suite`, or `corpus_target`

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

Portable corpora use the same target discovery but must select corpus mode:

```bash
cayu eval validate evals.json
cayu eval inspect evals.json --json
cayu eval merge combined.json team-a.json team-b.json
cayu eval run --corpus combined.json --suite refund-regressions \
  --output refund-results.json --html-output refund-results.html
```

`--suite` is optional only when the corpus contains exactly one suite. Corpus
trial counts and timeouts come from the content-addressed corpus contract and
cannot be replaced by command-line flags. `merge` includes an existing
destination as its first input, deduplicates equal content, rejects a same-ID
revision conflict by default, validates the complete result, and atomically
replaces the destination. `--replace-conflicts` deliberately selects the last
conflicting definition in command-line order.

`eval run` and `eval compare` have stable CI exits: `0` means the selected suite
passed, or a compatible comparison found no regression; `1` means execution
completed and produced a failed result or a compatible regression; `2` means
the command could not make a conclusive pass/fail decision. Exit `2` covers
invalid corpus or result input, unavailable/error/skipped evidence, incomparable
contracts, target/configuration errors, and execution errors. Direct-suite runs
use the same `0`/`1`/`2` status mapping. `eval report` and `eval compare`
auto-detect direct `EvalRun`, captured `CapturedEvaluationResultV1`, and fresh
`CorpusExecutionResult` documents. Captured and fresh published results can be
compared with each other through the same compatibility projection and stable
exit contract. These commands operate only on the supplied paths and do not
perform project discovery.

## Portable corpus documents

`EvalCorpusDocument` is Cayu's bounded, JSON-portable definition format for
reusable eval suites and cases. A document describes exactly one trusted
`target_key`. A runnable case contains only user-role text input; a captured-only
case uses `input: null`. Documents also contain bounded trial settings,
diagnostic source/pricing identities, an explicit evidence policy, and a closed
set of structural assertion specifications. It cannot contain a `CayuApp`,
provider/model/environment selection, import path, callback, raw session ID, or
runtime event payload.

The portable assertion kinds in schema version 1 cover root and child terminal
status, final-output equality/containment, tool presence/order/count, model-step
and token limits, recorded usage, estimated-cost limits, and trusted model
judgments. Cost assertions require a `PricingProfileIdentityV1`; the identity
fingerprints trusted pricing used elsewhere and never embeds or authorizes a
`PriceBook`. A `ModelJudgeAssertionSpec` remains data-only: it carries a bounded
rubric and rubric version, threshold, transcript-selection flag, trusted
evaluator key. The target resolves that key to its local trusted judge
implementation at execution time. Each published result records the resolved
implementation revision, so the same portable corpus remains reusable across
trusted evaluator rollouts while comparisons still reject different judges.

Corpus documents are definitions, not executable application configuration.
Parsing one never imports project code or invokes a provider, tool, environment,
hook, or runtime. A trusted caller resolves `target_key` to local application
bootstrap code. Captured source identity remains diagnostic provenance; fresh
execution records its own target release and AppManifest and does not pretend
that the source application is reproducible or unchanged.

Use the `.create(...)` factories for `EvalSuiteSpec`, `EvalCaseSpec`, and
`EvalCorpusDocument`. They validate and canonicalize their inputs and compute
`sha256:` content revisions. Suite revisions cover reusable suite settings;
cases reference suites by `suite_id`, so cases from independent corpus fragments
can be merged without rewriting suite membership. Case, suite, evidence-policy,
and corpus revisions change whenever their covered content changes;
`assertion_spec_revision(...)` provides the same identity for one assertion.
Every suite must contain at least one case. A fresh-execution suite cannot mix
runnable and captured-only cases; captured-only suites remain valid definition
and publication shapes but are rejected by fresh-run admission until runnable
input is authored.

`eval_corpus_to_json(...)`, `eval_corpus_from_json(...)`, and
`load_eval_corpus(...)` enforce schema version 1 and Cayu's durable-JSON rules,
including duplicate-key, non-finite-number, integer-range, Unicode, and nesting
validation. Input is rejected before an unbounded read or decode. The hard
document limit is 8 MiB, with at most 64 suites, 1,000 cases, 64 assertions per
case, 0 to 16 messages per case, 65,536 characters per message, 262,144 input
characters per case, 100 sequential trials, and a 3,600-second per-trial timeout.
Each suite may expand to at most 10,000 published assertion results across its
cases and trials, matching the boundary of the one-suite execution result. A
multi-suite corpus may exceed that aggregate because suites execute and publish
independently; inspection reports the complete corpus-wide count without
materializing result graphs.
Unknown fields and assertion kinds fail closed; schema version 1 has no legacy
compatibility loader.

### Trusted corpus execution

A corpus deliberately contains no executable application authority. Local SDK
and CLI execution resolves that authority from the project-owned eval target:

```python
from cayu import (
    AgentSpec,
    CayuApp,
    CorpusTarget,
    EvalPlan,
    EvaluationEvidencePolicySpec,
    Message,
    ModelJudgeAssertionSpec,
    ModelJudgeTarget,
    RunRequest,
)


def build_eval() -> EvalPlan:
    app = build_app()
    judge_app = CayuApp()
    judge_app.register_provider(build_judge_provider(), default=True)
    judge_app.register_agent(AgentSpec(name="quality-judge", model="judge-model"))
    judge = ModelJudgeTarget(
        key="quality-judge",
        app=judge_app,
        agent_name="quality-judge",
    )
    quality_assertion = ModelJudgeAssertionSpec(
        id="answer-quality",
        evaluator_key=judge.key,
        rubric="Score correctness and usefulness.",
        rubric_version="quality-v1",
        threshold=0.8,
        include_transcript=False,
    )
    # Include quality_assertion in the authority-free EvalCaseSpec.
    return EvalPlan(
        corpus_target=CorpusTarget(
            key="refund-agent",
            app=app,
            request_base=RunRequest(
                agent_name="refund-agent",
                messages=[],
                max_steps=12,
            ),
            bootstrap_messages=(
                Message.text("system", "Follow the production refund policy."),
            ),
            application_release_id="refund-service-2026-08-06",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            # price_book=trusted_price_book,  # required when the corpus uses cost assertions
            model_judges=(judge,),
        )
    )
```

`CorpusTarget` is immutable and defensively copies its request, bootstrap,
evidence-policy, limit, optional pricing, and model-judge target inputs. Its request base must have
no messages, session/parent/causal/task identity, structured-output request,
prior redaction state, or runtime authority. Bootstrap messages are bounded
single-text-part system or user messages. Compilation produces exactly that
trusted bootstrap followed by the corpus's user messages; every trial then gets
a fresh runtime session and causal identity from the existing evaluator. The
trusted request base is capped at 64 KiB and the published AppManifest at 1 MiB.
Input is bounded both per case and across the fully compiled suite, including the
trusted bootstrap repeated for each case, so compilation cannot multiply one
large bootstrap into an unbounded working set. The compiled-suite ceiling is
8,388,608 input characters. A target may lower case, trial, timeout,
concurrency, bootstrap, and input ceilings but cannot raise Cayu's hard limits.
The target key and application release ID are public result identity: both must
cross the target app's workload-secret redaction boundary unchanged, or target
validation fails before provider dispatch.

Before provider dispatch, `compile_corpus_suite(...)` revalidates the complete
corpus and matches its target key, evidence-policy revision, applicable trusted
PriceBook identity, selected suite, and trial/input/case/timeout ceilings. It
compiles the allowlisted assertions through the same portable assertion adapter
used by `compile_assertion_spec(...)`; there is no second evaluator. All cost
assertions in the selected suite share one
compile-time pricing binding, so the trusted PriceBook is validated and
fingerprinted once for the suite rather than once per assertion.

Portable model judges deepen that same compiler rather than introducing a
scorer or plugin registry. `ModelJudgeTarget` binds one portable evaluator key
to a locally constructed `CayuApp` and registered agent. The agent must exist
and be tool-free. Missing keys, invalid registrations, missing or ambiguous
provider resolution, and tool-bearing
registrations reject during compilation before the candidate provider is
called. The resolved implementation revision covers Cayu's model-judge execution
semantics, evaluator key, agent name, the judge app's complete public
`AppManifest`, and the exact secret-redacted agent specification, including its
system prompt, provider options, and thinking configuration. A trusted judge
rollout changes that resolved revision, not the portable corpus contract: the
fresh revision is published with the result and cross-revision comparisons are
incomparable. Deterministic specs do not resolve or invoke this authority.
`evaluate_assertion_spec(...)` and standalone `compile_assertion_spec(...)`
therefore reject a model-judge spec: executable resolution is available only
through an explicit trusted `CorpusTarget`.
`run_corpus_suite(...)` and corpus-mode
`run_eval_plan(...)` execute that compiled suite through `run_eval_suite(...)`,
bind the pre-dispatch corpus contract, and publish through
`publish_eval_run(...)`. A changed AppManifest during execution rejects the
result instead of publishing it under stale target diagnostics.

The returned `CorpusExecutionResult` contains the safe `PublishedEvalRun` plus
the fresh application release and complete bounded public `AppManifest`. The
manifest is diagnostic, not a reproducibility claim, and its fingerprint is
recomputed during validation. `corpus_execution_result_to_json(...)` and
`load_corpus_execution_result(...)` provide bounded deterministic JSON;
`render_corpus_execution_html(...)` renders only the published graph. Each trial
may include a Cayu-produced preview of the same app-redacted output evidence used
by its assertions: at most 16 KiB per trial and 2 MiB across a published run,
with explicit availability/truncation state, retained size, and retained-content
digest. The raw final output and omitted preview suffix are discarded before
publication. Neither format contains trajectories, session IDs, provider
payloads, exception text, credentials, or executable target objects.

`corpus_execution_compatibility(...)` is the typed precondition for later
regression comparison. It requires the same target key, corpus/suite/case/
assertion contract, evidence policy, and applicable pricing identity, while
deliberately allowing a different fresh application release and AppManifest.
It returns stable incomparable reason codes; it does not silently compare
different evaluation contracts. Model-judge rubric text/version, threshold,
transcript selection, and evaluator key contribute to the assertion and corpus
revisions. The resolved implementation revision contributes only to the
published assertion binding, so a judge rollout leaves a portable corpus valid
while making cross-revision results incomparable rather than manufacturing a
score delta.

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
context contract used by direct assertions. Exact aggregate token counts use
Cayu's canonical nonnegative decimal-string JSON representation; counts above the
IEEE-754 safe-integer assertion ceiling remain lossless but are marked
`limit_exceeded` and cannot produce a scored usage assertion. This keeps every
numeric corpus field exact across Python, browser, and other portable JSON
boundaries. The serialized evidence view is
capped at 10 MiB, enough to retain every field at its declared character and
cardinality ceiling, including four-byte Unicode.

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
boundary; the compiler-supplied app is used for offline replay. Existing built-in
assertions and compiled specs share the same decision functions, so a missing or
bounded-away observation is `unavailable`, while a complete observed negative is
`failed`. Tool-order assertions use model-requested transcript order;
tool-presence/count assertions use calls that actually started.

A portable model judge is intentionally online rather than part of that pure
evidence adapter. It receives the candidate task and complete bounded final
output after the candidate app's secret-redaction boundary; setting
`include_transcript=True` additionally sends the redacted transcript when its
rendered text fits the portable 262,144-character bound. Treat those values as
provider-bound data and apply the same privacy review used for any external
model call. Missing, truncated, or over-limit graded evidence is `unavailable`
and does not invoke the judge. Candidate text is delimited as untrusted data and
embedded closing delimiters are neutralized; the trusted judge agent receives
no tools. The internal `LLMJudge` audit record may contain the exact prompt and
raw judge output, but portable publication projects only the frozen rubric
contract, score/outcome, and a fixed safe diagnostic (`judgment_recorded`,
`evaluator_error`, or `evidence_unavailable`). It never publishes raw judge
output, rationale, provider/model identity, prompt, exception text, app, or
credentials.

`publish_eval_run(...)` is the only public result projection for a portable
corpus run. It matches the complete internal suite result back to the corpus and
produces a content-addressed schema-version-2 `PublishedEvalRun` containing
every case, trial, assertion outcome, safe structural detail, duration, and
identity-free aggregate usage. Every complete trial carries its exact aggregate
usage, and conclusive
usage-derived observations cannot exist without that summary. A publishable run
carries large aggregate counters in the same canonical decimal-string JSON
representation as Cayu's other aggregate usage surfaces. Cost observations are
published only when their allowlisted metadata exactly matches any retained
`SessionCostSummary`, including currency, total, and the priced/unpriced step
partition. The run carries the exact corpus, suite, case, evidence-policy,
applicable pricing-profile, trial-count, and timeout contract fixed before provider dispatch;
publication rejects an absent or different execution contract. Every internal
assertion result must also carry the exact corpus assertion revision it evaluated,
preventing publication under a different expectation. Final-output decisions are
carried as the shared evaluator's safe boolean observation. Trusted corpus
execution can additionally carry a bounded preview of that same redacted evidence;
direct publication without the runner-owned projection marks output unavailable
instead of copying `EvalTrialResult.final_output`. Tool-order
decisions are checked against the complete bounded order retained by evaluation;
only boolean matches and safe counts cross the publishing boundary.
Model-judge results retain the bounded rubric and rubric version, threshold,
transcript-selection flag, evaluator key, implementation revision, continuous
score/outcome, and fixed safe diagnostic. A valid finite score is candidate
evidence: the threshold decides `passed` versus `failed`. Missing authority,
judge configuration drift, provider/runtime failure, an attempted tool call,
an incomplete session, empty output, or an invalid score is `error` with no
numeric score; judge failure is never converted into a candidate-quality
failure.
Trial, case, and run scores and statuses are rederived from the retained
published children. Public diagnostics use fixed Cayu-owned reason codes and
messages that distinguish assertion, lifecycle, evidence, and timeout failures
without copying raw exception text. Raw assertion metadata, raw final output,
trajectories, concrete session IDs, and provider/model identity are never copied.
Cost results require
the corpus pricing-profile fingerprint. The closed schema-v2 graph is bounded to
32 MiB and is the reporting, comparison, and CI substrate for portable corpus
execution; the lossless `EvalRun` does not cross that publishing boundary.
The publication model defines and enforces this boundary independently of execution.
A trusted corpus executor must construct the contract from its resolved local target;
the unconstrained Python runner cannot accept a caller-supplied publication contract.

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
  the application again. The Control Plane can persist this captured result and, for a simple
  safely reconstructable invocation, launch a bounded fresh trial; multi-stage scenario replay is
  a separate versioned contract.

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

For payload-free operational/accounting evidence rather than a transcript-based
eval trajectory, use `runtime_evidence(app, request)`. That public v4 projection
accepts a bounded nonterminal or terminal lineage, preserves attempts, retries,
operation-specific usage, governing execution-profile fingerprints, tools,
approvals, taint, recovery, receipts, and optional causal-budget totals, and
excludes prompts, outputs, and tool payloads.
It does not replace the coherent terminal snapshot required for trajectory
promotion. See [Bounded runtime-evidence projection](runtime-contracts.md#bounded-runtime-evidence-projection).

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

Every promoted session also receives the same typed `MemoryAttribution` used by
`runtime_evidence(...)`. Promotion queries only the dedicated bounded receipt/exposure
surface and shares one `SessionTrajectoryBounds.memory_attribution_bounds` budget across the
whole tree. Projection-level `unavailable`, `redacted`, `truncated`, and `contradictory`
states remain distinct from exposure-level `indeterminate`; none is rewritten as complete
empty memory use. This promotion is read-only and does not call providers, tools,
environments, applications, or recovery.

The built-in memory, SQLite, and PostgreSQL stores provide every required read.
A custom store must implement and advertise exact terminal evidence and bounded
session lineage. The lineage projection contains only pre-hydration-bounded
structural identity and payload-free origin-event fingerprints; Cayu does not
fall back to full topology objects or payload-bearing event reads.

`promotable_run_input(app, trajectory, source_agent_name=...)` is the narrower
automatic portable-corpus boundary. It returns a bounded, redacted
`PromotableRunInputV1` only when the configured source agent matches and the
captured tree is complete and quiescent. The root and all descendants may be
completed or failed; a failed run remains useful as a regression to fix.

Schema v1 accepts exactly one fresh invocation containing one or more
caller-supplied user messages with exactly one text part per message. It rejects
resumes, approval continuations, queued or later input, structured output,
caller-supplied system/assistant/tool messages, and file or other structured parts
with stable `SessionPromotionErrorCode` values. These rules affect automatic
portable promotion only: normal Cayu sessions and direct Python evals retain all
of those capabilities, while runtime tool calls, artifacts, and admitted child
agents remain eligible. Tool calls and child status remain available as portable
assertion evidence; corpus v1 does not silently infer replay input or an artifact
assertion from uncaptured state. Caller-driven approval, resume, queued-input, and
later-interaction phases are checked recursively across every admitted descendant,
not only on the root.

The initial-input boundary is a single versioned runtime-attested fact emitted by
new runs and preserved by the built-in stores. Caller-authored copies are stripped
before persistence. SQL revision 31 also stores an explicit runtime-ownership proof
bit. Revision 31 is a breaking schema boundary because pre-31 readers would expose
the unknown marker as ordinary event payload; mixed-version operation is rejected.
Rows written before migration retain a false proof bit, so payload presence alone
never grants authority. The marker binds the exact transcript start index, message
count, and canonical SHA-256 digest of the redacted request messages. Promotion
verifies the complete marker facts and their selected messages against a recursively
revalidated trajectory before accepting input. A domain-separated capture
fingerprint commits the finalized public trajectory together with the marker's
private start, count, message digest, redaction mode, and output mode, so neither side
can be changed independently. The marker and its private interpretation are
intentionally absent from serialized trajectories, so a detached or older trajectory
cannot guess which
transcript messages were caller input; it fails closed as
`input_evidence_unavailable` instead. Multiple text parts reject because their
provider-specific boundaries cannot be represented exactly by corpus v1's single
text field. The resulting sanitized input and redaction fact carry one exact content
revision.

`build_promotion_candidate(...)` applies that eligibility boundary and produces
one immutable `PromotionCandidateV1`: the sanitized input, standard public
assertion evidence, evidence policy, optional pricing identity, diagnostic
`PromotionSourceV1`, one suite, one case, and stable warning codes. Source
provenance records the application release ID and diagnostic app-manifest
schema/fingerprint together with the selected source agent, sanitized-input
revision, and redaction fact; it is useful for review but is not executable
authority or a claim that the application can be rebuilt from the corpus alone.
Warning codes are derived from those captured facts and the evidence status;
callers cannot remove or invent them.

Every new candidate starts with one `root_status` assertion expecting
`completed`, even when the captured source failed. The failure is therefore a
regression to fix rather than a golden failed result. The case and suite are
ordinary corpus specs: an author may recreate them with their `.create(...)`
factories to edit names, trial settings, input, or assertions, then recreate the
candidate to obtain exact new content revisions. The default case ID is derived
from the target key, captured sanitized-input revision, diagnostic source
identity, and public-safe evidence revision. It stays fixed across candidate
edits and contains no raw session ID.
Candidate configuration labels are rejected if the application redactor detects
a workload secret. This boundary includes the target-derived suite identity and
every persisted pricing-profile identity string.

`score_promotion_candidate(...)` evaluates the reviewed case against that same
captured evidence through `evaluate_assertion_specs(...)`; it does not introduce
a second scorer. Before scoring, it rechecks the target, release, app manifest,
selected source agent, promotion eligibility, exact sanitized-input revision and
redaction fact, evidence policy, public evidence revision, and redaction-safe
configuration identity. A changed snapshot, source-agent mapping, sanitized
source input, or relevant redaction result requires a new candidate. Only
currencies requested by edited cost assertions are reprojected, and supplied
pricing must match the candidate's exact `PricingProfileIdentityV1`. Missing or
partially unpriced cost evidence produces an `unavailable` assertion and a null
score—it cannot pass. The returned
`CapturedRunScoreV1` contains bounded published assertion details and content
revisions, never raw assertion metadata, exception text, or session identity.

After review, `corpus_from_promotion_candidate(...)` constructs the exact
one-suite, one-case `EvalCorpusDocument`; `export_promotion_corpus(...)` returns
its canonical UTF-8 JSON bytes. Export revalidates every candidate, source,
policy, pricing, suite, case, assertion, and content revision through the corpus
factories. Cost assertions without a compatible pricing-profile identity reject.
Preview evidence, warnings, app internals, runtime configuration, and session
identity are not corpus fields. The same valid candidate produces byte-identical
output across processes.

### Click-to-evaluate captured sessions

On an authenticated server with a generated or explicitly registered eval
target, every coherently retained completed or failed session exposes
**Evaluate**. No per-session Python configuration is required. The server maps
the session's root agent to one unambiguous target, reconstructs bounded terminal
evidence, and returns a side-effect-free preview. Preview never calls a provider,
tool, environment, hook, or application workload and never writes a corpus or
result.

The review sheet shows retained status, output, tool, step, usage, and cost
evidence. Assertion quick-adds begin from those observed facts; operators can
edit suite/case identity and any portable non-judge assertion before rescoring.
The initial root assertion always expects `completed`, so a captured failure is
saved as a regression to fix rather than silently approved as correct behavior.
Every edit makes the displayed score stale and disables save/export until the
server has reconstructed the current session and scored the exact edited
candidate again. A changed session, release, manifest, evidence policy, pricing
profile, target mapping, or redaction result returns a conflict and requires a
new preview.

**Save evaluation** atomically persists two immutable documents: a one-case
expectation corpus and its captured score result. That corpus deliberately uses
`input: null`; Cayu does not invent a prompt, flatten a resumed conversation, or
pretend that historical execution authority is replayable. The saved result is
valid and useful for review, baselining, release comparison, and future scenario
authoring even when runnable corpus-v1 conversion is unavailable. **Export eval
JSON** returns the same deterministic captured-only corpus without writing.

Runnable corpus-v1 conversion and scenario-v2 capture are independent
capabilities. Simple fresh invocations may satisfy
`build_promotion_candidate(...)`. Multi-stage sessions can instead produce an
ordered scenario preview from their retained initial, queued, resumed,
approval-checkpoint, and file-backed stimuli. Neither conversion controls
captured scoring or persistence: when exact source material is missing, the
captured evaluation remains usable and only the affected conversion reports why.

When that conversion is available, the same review sheet exposes **Run fresh
trial**. The default is one trial at concurrency one. Operators can contract the
published target's trial timeout and model-step ceiling, add run-scoped token,
tool-call, or elapsed-time limits, and—when the server owns a compatible
`PriceBook`—set an estimated-cost ceiling. The browser never supplies tools,
environments, credentials, pricing schedules, approval rules, or other execution
authority. It submits the reviewed expectation contract and bounded settings;
the server reconstructs runnable input from its current target baseline, scores
and saves the captured result, then admits the fresh run through the ordinary
durable worker.

Runtime and estimated-cost ceilings apply independently to each fresh trial;
they are not aggregate ceilings across the eval run. Trial count and concurrency
bound the run's aggregate scale. Generated project targets permit only one trial,
while an application-owned target that permits multiple trials must account for
that multiplication when choosing per-trial limits.

The fresh session uses the target's normal provider, tools, environment,
approval, and operator policy. Authenticated HTTP provenance and every requested
contraction are persisted with run admission, so a worker restart cannot silently
turn an operator launch into unattributed SDK work or recover with broader
limits. Before writing, the server resolves the target's normal provider/model
route and accepts a cost ceiling only when the current server-owned price book
has compatible pricing in the selected published currency. It then preflights
the complete effective request and enforces target ceilings again during
compilation and execution. Tool-effect metadata neither grants nor denies
authority.

Generated project targets deliberately permit only one trial at concurrency one.
Increasing repetition or parallelism, substituting fixtures, bypassing normal
approvals, or selecting different tool/environment authority requires an
explicit application-owned target profile. A session that cannot be
reconstructed is not partially replayed; captured scoring remains usable while
conversion returns a factual diagnostic.

Saved results appear in the target-scoped **Evals → Results** catalog alongside
fresh results. Selecting a result exposes its immutable public-safe score and
corpus identity. **Approve baseline** performs an actor-attributed, idempotent
compare-and-swap update; the request cannot supply or spoof its actor. Concurrent
baseline changes fail instead of silently overwriting another operator's choice.

### Portable multi-stage scenarios

Corpus v1 remains the supported assertion and execution contract. Scenario v2
adds a separate, authority-free description of external stimuli for sessions
that need more than one initial user message. A scenario can contain:

- exactly one initial input, followed by ordered queued or resumed inputs;
- checkpoints that require a new approval decision when the named tool call is
  reached;
- text, portable JSON, and file parts whose content is resolved through a
  declared fixture digest or stable artifact reference; and
- named provider, tool, environment, artifact, or other secret requirements
  without their values or handles.

```python
from cayu import (
    EvalScenarioDocumentV2,
    ScenarioInitialInputEventV2,
    ScenarioInputV2,
    ScenarioTextPartV2,
    ScenarioUserMessageV2,
    compile_eval_scenario,
)

message = ScenarioUserMessageV2.create((ScenarioTextPartV2(text="Start checkout"),))
scenario = EvalScenarioDocumentV2.create(
    id="checkout",
    target_key="support-agent",
    name="Checkout",
    events=(
        ScenarioInitialInputEventV2(
            sequence=0,
            id="initial",
            input=ScenarioInputV2.create((message,)),
        ),
    ),
)
compiled = compile_eval_scenario(scenario)
```

Compilation validates and indexes the portable template; it does not resolve a
provider, tool, environment, artifact, secret, actor, or approval. Those are
trusted launch-time bindings. An approval checkpoint deliberately carries no
approve/deny choice or reusable authorization. `scenario_from_corpus_case(...)`
provides the explicit corpus-v1 bridge for runnable cases, while captured-only
cases fail until input is authored.

Scenario JSON is strict, deterministic, content-revisioned, and capped at 8
MiB. Event, message, part, text, JSON-part, artifact, secret, and aggregate
artifact-byte limits are validated before persistence. Built-in stores expose
`save_scenario`, `load_scenario`, and `list_scenarios`; every save crosses the
configured credential-redaction boundary before it writes. SQLite and
PostgreSQL scenario persistence requires additive storage revision 53.

`capture_eval_scenario_from_session(...)` reconstructs one bounded terminal
session without executing application work. Initial, ordinary resume, and
delivered queue boundaries are accepted only when runtime-owned transcript
attestations bind their exact message positions and digests. Approval history is
projected as a fresh-decision checkpoint, never as reusable authorization. File
parts are read from the source environment's current artifact store, checked
against retained attachment metadata and scope, and matched to a private
runtime-owned digest of the exact bytes resolved for the source model request.
They are represented only by a content digest plus artifact reference; file
bytes do not enter the scenario.
Caller-supplied file-attachment metadata has no scenario-v2 representation and
therefore returns an unsupported-part diagnostic instead of being silently
dropped.
SQLite and PostgreSQL session stores require breaking storage revision 54 before
writing these private attestations. The revision adds an independent
file-attachment proof column but performs no historical backfill; older sessions
that contain files but lack source-time proof fail conversion closed, and older
readers cannot share the migrated store.

Capture is fail-closed and diagnostic rather than all-or-nothing for the
surrounding evaluation workflow. Redacted input, historical evidence that
predates the required attestation, missing or inaccessible artifacts,
contradictory boundaries, and bounded-size failures return stable messages and
remediation while the captured score remains available. The operation reloads
terminal evidence after artifact reads so a changing source cannot be published
as one coherent scenario. The authenticated Control Plane preview exposes this
result and never invokes providers, tools, environments, hooks, recovery, or
mutation paths.

The same Evaluate sheet now opens an ordered scenario editor. Operators can
rename the scenario, edit text and portable JSON, add/remove/reorder queued,
resumed, and approval-checkpoint events, edit named secret requirements, select
an environment, and set ordinary trial, concurrency, timeout, and per-run
bounds. Saving writes the exact immutable revision shown by preview; a stale
revision fails before any store or artifact mutation. Saved scenarios appear in
the target-scoped **Evals → Catalog** view and can be reopened, edited into a new
revision, and downloaded.

**Check readiness** performs launch preflight against current server authority.
It does not invoke a provider, run a tool, materialize an environment factory,
or start application work. A successful result freezes the current release,
AppManifest fingerprint, target, agent, environment, fresh-approval behavior,
the unchanged target limits, any separate operator-selected run contraction,
cost budget, named vault requirements, and verified
environment-scoped artifact identities in a content-revisioned binding. Secret
preflight checks only the selected environment's published logical-name mapping;
it neither resolves nor serializes a vault handle or secret value. An
unsuccessful result returns stable, public-safe diagnostics tied to the exact
event or requirement when applicable. Target/provider/environment mismatches,
broadened limits, missing pricing, unavailable tools, policies that do not prove
a fresh approval pause, unpublished secret bindings, and unsafe public content
all fail closed.

Target and operator limits remain separate authorities in that binding. In
particular, adding a per-run contraction never converts a session-cumulative
target ceiling into a fresh allowance for each scenario stage or resume. File
attachment count and byte ceilings are checked independently for every initial,
queued, and resumed input. Each file part is one runtime attachment occurrence,
so repeated references to the same immutable requirement count repeatedly.

A session-scoped artifact cannot be attached to the fresh session. For a
retained source artifact in the selected static environment, **Prepare fixture**
verifies the exact size, media type, filename, and SHA-256 digest, then performs
an explicit idempotent copy into that environment's artifact store under an
environment-scoped identity. It returns a new unsaved scenario revision and
reruns preflight. The operation never searches another environment's store,
and factory-backed environments remain gated until a concrete server-published
binding can be proven without allocating work.

The protected API exposes the same bounded workflow at:

- `POST /api/evals/scenarios/preview` for canonical draft compilation and
  side-effect-free current-authority preflight;
- `POST /api/evals/scenarios` plus target-scoped `GET` catalog/detail/download
  routes for immutable persistence; and
- `POST /api/evals/scenarios/artifacts/{requirement_id}/materialize` for the
  explicit idempotent fixture operation.

Scenario execution and fresh approval/resume recovery remain the next layer.
They consume the reviewed scenario and preflight binding rather than placing
runtime authority into the portable document.

### Durable eval catalog and run state

Use an `EvalStore` when promoted corpora, queued work, and published results
must survive beyond the promotion request. `SQLiteEvalStore` is restart-durable
for one embedded database; `PostgresEvalStore` supports shared multi-worker
claims; `InMemoryEvalStore` is intentionally process-local and is suitable for
tests and transient SDK workflows only. SQLite and PostgreSQL require storage
schema revision 50 for corpora and run state, and revision 53 for scenario
persistence. Session-backed production capture additionally requires revision
54. Corpus and scenario saves, run admission, and result publication require
the active application's complete JSON redaction boundary. A configured
workload secret or redaction failure rejects before any write; the store never
retains the redaction function or secret registry.

```python
from cayu import EvalRunFailureCode, EvalRunRequest, EvalRunStatus, SQLiteEvalStore
from cayu.evals.execution import run_corpus_suite
from cayu.storage.migrations import SchemaMode

store = SQLiteEvalStore("cayu.db", schema_mode=SchemaMode.MIGRATE)
await store.save_corpus(
    corpus,
    redact_json=target.app.redact_json,
)

request = EvalRunRequest(
    run_id="refund-regression-2026-08-07",
    idempotency_key="sha256:" + request_digest,
    corpus_revision=corpus.revision,
    target_key=corpus.target_key,
    suite_id=corpus.suites[0].id,
    suite_revision=corpus.suites[0].revision,
    max_concurrency=4,
)
await store.admit_run(
    request,
    redact_json=target.app.redact_json,
)

lease = await store.claim_run()
if lease is not None:
    if lease.run.status is EvalRunStatus.CANCELLING:
        await store.finish_cancel(lease.claim)
    else:
        try:
            result = await run_corpus_suite(
                target,
                corpus,
                lease.run.spec.suite_id,
                max_concurrency=lease.run.spec.max_concurrency,
            )
        except Exception:
            # Persist a closed diagnostic code; do not persist exception text.
            await store.fail_run(lease.claim, EvalRunFailureCode.EXECUTION_FAILED)
            raise
        else:
            await store.publish_result(
                lease.claim,
                result,
                redact_json=target.app.redact_json,
            )
```

Corpus revisions and results are immutable. Admission is idempotent, claims use
expiring fenced epochs, cancellation intent is durable, and publication is
atomic with terminal run state. A retry with a stale lease cannot heartbeat,
publish, fail, cancel, or release another worker's run. Ordinary run reads never
contain the private claim token or admission idempotency digest. Every run
record retains a bounded `attempt_count`: zero before first claim and otherwise
the latest fenced ownership epoch, including after terminalization. This makes
recovery/retry attempts attributable without exposing the private claim token.
Workers must heartbeat active claims before their lease expires, stop work when
cancellation is requested, and release still-owned work during a controlled
shutdown. Those execution-loop policies are deliberately outside the store;
the server-attached coordinator described below provides the built-in durable
implementation.

Catalog and run lists use opaque keyset cursors plus caller-controlled item and
byte ceilings. Full corpus and result reads check their exact stored UTF-8 size
before hydrating the document. The durable result is the bounded,
credential-redacted `CorpusExecutionResult`; stores never retain trajectories,
provider request payloads, credentials, executable targets, or arbitrary
exception text. Claim APIs do not accept or persist worker labels; private,
random claim tokens and fenced epochs provide ownership. `EvalStore` owns
persistence and coordination only—the caller still supplies the trusted target
and performs execution.

Captured-session scores and fresh executions use different immutable source
documents but one comparison contract. `CapturedEvaluationResultV1` binds a
`CapturedRunScoreV1` to the exact stored corpus, suite, case, assertion,
evidence-policy, pricing, application-release, and AppManifest identities that
produced it. It deliberately contains no session identifier, executable app,
store handle, credential, or historical runtime authority.
Each captured result uses a single-case suite containing exactly the scored
session; a partial score cannot be published as the result of a larger suite.

`save_captured_result(...)` validates and redaction-scans the complete corpus
and result before atomically saving both. A process loss cannot leave a visible
captured result without its immutable corpus. `load_result_by_revision(...)`
and `load_result_record(...)` address captured and fresh results through the
same content revision; the metadata record exposes the origin explicitly.
Fresh publication writes its origin-aware record in the same transaction as
the existing terminal run result.

```python
from cayu import (
    CapturedEvaluationResultV1,
    EvalBaselineKey,
    EvalBaselineUpdate,
    EvalResultTargetIdentityV1,
)

captured = CapturedEvaluationResultV1.create(
    corpus=corpus,
    target=EvalResultTargetIdentityV1(
        target_key=corpus.target_key,
        application_release_id=source.application_release_id,
        app_manifest_schema_version=source.app_manifest_schema_version,
        app_manifest_fingerprint=source.app_manifest_fingerprint,
    ),
    score=captured_score,
)
record = await store.save_captured_result(
    corpus,
    captured,
    redact_json=app.redact_json,
)
```

Baselines are explicit mutable pointers to immutable results. The key is the
exact target, corpus revision, and suite; a result from another scope cannot be
selected. Every update carries an authenticated actor identifier, an expected
generation, and an idempotent operation digest. The store performs an atomic
compare-and-swap and appends an immutable mutation record. Retrying the same
operation returns the original audit fact, while reusing its digest for another
mutation or losing the generation race fails closed.

```python
key = EvalBaselineKey(
    target_key=corpus.target_key,
    corpus_revision=corpus.revision,
    suite_id=corpus.suites[0].id,
)
mutation = await store.set_baseline(
    EvalBaselineUpdate(
        key=key,
        result_revision=record.revision,
        expected_generation=0,
        operation_id="sha256:" + operation_digest,
        actor_id=authenticated_subject,
    ),
    redact_json=app.redact_json,
)
```

`eval_result_projection(...)`, `eval_result_compatibility(...)`, and
`compare_eval_results(...)` provide the common captured/fresh comparison path.
Application releases may differ, but target, corpus, suite, case, assertion,
evidence-policy, and applicable-pricing contracts must match. Cayu never picks
a baseline inside the comparison function. The Control Plane uses the suite's
explicitly approved baseline pointer by default and permits a manual immutable
result-revision override. Existing custom `EvalStore` implementations remain
source-compatible and advertise this optional contract only when
`captured_results` is true.

### Server-attached durable execution

When a project is started with `cayu serve`, Cayu assembles the non-executable
part of this configuration from project-owned declarations:

- `[project].name` becomes the normalized project identity.
- `CAYU_RELEASE_ID` selects the application release identity. When it is
  absent, Cayu uses the bounded public application-manifest fingerprint.
- `CAYU_DATABASE_URL` or `[tool.cayu.session_store]` selects the durable
  `EvalStore` backend. Trusted loopback `cayu serve --dev` may instead create
  the project-local `data/cayu.db` default.

After the application is built, project assembly generates one trusted
normal-authority target for every registered agent. The empty request base names
that agent and otherwise retains ordinary runtime defaults. A target key is
derived from length-delimited UTF-8 project, agent, and profile identities under
the `cayu-generated-eval-target-v1` domain and published as
`eval.<sha256>`. Release identity is deliberately absent from this derivation;
results separately retain the current release and exact application manifest.

The registry is process-local and bounded to 128 targets. It publishes only safe
identity through `GET /api/evals/targets`; the application object and executable
request authority are never serialized. Corpus and run list queries are scoped
to a published target key, with the registry's deterministic default used when
the query omits one. Imports, reads, admission, cancellation, comparison, and
worker claims resolve persisted target identity back through the registry, so
work for an unknown or foreign target is neither exposed nor claimed.

The generated profile id is currently `default`. It preserves normal agent
provider, tool, environment, approval, and policy selection. The profile
dimension is part of stable identity now so later server-published profiles can
represent deliberate fixture, isolation, or authority changes without changing
existing keys. Generated targets default to one trial and concurrency one; their
catalog entries publish the server-enforced trial, concurrency, timeout, and
model-step ceilings plus the target-compatible cost currencies, if any. A
non-empty currency list is the only condition that marks cost budgets available;
the browser cannot invent another currency and admission repeats compatibility
preflight against current pricing. An
explicit `EvalsConfig` remains the complete low-level contract
and takes precedence as one indivisible singleton registry; Cayu never merges its
target with the automatically assembled store. Arbitrary embedded
`create_server(...)` and `mount_cayu(...)` integrations continue to provide
trusted runtime objects explicitly.

Generated maintained-service factories carry an opaque
`ProjectControlPlaneContext` into `create_agent_service(...)`. Existing
factories continue to start unchanged, but `cayu check` reports
`EVALS_SERVICE_FACTORY_CONTEXT_MIGRATION_REQUIRED` until they are updated. Run
`cayu generate service-context --dry-run`, review the edit, then run
`cayu generate service-context`. Customized factories fail closed for manual
review instead of being rewritten heuristically.

An embedded authenticated Cayu server can attach exactly one trusted
`CorpusTarget` to a durable `SQLiteEvalStore` or `PostgresEvalStore`.
`EvalsConfig` is complete programmatic V1 wiring and is off by default:

```python
from cayu import SQLiteEvalStore
from cayu.server import BasicAuth, EvalsConfig, ServerConfig, create_server
from cayu.storage.migrations import SchemaMode

eval_store = SQLiteEvalStore("cayu.db", schema_mode=SchemaMode.MIGRATE)
server = create_server(
    target.app,
    config=ServerConfig.protected(
        BasicAuth(username="operator", password=resolved_password),
        evals=EvalsConfig(
            target=target,
            store=eval_store,
        ),
    ),
)
```

The target must reference the exact `CayuApp` attached to the server. Open
access, a disabled API, an in-memory store, incomplete wiring, or an unavailable
target identity rejects during construction and mounts no Evals execution
surface. `target` and `store` are excluded from configuration serialization and
diagnostics. `ServerSettings` does not deserialize application objects,
credentials, database handles, or executable targets from environment values;
applications resolve those trusted objects before constructing `EvalsConfig`.

The protected `/api/evals` surface imports and downloads immutable corpora,
lists corpora/suites/cases, creates and lists runs, reads status and terminal
results, requests cancellation, checks comparison compatibility, and downloads
deterministic JSON or standalone HTML reports. Run admission requires an
`Idempotency-Key` header; Cayu persists only a target-scoped SHA-256 digest of
that value. HTTP documents can select only a stored corpus revision, suite, and
bounded concurrency, and can contract model steps or run-scoped token,
tool-call, elapsed-time, and server-priced cost limits. They cannot broaden the
target request or carry an application, import path, provider
credential, callback, PriceBook, tool/environment wiring, request template, or
another target.

Every Evals request authenticates before its handler runs, and JSON request
bodies cross a byte ceiling before parsing. Corpus import compiles every suite
against the attached target before the immutable revision is saved. Run
creation repeats the selected-suite compatibility check before persisting
admission, and the provider is never invoked in either path. A shared store is
filtered at its query and claim boundaries by target key; another target's
corpora and runs are neither exposed nor claimed by this server.

HTTP admission records a bounded server-verified subject/tenant projection, or
an explicitly unattributed HTTP source when the server is configured for open
access, alongside those contractions. It contains no authentication claims or
executable authority. The worker combines that durable invocation with the
current server-owned target and may only narrow the target's request. A stored
HTTP run that tries to inherit host-asserted SDK origin is rejected rather than
laundered across the server boundary.

The embedded coordinator claims persisted work with the store's private token,
epoch, and expiring lease, then invokes the same compiled execution core used by
the SDK/CLI `run_corpus_suite(...)` entry point. It heartbeats ownership,
observes durable cancellation, cancels fresh execution cooperatively, and
atomically publishes one credential-scanned terminal result. Ownership loss
stops local work without publishing. Controlled shutdown cancels and releases
still-owned work so another process or restart can claim it; an unclean stop
remains recoverable after lease expiry. No partial run is ever represented as
passed.

The fence covers publication, not external side effects. If a lease is lost
after a model request began, recovery may execute the candidate and judge calls
again; Cayu does not claim exactly-once model calls. Only the current fenced
owner can publish, and the terminal run's `attempt_count` records how many
ownership epochs were issued. Corpus trial/case/assertion ceilings and the
configured concurrency bound each attempt; operators should also apply normal
provider budgets and rate limits to judge apps.

SQLite is the embedded single-database choice. PostgreSQL permits multiple
server processes to compete safely for the same target's queued work through
fenced claims. This is not an arbitrary target registry, generic queue, remote
worker protocol, or hosted eval service.

### First-class dashboard Evals area

The bundled dashboard always exposes an **Evals** navigation area and accepts
authenticated direct links to `/evals`. The contract's required
`capabilities.evals_readiness` projection reports each product operation as
`ready`, `gated`, or `unsupported` with a closed reason code. The operations are
captured evaluation, catalog read and write, captured-result persistence,
scenario conversion, fresh launch, cancellation, comparison, and reports.
This projection is discovery metadata, not authorization: the authenticated
routes and their mutation and runtime checks remain authoritative.

If catalog reads are not ready, the page renders a deterministic readiness
shell and does not query absent Evals endpoints. Deployment-gated operations
remain visible with their factual missing dependency, planned framework work is
labeled separately from a genuine runtime incompatibility, and ready operations
remain independently visible. Project serving automatically assembles durable
storage, project/release identity, and the bounded generated target registry
when their declarations are available. Existing explicit Evals and
evaluation-promotion configuration remains supported and authoritative.

When `surfaces.evals.read` and `evals_readiness.catalog_read` are enabled, the
catalog pages through immutable corpus revisions, suites, and cases without
hydrating complete corpus documents. Operators can import an 8 MiB-or-smaller
corpus file, download canonical corpus JSON, select a suite, choose bounded
concurrency and runtime ceilings, and launch a durable run against the selected
published target. After a bounded browser preflight, import forwards the selected
file bytes unchanged so
the server's strict duplicate-key, UTF-8, portable-JSON, and corpus validation
remains authoritative. Mutation controls remain disabled when
`surfaces.evals.mutate` is unavailable even when catalog reads are allowed.

Launches use a cryptographically random `Idempotency-Key`. If a response is
ambiguous, retrying the unchanged launch reuses that key rather than duplicating
provider work. The dashboard follows queued, running, and cancelling records by
their durable run ID; cancellation is a server request, not a fabricated local
terminal state. Opaque catalog/run cursors and selected corpus, suite, run,
status, and comparison identities remain in bounded URL state. Superseded reads
are cancelled and a changed corpus never reuses another corpus's suite or case
projection.

The session-side **Run fresh trial** action uses the same retry registry and run
worker. Its request identity includes the captured candidate revision and every
execution setting, so changing a bound creates a new admission identity while an
ambiguous retry of unchanged work remains idempotent. Successful launch opens the
ordinary Evals run view; there is no dashboard-only execution engine.

A completed run exposes the complete safe published graph. The result view
shows target release/AppManifest identity, run/case/trial status and score,
duration, evidence completeness, output availability/truncation, usage,
observed or unavailable cost, diagnostics, and every assertion's structural
evidence. Cases and trials are selected rather than rendering as many as 10,000
assertion results into the DOM at once. JSON and standalone HTML downloads come
from the server's deterministic report endpoints. Only the actively viewed
complete result graph is retained by the query cache; switching runs or leaving
the result view evicts it immediately.

Comparison also remains server-authoritative. The dashboard uses the suite's
approved captured or fresh result by default; the operator can instead select
or paste another immutable `sha256:` result revision. The server returns both
origin-aware catalog records, its typed compatibility verdict, and immutable
summaries side by side. Different
application releases are allowed, while changed corpus, suite, case, assertion,
evidence-policy, target-key, or applicable-pricing contracts are explicitly
incomparable. For compatible runs, status regressions and score drops beyond the
selected tolerance are reported at run and case scope. The browser never invents
a baseline or a universal regression score.

The SDK, server, dashboard, JSON, HTML, and CLI share that same comparison
projection. `compare_eval_results(...)` accepts captured and fresh origins;
`eval_result_to_json(...)` and `render_eval_result_html(...)` report either
origin without reading an application or recomputing assertions.
`corpus_execution_comparison_to_json(...)` and
`render_corpus_execution_comparison_html(...)` render the shared comparison.
An incompatible comparison contains only typed mismatch reasons and result
summaries—never fabricated regressions.

### Release acceptance: dashboard to local CI

The repository's credential-free browser contract proves the complete
promotion, save/download, durable launch/cancellation, result inspection,
comparison, report, local rerun, and CI-exit journey against an installed Cayu
package:

```bash
python examples/dashboard_behavior_live.py
```

Release-artifact CI runs that script with the built wheel's Python executable,
with `PYTHONPATH` and provider credentials removed. Its local rerun deliberately
uses a different application release ID and must still compare cleanly against
the downloaded dashboard result.

The focused real-application check starts from a freshly generated project and
an actual OpenAI or Anthropic model. It runs `cayu serve --dev`, creates a
session in Control Plane, adds an output-content assertion, launches one fresh
trial, approves the captured result as baseline, compares captured to fresh,
downloads both result origins and their reports, and proves the stable CLI
comparison exit. The generated project contains no Evals-specific changes:

```bash
# Two agent executions: source capture and one bounded fresh trial.
CAYU_PROVIDER=openai OPENAI_API_KEY=... \
  uv run python examples/evals_release_acceptance_live.py

CAYU_PROVIDER=anthropic ANTHROPIC_API_KEY=... \
  uv run python examples/evals_release_acceptance_live.py
```

Model overrides use `CAYU_OPENAI_MODEL` or `CAYU_ANTHROPIC_MODEL`. The check is
credential-gated and never falls back to a scripted provider; a missing or
mismatched selected credential exits before project creation or application
execution. Nightly verification exposes it as
`evals-release-acceptance-live`.

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

Each `LLMJudge` session applies a durable zero-application-tool capability
ceiling. The registered judge agent may therefore have tools for other
workloads, but judge requests contain none of their definitions, and even a
provider-fabricated hidden call is blocked before authorization, approval, or
execution. This controls tool visibility and authority; it does not remove
candidate text from the judge prompt or replace the application's privacy
review for provider-bound evaluation data.

For authority-free corpora, use `ModelJudgeAssertionSpec` plus a trusted
`ModelJudgeTarget` on `CorpusTarget`. SDK `run_corpus_suite(...)`, corpus-mode
`cayu eval run`, and the server-attached durable worker all resolve and execute
that same compiled assertion. Unlike direct `LLMJudge` results, the portable
published result deliberately omits raw audit metadata and retains only its
bounded contract, score/outcome, and safe diagnostic.

Candidate quality and judge health are separate outcomes. Only a successfully parsed,
finite score can produce `passed` or `failed`. Invalid judge configuration, provider or
runtime failure, an attempted tool call, an incomplete judge session, empty output, and an
invalid score produce `error` with no numeric score. Release gates must treat that state as
inconclusive rather than evidence that the candidate failed the rubric.

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

Saved `EvalRun` baselines use schema version `7`. Version 7 preserves the complete
ordered trial graph, explicit outcome/null-score contract, conclusive-evidence
state, the exact portable assertion revision behind each result, and the optional
portable execution contract a trusted executor fixes before dispatch. It retains
identity-free aggregate usage for every complete trial, canonical large counters,
and durable-JSON validation. A contracted run must retain exactly the requested
number of trials for every case.
`load_eval_run(...)` rejects missing versions and versions 1–6;
regenerate those baselines with the current Cayu version. No compatibility loader
or migration is used.

Standalone exports use a versioned document envelope. The current trajectory
schema version is `4`; `load_trajectory(...)` rejects files without that version
or with an unsupported version before validating the trajectory payload. This is an
intentional clean break from Cayu's earlier unversioned preview exports: they
are not migrated and must be regenerated. The trajectory schema version is
independent from `EvalRun.schema_version`. Version 4 retains immutable session
invocation provenance and typed memory attribution in session-backed trajectories
while preserving Cayu's
durable-JSON contract: nonportable text or numbers, duplicate object keys, and
excessive nesting are rejected before an existing export can be overwritten or
an imported trajectory can be replayed. Earlier exports must be regenerated;
Cayu does not invent provenance while loading them.

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
