# Runtime-Native Evals

Cayu evals are designed to test agent behavior through the Cayu runtime, not
only final model text.

The goal is two-part:

- provide stable abstractions so applications can bring their own eval stack
- provide a simple local/CI default for teams that want something built in

The built-in runner evaluates normal `CayuApp.run(...)` sessions and then
asserts over the durable runtime state Cayu already owns: sessions, events,
transcripts, tool calls, usage, workspaces, and artifacts.

## Start with the workflow you need

The installed package includes three focused, version-matched guides:

- `cayu guide evals-first` creates a Control Plane suite, runs it, approves a
  baseline, and compares the next result.
- `cayu guide evals-ai-quality` adds bounded AI-judge authority, a rubric,
  reference truth, and calibration.
- `cayu guide evals-production` promotes retained sessions and covers
  multi-stage scenarios, tools, process behavior, and memory evaluation.

These are the onboarding paths. The rest of this document is the complete SDK,
schema, HTTP, storage, embedding, and operational reference.

## Minimal Example

```python
from cayu import (
    AgentSpec,
    EXTERNAL_CONTAINER_RESET_CONTRACT_REVISION,
    EXTERNAL_CONTAINER_RUNNER_REVISION,
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
- `EvalPlan(workflow_target=WorkflowEvalTarget(...), suite=suite)` for a direct suite
- `EvalPlan(workflow_target=WorkflowEvalTarget(...))` with `--corpus`
- `(app, suite)`
- an object or dict with `app` and `suite`, `corpus_target`, or `workflow_target`

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

This CLI target is hermetic test-program authority. `[tool.cayu].eval_target` builds the
app, fixtures, request base, and suites selected by `cayu eval run`; it is not the target
shown in the Control Plane. Control Plane trials instead use the current target published
by the running server, including that mounted app's current provider, tools, environment,
approval policy, and runtime policy.

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

## Author-first suite contracts

`EvalSuiteDraftV1` remains the closed deterministic authoring contract.
`EvalSuiteDraftV2` is the bounded, authority-free SDK and Control Plane
contract for a reusable evaluation that may also contain revision-free
`StructuredModelJudgeAssertionDraftV1` material. The server—not the browser—
compiles rubric and public-reference drafts into content-addressed immutable
identities during preview. `EvalSuiteDraftV3` adds revision-free `trials`,
`minimum_passed_trials`, and `max_concurrency` editor settings. The server—not
the browser—compiles those settings into an immutable
`EvalSuiteTrialPolicyV1` revision during preview. A draft contains one stable
suite ID, a published target key, bounded trial settings, and one or more case
drafts.
Each case selects exactly one stimulus:

- `EvalSimpleInputStimulusV1` carries the existing portable `RunInputSpec`;
- `EvalScenarioStimulusV1` links an exact, separately persisted scenario ID and
  content revision.

`compile_eval_suite_authoring_draft(...)` preserves the V1 wire contract or
compiles an explicit V2 or V3 draft, sorts cases by stable ID, and computes immutable
case, suite-spec, rubric, public-reference, and complete suite-document
revisions. Captured source
identity is optional for authored cases; Cayu does not fabricate production
provenance for a fresh input. When a source-less authored case is adapted to the
existing corpus runner, Cayu uses an explicitly domain-tagged authored-definition
identity derived only from the immutable case revision. It never claims that
identity is an application release or production capture, and it remains stable
across real application releases so the same saved suite can serve as a
regression contract. The fresh result separately records the actual execution
release and AppManifest. `add_eval_case(...)`, `duplicate_eval_case(...)`,
and `revise_eval_case(...)` return new immutable suite revisions. Case revision
checks reject stale edits without changing prior results.

`eval_suite_selection(document)` freezes the full suite. Passing explicit case
IDs freezes a non-empty subset instead. Both forms bind the suite document,
suite settings, and every selected case revision, so later edits cannot silently
change admitted work.

Protected servers with durable Evals expose preview/save/catalog/download at
`/api/evals/suites`. Preview canonicalizes a draft without persistence or
execution and reports target or scenario-reference readiness. Save accepts only
the exact reviewed revision, scans it through the target's credential-redaction
boundary, and atomically verifies every scenario reference. The authored-suite
 catalog was introduced at storage revision 64; current SQLite and PostgreSQL
 EvalStore implementations require revision 72 for restart-safe trial
 checkpoints. Custom EvalStore adapters remain readable but cannot admit fresh
 eval work unless they implement the fenced terminal-checkpoint contract and
 declare `trial_checkpointing = True`. These authoring contracts
do not create provider, tool, environment, fixture, secret, or runtime
authority. New V3 drafts still default to one trial, one required pass, and
concurrency one; V1/V2 documents remain readable with that fail-closed policy.

### Control Plane: create and run an evaluation

For an ordinary project started with `cayu serve`, users do not need to write an
Evals-specific Python suite. Select a server-published target on the **Evals**
page and choose **New evaluation**. The authoring sheet supports the complete
deterministic and structured-judge workflow:

1. name a reusable suite and set trials per case, required passes, maximum
   concurrency, and the per-trial timeout;
2. add, duplicate, remove, and select cases;
3. enter one or more ordered user messages for a simple fresh session, or build
   and save a controlled multi-stage scenario;
4. define expected behavior with status, output, tool, memory, model-step, usage,
   token, and cost assertions, and optionally add a one-to-eight-criterion AI
   judge rubric using a current trusted server-published profile;
5. check current target and scenario readiness, save the reviewed immutable
   suite revision, and review the exact maximum candidate work, judge work,
   priced cost (when fully priceable), and execution profiles for either the
   full suite or an explicit subset;
6. launch the selected cases and follow the resulting durable run in the normal
   Runs and Results views.

Saved revisions remain in the authoring catalog and can be loaded, revised into
a new immutable revision, or run again. A launch freezes an
`EvalSuiteSelectionV1`, so a subset always records the exact suite and case
revisions that were reviewed. A completed result can use the existing baseline,
comparison, JSON-report, and HTML-report workflow.

Editing a saved scenario immediately makes prior suite and launch readiness
stale. Save the new scenario revision, check and save the resulting suite
revision, and check launch readiness again before execution. Dirty scenario
work also locks case and suite transitions until the operator explicitly saves
or discards it, so changing views cannot silently lose a child-editor draft.
While a scenario operation is pending, the complete parent form remains locked
so a late response cannot be attached to a different case identity, draft, or
selection. Editable portable case IDs are not browser row identity: temporary
invalid or duplicate IDs do not move the active editor, change case selection,
or cross-wire in-progress scenario state while the operator corrects them.
Within suite authoring, an artifact override remains transient until **Prepare
fixture** embeds its reusable retained reference in the scenario. The scenario
cannot be saved into the suite, and parent transitions remain locked, until
every nonblank override has been materialized this way.

Simple-input cases in one selection execute together as one ordinary corpus
run. Each selected multi-stage scenario executes as its own restart-safe
scenario run because it has independent progress, approval, resume, and fixture
state. One launch response maps every selected case to its resulting run; mixed
selections therefore remain one user action without pretending that separately
recoverable scenarios share one runtime state machine.

Control Plane-authored suite launches execute the complete immutable trial
policy. Every admitted trial runs; Cayu does not stop early after enough passes
or failures. A case passes only when it reaches the required pass count and has
zero runtime errors, evaluator errors, unavailable required evidence, or
cancellations. The launch request must echo the reviewed exposure revision, so
changed execution profiles, pricing, selection, or work bounds force another
readiness check. Repeated assertion aggregates use trial pass rate against the
same required-pass rate; their metadata retains the mean per-trial graded score.
Simple cases share one run; each independently recoverable
scenario has its own run. All runs admitted by one suite launch share a durable
launch identity. Cayu partitions the reviewed concurrency ceiling across
durable lanes and leases at most one run in each lane, so the suite-wide cap
remains true across multiple coordinators without serializing independent work.
The browser
can author structured rubric, threshold, permitted evidence, and public or
server-held private reference identity, but it cannot create a provider,
credential, judge application, privacy policy, or private-reference content.
Only profiles and private-reference identities published for the selected
target are selectable.

Preflight keeps candidate and judge work separate. Candidate trials and maximum
model steps are always explicit. Runtime token and cost limits are post-observation
stop thresholds: one provider completion may cross them, so Cayu never presents
those thresholds as exact candidate maxima. Candidate token exposure remains
unavailable until the selected execution profile publishes a reservation-backed
bound, and candidate cost reports `candidate_cost_not_hard_bounded` when a priced
but non-reserved threshold is configured. Judge token and cost settings are also
post-observation stop thresholds, so judge maxima remain unavailable even when
every selected judge profile is priced. Preflight reports
`judge_cost_not_hard_bounded` for complete pricing and
`judge_pricing_incomplete` when pricing identity is incomplete rather than
displaying an unenforceable maximum or zero.
Completed results retain each trial's observed candidate usage/cost evidence and
each rubric-string or structured judgment's observed judge usage/cost; unpriced
observations remain explicitly unavailable.

### Control Plane: calibrate an AI judge

After adding a structured judge, choose **Check suite**. A successful preview
returns the exact compiled rubric, reference, profile, implementation, and
evidence-policy identity. The rubric editor then exposes **Calibrate on fixed
evidence**:

1. provide an operator-declared source ID, known task, and candidate output,
   plus the fixed transcript only when the reviewed evidence selection requires
   one;
2. enter a human score from `0` to `1` for every criterion;
3. choose one to ten repeated judge calls and check the token/cost ceiling;
4. run calibration and inspect every typed criterion judgment, aggregate
   disagreement, pass/fail agreement, evaluator error, usage, and cost record.

Calibration invokes only the isolated judge path. It never starts the candidate
agent, calls candidate tools, prepares fixtures, or creates environment effects.
Repeated trials therefore measure judge agreement and stability over one exact,
content-addressed operator-supplied evidence snapshot. The declared source ID is
part of that hash and preserves the operator's link to the source material; it
does **not** prove that the pasted evidence came from that source, measure
candidate reliability, or measure end-to-end task success. Same-model judging is never
silently chosen: the server publishes the candidate-route relationship, the
user must explicitly select the labelled profile, the profile must permit it,
and the preview and report retain the relationship.

Completed calibration reports are immutable and durable in SQLite/PostgreSQL
storage revision 68. The run request carries a stable run ID, so a response-loss
retry or server restart returns the already stored report without paying for or
executing the judge again. Public reference truth crosses the application
redaction boundary before persistence. Private reference content stays inside
the server-owned judge binding and is absent from the browser, portable suite,
calibration evidence, and report.

The protected HTTP contract mirrors the same review boundary:

- `POST /api/evals/suites/preview` canonicalizes an unsaved draft;
- `POST /api/evals/suites` saves the exact reviewed revision;
- `GET /api/evals/suites` and `GET /api/evals/suites/{revision}` list and load
  reusable immutable suites;
- `POST /api/evals/suites/{revision}/runs/preview` freezes and preflights a full
  or subset selection;
- `POST /api/evals/suites/{revision}/runs` admits that selection with an
  `Idempotency-Key`;
- `POST /api/evals/judge-calibrations/preview` compiles fixed evidence and human
  labels and reports exact judge work without execution;
- `POST /api/evals/judge-calibrations` runs or idempotently recovers the reviewed
  calibration;
- `GET /api/evals/judge-calibrations/{revision}` loads an immutable completed
  report.

These routes select existing server-owned application authority. They never
accept providers, secrets, tool implementations, environment objects, database
connections, or callbacks from the browser.

## Portable corpus documents

`EvalCorpusDocument` is Cayu's bounded, JSON-portable definition format for
reusable eval suites and cases. A document describes exactly one trusted
`target_key`. A runnable case contains only user-role text input; a captured-only
case uses `input: null`. Documents also contain bounded trial settings,
diagnostic source/pricing identities, an explicit evidence policy, and a closed
set of structural assertion specifications. It cannot contain a `CayuApp`,
provider/model/environment selection, import path, callback, raw session ID, or
runtime event payload.

Portable corpus schema version 4 adds bounded memory-attribution assertions to
the version 3 surface, which introduced the explicit immutable suite trial
policy. The assertion surface covers root and child terminal status,
final-output equality/containment, tool presence/order/count, bounded
tool-argument/result JSON subsets, workspace-file structure, artifact structure
and explicitly retained public artifact text, memory admission/provider-exposure
ranges, model-step and token limits, recorded usage, estimated-cost limits, and
trusted model judgments. Cost assertions require a `PricingProfileIdentityV1`;
the identity fingerprints trusted pricing used elsewhere and never embeds or
authorizes a `PriceBook`. A `ModelJudgeAssertionSpec` remains data-only: it
carries a bounded rubric and rubric version, threshold, transcript-selection
flag, and trusted evaluator key. The target resolves that key to its local
trusted judge implementation at execution time. Each published result records
the resolved implementation revision, so the same portable corpus remains
reusable across trusted evaluator rollouts while comparisons still reject
different judges.

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
`load_eval_corpus(...)` enforce schema version 4 and Cayu's durable-JSON rules,
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
Unknown fields and assertion kinds fail closed; schema version 4 has no
prior-version compatibility loader.

### Safe tool argument and result assertions

`ToolArgumentsContainAssertionSpec` selects one call by exact registered tool
name and one-based `occurrence`, then checks that its finalized public arguments
contain a bounded expected JSON object. `ToolResultContainsAssertionSpec` uses
the same stable selection but is admitted only when the trusted target enables
public-safe result retention. In Control Plane these are **Tool arguments
contain JSON** and **Tool result contains JSON**; they are available in new
evaluations, suite edits and duplication, and captured-session promotion.

```python
from cayu import ToolArgumentsContainAssertionSpec, ToolResultContainsAssertionSpec

arguments = ToolArgumentsContainAssertionSpec(
    id="search-query",
    tool_name="search",
    occurrence=1,
    expected_subset={"query": "cayu", "filters": {"team": "runtime"}},
)
result = ToolResultContainsAssertionSpec(
    id="search-succeeded",
    tool_name="search",
    occurrence=1,
    expected_subset={"structured": {"status": "ok"}, "is_error": False},
)
```

Object matching is recursive subset matching: extra actual object keys are
ignored. Arrays are positional and require equal length. Strings, booleans,
and null compare by exact JSON kind and value; finite JSON numbers compare by
their exact decimal value, so `1` and `1.0` match but `true` and `1` do not.
There is no regex, JSONPath, executable predicate, schema evaluator, or
unbounded deep equality. Expected and retained values are limited independently
to 4 KiB, 12 levels, and 128 nodes; each trial retains at most 256 ordered call
identities. Tool-result subsets may select only `content`, `structured`, and
`is_error` and must select at least one of them.

The standard policy retains finalized tool arguments because they already cross
Cayu's public runtime event boundary. It does not retain tool results. A trusted
application owner may opt into public-safe result evidence with
`EvaluationEvidencePolicySpec.create(include_tool_results=True)` on the target;
HTTP and browser callers cannot grant that authority. Cayu applies the
application secret-redaction boundary before either value becomes portable and
never falls back to raw transcript arguments or unrestricted runtime results.

Outcomes keep the reason exact. A missing selected occurrence is a conclusive
`failed`; a present value that does not contain the subset is also `failed`.
Unsupported capture, unavailable terminal evidence, malformed retained data,
incompatible call identity, cardinality overflow, truncation, and redaction on
an expected path are distinct `unavailable` observation states. Redaction on an
unselected extra field does not prevent a safe comparison. These states and the
bounded safe actual value are the same in SDK/HTTP results, Control Plane
drill-down, CLI, and JSON/HTML reports.

### Portable workspace and artifact assertions

`WorkspaceFileAssertionSpec` checks one canonical relative POSIX path for
presence or absence. A present-file expectation may also require minimum and
maximum byte size and an exact lowercase SHA-256 digest. The portable path has
no glob, directory walk, absolute root, parent traversal, platform alias, file
permission, owner, timestamp, or executable predicate.

`ArtifactAssertionSpec` selects artifacts owned by the current session or
current environment. It can filter by exact filename, content type, and byte
range; optionally require a whole-artifact SHA-256 digest or bounded text; and
enforce a minimum and optional maximum matching count.

```python
from cayu import ArtifactAssertionSpec, WorkspaceFileAssertionSpec

workspace_output = WorkspaceFileAssertionSpec(
    id="workspace-report",
    path="output/report.json",
    minimum_bytes=1,
)
artifact_output = ArtifactAssertionSpec(
    id="published-report",
    scope="session",
    filename="report.json",
    content_type="application/json",
    min_count=1,
    max_count=1,
)
```

Both kinds are available in **Evals → New evaluation**, saved-suite editing and
duplication, imported/exported corpus JSON, and captured-session promotion. No
Python assertion class is required. Control Plane supplies expectation values;
the trusted target continues to own the app, environment, stores, redactor, and
evidence policy.

Workspace evidence is always structural. Cayu reads only the declared paths
under a fixed 1 MiB ceiling and retains presence, exact total size, and a digest
only when the complete object was read. Raw workspace bytes never enter the
portable evidence, result, store, API, report, or browser. A partial read cannot
stand in for a whole-file digest. Application-owned Python code may still use
`WorkspaceFileContains` when a private content predicate is appropriate; that
assertion is deliberately not portable or browser-authored.

Artifact listing is restricted at the store to the current session or
environment and only structurally matching candidates are retained. Cayu reads
artifact bytes only when a declared digest or text check needs them. Reads are
bounded to 1 MiB. Unsupported media, invalid UTF-8, truncation, redaction,
malformed store output, or an unavailable read produces an unavailable
observation rather than a candidate mismatch. Portable details omit artifact
IDs, store/session/environment identities, timestamps, arbitrary metadata, and
binary content.

Artifact text is off by default. A trusted application owner may enable it on
the execution profile with
`EvaluationEvidencePolicySpec.create(include_artifact_text=True)`; HTTP and
browser input cannot grant that authority. Only supported textual media are
decoded, only a complete bounded object is eligible, and the application
redactor runs before at most 64 KiB of text becomes portable. A suite containing
`text_contains` is rejected during compilation and reported not ready in
Control Plane when the selected profile does not publish that exact policy.

Captured-session quick-adds use only complete retained observations and label
them as observed drafts, not truth. When a production trajectory did not retain
a path/scope, or retained it incompletely, Control Plane creates an explicit
editable expectation instead of inventing an observation. Fresh execution then
captures exactly the probes required by that reviewed assertion.

### Memory assertions and correct-use judging

`MemoryAttributionAssertionSpec` is the portable deterministic check for
runtime-proven memory structure. It requires complete bounded attribution and
can constrain admitted-item and proven provider-exposure counts independently
within the fixed capture envelope (0–1,000 admitted items and 0–100 exposure
records per trial).
Truncated, unavailable, contradictory, or indeterminate exposure evidence is
`unavailable`; it is never converted into a failed count comparison. Session
aliases are public-safe correlation aids, not stable case or expectation
identity, and the assertion never matches on them.

```python
from cayu import MemoryAttributionAssertionSpec

memory_reached_provider = MemoryAttributionAssertionSpec(
    id="memory-reached-provider",
    min_admitted_items=1,
    min_provider_exposures=1,
)
```

Ordinary users do not need this Python form. In **Evals → New evaluation →
Evaluate memory**, choose **Require memory exposure** to add the same structural
contract. When promoting a completed production session, quick-add freezes only
complete observed counts; otherwise it creates no false observed claim.

Structural exposure does not prove correct use. Choose **Add reference-backed
judge**, replace the deliberately blank fact with trusted reference truth, and
review the `memory-use` rubric. Its three default criteria score agreement with
the reference, relevant grounded use, and avoidance of unsupported
memory-derived claims through the existing structured-judge system. The target
must publish a trusted judge profile that permits final-output and
public-reference evidence; the browser cannot create provider, credential,
privacy-policy, or private-reference authority. A judge score still does not
prove that memory caused the result—the same answer may have been possible
without memory.

Results therefore present three explicit claims:

- **Structural** shows admission, provider exposure, completeness, limitations,
  and bounded source-tree evidence.
- **Semantic use** shows only an admitted `memory-use` structured judgment bound
  to trusted reference truth.
- **Causal contribution** remains not established for an ordinary run and
  requires a paired memory intervention experiment.

This separation applies equally to a newly authored fresh session and a
promoted production session. Tool-call, tool-argument/result, process, output,
and memory assertions can be combined in one case when the expected behavior
includes searching the right source, calling the right tools, and reaching the
right end state.

### Portable lifecycle, approval, and child assertions

`ProcessEventAssertionSpec` requires or forbids one decision-useful runtime
fact with bounded minimum and maximum counts. `ProcessEventsInOrderAssertionSpec`
is the advanced protocol form: Cayu filters the root session's ordered event
record to the fact kinds named by the assertion and then requires the complete
filtered sequence, including duplicates, to match exactly. Other runtime events
do not affect that selected protocol check.

```python
from cayu import ProcessEventAssertionSpec, ProcessEventsInOrderAssertionSpec

approval_required = ProcessEventAssertionSpec(
    id="approval-required",
    event="tool_approval_requested",
    min_count=1,
)
approval_not_denied = ProcessEventAssertionSpec(
    id="approval-not-denied",
    event="tool_approval_denied",
    min_count=0,
    max_count=0,
)
approval_protocol = ProcessEventsInOrderAssertionSpec(
    id="approval-protocol",
    events=(
        "tool_approval_requested",
        "tool_approved",
        "tool_call_started",
        "tool_call_completed",
    ),
)
```

The closed vocabulary covers session started, resumed, awaiting-input,
completed, failed, interrupted, and limit-reached facts; tool started,
completed, failed, and blocked facts; approval requested, approved, denied, and
expired facts; structured-output validated and failed facts; and the budget
limit-reached fact. Portable input cannot name custom events or internal event
strings and cannot inspect raw payloads, approval IDs, actors, reasons, policy
metadata, or secrets. Direct Python `EventOccurred` remains available for
application-owned specialized assertions, but it is intentionally not a
portable Control Plane or corpus contract.

Process evidence retains at most 4,096 allowlisted facts. Missing root evidence
and bounded prefixes are `unavailable`, never a pass or candidate mismatch.
Exact ordering applies only within an ordinary root session. A typed workflow
target projects the workflow's causally ordered child closure in durable lineage
order. `ChildStatusAssertionSpec` separately
supports completed, failed, and interrupted direct children; incomplete child
tree capture remains unavailable. Control Plane can author all three contracts
for a new suite or captured-session promotion, and SDK, stores, result
presentation, CLI, comparison, and JSON/HTML reports consume the same evaluator
and safe published details.

Captured-session quick authoring uses only complete process evidence. For a
trace longer than the 256-event order contract, it selects whole observed event
kinds whose complete filtered occurrence sequence fits the bound; it never
truncates one kind or substitutes generic events. When evidence is incomplete
or no exact observed sequence is representable, Control Plane labels the action
`Add expectation` and creates an editable explicit draft instead of claiming
the default was observed.

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
bootstrap, and input ceilings but cannot raise Cayu's hard limits. Concurrency
is different: the target publishes a finite per-run ceiling, but Runtime does
not impose a universal upper bound on the operator-selected value.
The target key and application release ID are public result identity: both must
cross the target app's workload-secret redaction boundary unchanged, or target
validation fails before provider dispatch.

### Application-owned workflow targets

`WorkflowEvalTarget` makes a `WorkflowBase` run—not one child agent session—the
candidate under evaluation. It reuses the same `EvalSuite`, corpus compiler,
assertions, scheduler, stores, reports, CLI, and server execution path:

```python
from cayu import (
    EvalPlan,
    WorkflowEvalExecution,
    WorkflowEvalResult,
    WorkflowEvalTarget,
)


def build_execution(invocation):
    runtime = build_runtime_for_trial(invocation)
    return WorkflowEvalExecution(
        app=runtime.app,
        workflow=RefundWorkflow(runtime.app),
        close=runtime.close,
    )


def project_result(evidence):
    answer = evidence.completion_event.payload["answer"]
    return WorkflowEvalResult(final_output=answer, structured_output={"answer": answer})


workflow_target = WorkflowEvalTarget(
    key="refund-workflow",
    app=build_manifest_reference_app(),
    request_base=workflow_profile_probe_request,
    application_release_id="refund-service-2026-09-04",
    workflow_spec=RefundWorkflow.spec,
    implementation_revision="sha256:...",
    result_projector_revision="sha256:...",
    execution_scope_revision="sha256:...",
    workflow_factory=build_execution,
    result_projector=project_result,
)

# Add suite=direct_suite for direct Python/CLI execution, or omit it and pass a corpus.
eval_plan = EvalPlan(workflow_target=workflow_target)
```

The factory is invoked exactly once per concrete case trial. Its input contains
only bounded case messages, public run/suite/case/trial identities, the
recovery-stable workflow run ID and idempotency key, and the target's bounded
application context; evaluator assertions and private evaluator state are not
exposed. The default `per_trial` scope requires a fresh application and workflow
instance for every trial. `shared` is opt-in, sequential only, and requires the
factory to return the same application/workflow pair. Every factory-returned
application must resolve the target's profile-probe request to the same exact
execution-profile fingerprint as the trusted target application. A per-trial
factory that rebuilds process-local providers, tools, policies, hooks, or
environments must give those components stable behavior identities when the
rebuilt instances are intended to be equivalent.

The root ID is deterministic for the target revision and concrete run/case/trial
slot. A replacement durable worker therefore re-enters the same workflow journal
and uses Runtime's existing attempt fencing and child-step recovery instead of
creating another retry/checkpoint system. Executable factories and projectors
remain in the registered Python target; corpus and server records retain only
their public behavior identity. Recovery checkpoints discard raw text,
structured output, transcripts, and trajectories after the bounded public
projection has been prepared.

The projector runs only after Cayu proves exactly one `workflow.completed` event
for the current durable workflow attempt. Its bounded typed result is bound to
that completion event and target/projector revisions. A missing, conflicting, or
superseded completion, invalid projector output, or failed/expired `close`
callback produces a stable error diagnostic and cannot publish candidate output.
The retained root trajectory contains the workflow journal, complete admitted
child tree, aggregate child usage, workflow-wide tool/process evidence, and the
typed structured output. Workspace or artifact evidence fails unavailable when
multiple child environments make one unambiguous observation impossible.
Workflow case timeout and caller cancellation cover factory construction,
workflow draining, projection, evidence capture, and assertions; the optional
`close` callback then has its own positive, finite timeout capped at 300 seconds.
`max_concurrency` continues to bound concrete Evals trials, while application
factories remain responsible for the Runtime environments owned by one trial.

Before provider dispatch, `compile_corpus_suite(...)` revalidates the complete
corpus and matches its target key, evidence-policy revision, applicable trusted
PriceBook identity, selected suite, and trial/input/case/timeout ceilings. It
compiles the allowlisted assertions through the same portable assertion adapter
used by `compile_assertion_spec(...)`; there is no second evaluator. All cost
assertions in the selected suite share one
compile-time pricing binding, so the trusted PriceBook is validated and
fingerprinted once for the suite rather than once per assertion.

### External process targets

An external process target runs an immutable multi-file candidate body through
the same corpus, scenario, assertion, usage, publication, comparison, and
durable-worker paths as an ordinary Cayu agent. It is a generic runtime seam;
benchmark selection, hidden evaluator truth, promotion policy, and baseline
policy remain outside Cayu.

`ExternalBodyReleaseV1.from_directory(...)` content-addresses every relative
regular file and executable bit, the canonical private-runtime path and file,
launch protocol, and entrypoint.
Symbolic links, special files, more than 512 files, and bodies larger than 32
MiB fail before execution. `ExternalProcessTargetIdentityV1` independently pins
that body, the evaluator runtime, target implementation, runner, environment,
fresh-reset contract, and evidence policy. Changing any one produces a new
target revision. `ExternalTrialIdentityV1` then binds the native eval run,
target, corpus, suite, case, and trial number before dispatch. These identities
are published in `CorpusExecutionResult.external_trials`; comparison validates
their order and exact contract instead of inferring lineage from a container or
session name.

```python
from cayu import (
    AgentSpec,
    ExternalBodyReleaseV1,
    ExternalContainerOperationAdapter,
    ExternalProcessModelProvider,
    ExternalProcessTargetIdentityV1,
    external_container_environment_revision,
)

image = "registry.example/candidate@sha256:" + image_digest
body = ExternalBodyReleaseV1.from_directory(
    candidate_root,
    private_runtime_path="private_runtime.py",
    launch_protocol_revision=launch_protocol_revision,
    entrypoint=(
        "python3",
        "{body}/private_runtime.py",
        "{body}/agent.py",
        "{request}",
    ),
)
identity = ExternalProcessTargetIdentityV1.create(
    body=body,
    evaluator_runtime_revision=evaluator_runtime_revision,
    target_implementation_revision=target_implementation_revision,
    runner_revision=EXTERNAL_CONTAINER_RUNNER_REVISION,
    environment_revision=external_container_environment_revision(
        image=image,
        runtime="runsc",
    ),
    reset_contract_revision=EXTERNAL_CONTAINER_RESET_CONTRACT_REVISION,
    evidence_policy_revision=evidence_policy.revision,
)
operations = ExternalContainerOperationAdapter(
    identity=identity,
    body_root=candidate_root,
    state_root=external_state_root,
    image=image,
    runtime="runsc",
)
app.register_provider(
    ExternalProcessModelProvider(identity=identity, operations=operations),
    default=True,
)
app.register_agent(AgentSpec(name="external-candidate", model="cayu.external-process.v1"))
# Set CorpusTarget.external_process=identity and use this agent in request_base.
```

The reference adapter admits only digest-pinned images on Docker with `runsc`
or Kata. Each trial receives a fresh non-root container with no network, no
mounts, no Linux capabilities, a read-only root, bounded tmpfs, CPU, memory,
PID, input, output, and daemon-log limits, and no inherited host environment.
The body and sealed request are copied while the container is stopped. Runtime
resolved attachments remain digest-attested in that bounded request. Structured
scenario JSON and files cross the existing typed provider boundary; corpus
assertions, expected answers, judge references, evaluator credentials, and
private evaluator state do not.

`RunInputSpec.opaque_external_case_ref` provides a bounded, revisioned alias for
a private evaluator case. It is accepted only for a target with an exact
`external_process` identity. A candidate may receive that alias in the private
launch envelope, but Cayu neither resolves it nor treats it as evaluator truth.
Ordinary corpus targets reject opaque references before provider dispatch.

The container name and external-effect key derive from the exact trial revision,
not from a retry-specific provider attempt. The canonical resolved launch request
has a separate digest committed into the operation authority, recovery alias,
container labels, provider state, and terminal receipt. Reusing a trial identity
with a different request fails before another effect can be adopted. Provider
start keys are atomic aliases to that bound effect, and the recoverable preparation
phase is retained before an alias becomes visible. A local phase journal and
identity labels reconcile process loss before create, after create/copy, after
start, and after completion; identity drift and ambiguous daemon state fail
closed. The reference adapter does not silently rerun a completed exact trial.
After a proven completed, failed, OOM, or cancelled state, it atomically retains
the bounded identity-bound terminal receipt and removes the container; exact
recovery validates the retained authority and request before using that receipt.
Cancellation targets the same operation, and terminal states remain distinct as
`failed`, `unavailable`, `cancelled`, `unknown`, `incomplete`, or
`identity_mismatch` through native publication and comparison. Output previews
remain bounded while the trusted assertion path consumes the complete bounded
candidate output. Candidate-private usage is retained only as explicitly
untrusted diagnostic output; it does not enter Cayu's native usage, cost, or
budget accounting. A trusted outer adapter must provide authoritative accounting
evidence through a separately owned contract when that evidence is available.

Trial scheduling is the normal work-conserving corpus scheduler: repeated trials
from one case can occupy every configured concurrency slot while published
ordering stays deterministic. Use `SQLiteEvalStore` for restart-durable embedded
admission/publication and a durable runtime `SessionStore` for production
session recovery. The reference container state directory must also survive a
worker restart. SQLite fencing controls publication; exact trial-keyed container
reconciliation controls this external effect. Custom adapters must provide an
equivalent durable idempotency and reconciliation contract before claiming
restart-safe external execution.

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

### Structured rubric judges

`StructuredModelJudgeAssertionSpec` is the typed, reproducible judge contract.
It replaces one opaque scalar decision with one to eight stable criterion IDs,
canonical decimal weights that sum exactly to `1`, a short explanation and
score per criterion, and a Cayu-computed weighted aggregate. The model does not
choose the aggregate or pass threshold. Missing, duplicated, reordered, extra,
out-of-range, overlong, or otherwise malformed criterion output is an evaluator
`error` with no candidate score. Corpus validation also caps each suite's
worst-case public explanation expansion at 2 MiB of characters before execution,
so trials and criterion counts cannot turn individually bounded responses into
an unbounded or predictably unpublishable result graph.

Every structured assertion pins an exact public `JudgeProfileIdentityV1` from
`model_judge_profile(...)` or the target catalog's `judge_profiles`. The profile
contains only its safe label, provider/model route, implementation revision,
allowed evidence, privacy-policy identity, timeout/token/cost ceilings, and
same-model posture. It contains no credential, private provider option, raw
prompt, system prompt, or executable object. Compilation rejects a missing or
changed profile before candidate dispatch. Cayu never infers a judge from the
candidate route. If the configured judge resolves to the same provider and
model, the profile must explicitly set `allow_same_model=True`; successful
results then retain `candidate_route_relation="same_model"`.

```python
from cayu import (
    EvalJudgeEvidenceSelectionV1,
    ModelJudgeTarget,
    PublicJudgeReferenceV1,
    StructuredModelJudgeAssertionSpec,
    StructuredRubricCriterionV1,
    StructuredRubricV1,
    model_judge_profile,
)

judge = ModelJudgeTarget(
    key="quality-judge",
    label="Quality judge",
    app=judge_app,
    agent_name="judge",
)
profile = model_judge_profile(judge)
rubric = StructuredRubricV1.create(
    id="answer-quality",
    criteria=(
        StructuredRubricCriterionV1(
            id="correctness",
            name="Correctness",
            description="The answer is factually correct.",
            weight="0.7",
        ),
        StructuredRubricCriterionV1(
            id="usefulness",
            name="Usefulness",
            description="The answer directly helps the user.",
            weight="0.3",
        ),
    ),
)
reference = PublicJudgeReferenceV1.create(
    id="refund-policy-answer",
    expected_facts=("The standard refund window is 30 days.",),
)
quality = StructuredModelJudgeAssertionSpec(
    id="answer-quality",
    judge_profile_key=profile.key,
    judge_profile_revision=profile.revision,
    rubric=rubric,
    reference=reference,
    threshold="0.8",
    evidence=EvalJudgeEvidenceSelectionV1(include_transcript=False),
)
```

Public reference answers/facts are portable but evaluator-only: execution sends
them to the judge, never to the candidate. They must also cross the candidate
application's workload-secret redaction boundary unchanged before execution.
Private truth uses
`PrivateJudgeReferenceTarget.create(...)` on the trusted server and places only
its `portable_identity()`—key, content revision, and privacy-policy identity—in
the corpus. Missing content, revision drift, policy drift, or disallowed
transcript/reference selection fails before dispatch. Private content never
enters corpus JSON or public results; criterion explanations are withheld for a
private-reference judgment so a judge cannot echo that truth through the result.
For public/no-reference judgments, explanations cross both the judge and
candidate applications' redaction boundaries and publish with an explicit
`available`, `redacted`, or `unavailable` state.

Structured judge calls remain tool-free and run in a private process-local
session store rather than the configured judge application's ordinary session
catalog. This keeps evaluator prompts—including private reference truth—out of
durable/control-plane session data even if cleanup fails. Calls run under the
profile's wall-clock, input-token, output-token, and total-token ceilings. Supplying both
`max_estimated_cost` and a trusted `PriceBook` adds a priced run budget that
rejects unpriced usage. Timeout, budget, provider, session, parser, and tool-call
failures are evaluator errors and never become a zero candidate-quality score.
The weighted aggregate remains exact decimal evidence. If an aggregate and
threshold are so close that the existing public float score envelope would
reverse their exact ordering, Cayu reports an evaluator configuration error
instead of publishing the wrong pass/fail result.
Successful judgments retain bounded observed judge model-step and token usage.
Profiles with trusted pricing also retain the exact currency-local estimated
cost and priced/unpriced step partition; profiles without pricing publish an
explicit unavailable cost state rather than an invented zero.
Suite-authoring schema V1 remains closed to deterministic kinds, including the
workspace and artifact structural contracts. Schema V2 carries the separately
versioned revision-free rubric/reference draft and immutable structured-judge
document contracts rather than admitting judge authority through V1.

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
different evaluation contracts. Scalar model-judge rubric text/version, threshold,
transcript selection, and evaluator key contribute to the assertion and corpus
revisions. The resolved implementation revision contributes only to the
published assertion binding, so a judge rollout leaves a portable corpus valid
while making cross-revision results incomparable rather than manufacturing a
score delta. Structured judges instead pin the exact public judge-profile
revision—including its implementation revision—in the corpus; a changed
profile therefore requires a reviewed corpus revision before another run.

Portable assertions consume one immutable `AssertionEvidenceView`, produced by
`project_assertion_evidence_view(...)` from a validated `Trajectory`. The view
contains only terminal statuses, bounded redacted final output, requested tool
names and counts, model-step/token counts, and optional currency-local cost
totals. Schema version 5 also carries one versioned
`EvalMemoryAttributionEvidenceV1` section. That section retains bounded runtime
receipt/exposure attribution, exact lifecycle states and fingerprints,
deterministic root/descendant tree paths, optional HMAC session aliases, and
explicit completeness and limitation codes. It carries no raw recalled text,
prompt, provider body, embedding, unrestricted event, or arbitrary private
metadata. The rest of the assertion view carries no session, event,
interaction, provider, model, agent,
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

Fresh eval execution captures memory only after the exact terminal root and
bounded descendant trajectory are available. It uses the same runtime
projection as historical `trajectory_from_session(...)` promotion and performs
a second bounded read before closing the trial; a changed closure becomes
typed `closure_changed` unavailable evidence rather than choosing either read.
The resulting trial-owned section is also the section consumed by portable
assertions, so suite-partitioned effective bounds are not replaced by defaults
during assertion or captured-result preparation.
The eval policy retains at most 100 source records per trial and partitions
aggregate 32 MiB source-read and 8 MiB runtime-projection budgets across both
closure reads before trials dispatch, with per-trial ceilings of 4 MiB and
512 KiB. It separately partitions at most 10,000 retained source wrappers and
9 MiB of final serialized memory evidence across the complete run before the
first trial dispatches. Large suites therefore reduce descendant retention—or
retain only an explicit projection-limit result—instead of completing provider
work and then exceeding the published-result ceiling. Runtime truncation remains distinct
from an eval source or byte limit because the runtime does not expose which of
its own limits fired. A complete zero-record projection is the only state that
sets `proves_empty`; missing, read-failed, redacted, truncated, contradictory,
deleted, legacy, incomplete-tree, deadline, and closure-change evidence cannot
be interpreted as empty. Deletion is asserted only from positive terminal
source evidence, never inferred from an unavailable or truncated store read.
For exposures, that positive census is the set of runtime-authored assistant
model attempts for model steps that durably admitted automatic recall;
auxiliary context-compaction attempts are excluded, and repeated count/model
events for one assistant attempt are deduplicated by runtime attempt identity.
Selected/planned, prepared, dispatch-started, acknowledged, completed, and
indeterminate exposure states remain the exact runtime lifecycle values.

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
tool-presence/count assertions use calls that actually started. Tool JSON
assertions select the same canonical started-call order, but read arguments
only from the matching finalized terminal event and results only from an
explicitly enabled public-safe result projection.

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
produces a content-addressed schema-version-10 `PublishedEvalRun` containing
every case, trial, exact source-trial revision, assertion outcome, safe
structural detail, duration, and identity-free aggregate usage. Every trial also
retains the exact bounded memory-attribution section used during evaluation, so
JSON, SDK, CLI, HTML, and Control Plane readers do not need raw runtime events
to distinguish complete, truncated, and unavailable evidence. Captured-session
scoring embeds the same
section in `CapturedRunScoreV1`; fresh and captured publications therefore use
one memory evidence model and revision contract. CLI JSON and Control Plane
schemas preserve the complete section. HTML reports retain its bounded
classification, limitation, and lifecycle summary while explicitly directing
full record inspection to JSON. The current compact generic
result-comparison projection explicitly reports memory attribution as
`unsupported` rather than silently dropping or comparing it; repeated-trial
memory statistics belong to the dedicated comparison contract. Every complete
trial carries its exact aggregate
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
Scalar model-judge results retain the bounded rubric and rubric version, threshold,
transcript-selection flag, evaluator key, implementation revision, continuous
score/outcome, fixed safe diagnostic, exact admitted judge profile and route,
and observed judge usage with priced cost or an explicit unpriced state. A valid
finite score is candidate evidence: the threshold decides `passed` versus
`failed`. Missing authority, judge configuration drift, provider/runtime failure,
an attempted tool call, incomplete accounting, an incomplete session, empty
output, or an invalid score is `error` with no numeric score; judge failure is
never converted into a candidate-quality failure.
Structured model-judge results retain the exact safe judge profile, route
relation, rubric/reference identities, evidence selection, canonical criterion
weights and scores, bounded explanation publication states, aggregate, threshold,
and fixed diagnostic. They never retain the raw prompt, raw judge response,
credentials, private options, or private reference content.
Trial, case, and run scores and statuses are rederived from the retained
published children. Public diagnostics use fixed Cayu-owned reason codes and
messages that distinguish assertion, lifecycle, evidence, and timeout failures
without copying raw exception text. Raw assertion metadata, raw final output,
trajectories, concrete session IDs, and provider/model identity are never copied.
Cost results require
the corpus pricing-profile fingerprint. The closed schema-v4 graph is bounded to
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
- portable workspace file presence/absence, size, and whole-file digest
- portable artifact metadata, size, count, whole-object digest, and opt-in public text
- application-owned workspace content and artifact creation assertions

`MaxEstimatedCost` fails closed: if even one observed model step has no matching
price, its outcome is `unavailable` and the retained cost summary reports both
priced and unpriced coverage instead of treating the missing price as zero.

## Workspace isolation

Cases in a suite run against the **same** `CayuApp`. Each case is a separate session, but it
shares the app's workspace unless you register the environment with an **environment factory**
(`register_environment_factory(...)`), which provisions a fresh environment — and a fresh
workspace — per session.

Because of this, `WorkspaceFileAssertionSpec`, `WorkspaceFileExists`, and
`WorkspaceFileContains` assert *"the file has this state in the workspace when
the case finished"*, **not** *"this case created it"*. With a single shared
environment, a file written by an earlier case can satisfy a later case's
workspace assertion. To make creation itself meaningful, register an
environment factory or use a trusted reset/fixture profile. The portable
assertion does not infer provenance from file presence.

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
  safely reconstructable invocation, launch a bounded current-app trial. This uses the mounted
  application's current provider, tools, environment, approvals, and policy; it is distinct from
  the hermetic target configured through `[tool.cayu].eval_target`. Scenario-v2 extends that path
  to ordered queued input, explicit session resumes, `ask_user` answers, files, portable JSON, and
  fresh approval checkpoints when current launch authority passes readiness preflight.

## Results and repeated trials

Every captured or fresh immutable result can be compiled with
`present_eval_result(...)` into `EvalResultPresentationV1`. This bounded,
versioned projection is the single human-facing result contract used by the
protected HTTP API, Control Plane, and HTML reporting. It does not rerun an
assertion or reinterpret raw model output. It validates and presents only the
facts already admitted into the immutable public result.

The projection retains the result revision and its underlying published-run or
captured-score `evaluation_revision`, together with target/release, app
manifest, corpus, suite, case, trial, assertion, evidence-policy, rubric,
reference, judge-profile, and evaluator-implementation identities.

The projection deliberately separates six dimensions that a single score
cannot represent safely:

- candidate outcome;
- deterministic assertion outcome;
- semantic quality;
- evaluator health;
- candidate runtime outcome; and
- retained evidence completeness.

An evaluator error therefore appears as an evaluator error and a candidate
that was not scored, not as a zero-quality candidate. Runtime failure and
unavailable or incomplete evidence are likewise explicit. CI callers can keep
using the stable `0` / `1` / `2` exit contract: candidate failures and genuine
regressions return `1`; evaluator/runtime failure, incomplete evidence, and
incompatible or identity-unmatched comparisons return `2`.

For every AI judgment, the presentation retains the safe judge-profile and
evaluator-implementation identity, independent/same-model label, diagnostic,
observed usage, and priced cost or explicit unpriced state. Structured judgments
add rubric and reference identity, evidence selection and privacy policy,
criterion scores and explanation states, exact weights and Cayu-computed
contributions, aggregate, threshold, and threshold outcome. Rubric-string
judgments retain their bounded rubric revision and threshold. Private reference
content, judge prompts, provider credentials/options, and raw judge output are
not part of this contract.

The Control Plane Results and Runs views expose the same projection with
case/trial drill-down. Users can see why a judgment passed or failed, including
each bounded explanation and contribution, without opening JSON. Downloaded
JSON reports use the versioned `EvalResultReportV2` envelope, which binds the
complete immutable source document to its canonical presentation. Those reports
remain valid inputs to `cayu eval report` and `cayu eval compare`; Cayu unwraps
and revalidates the binding instead of guessing the format. HTML renders the
same explainable facts for portable review.

`compare_eval_results(...)` pairs structured judgments and tool JSON
observations only when the immutable evaluation contract and exact
`(case_id, trial_number, assertion_id)` identity match. Structured comparisons
report criterion and aggregate deltas, evaluator recovery/regression, observed
usage/cost, explanation state, and exact unmatched identities. Tool JSON
comparisons separately report assertion-contract incompatibility, evidence-state
changes, safe observed-value changes, and outcome regressions. They do not
mistake a changed extra value for an assertion change. Cayu never pairs a
captured observation with a fresh trial heuristically and never diffs
incompatible contracts. The protected comparison API, Control Plane, CLI JSON,
and HTML comparison all consume this same schema-version-4 comparison document.

`run_eval_case(..., trials=N)` defaults to an all-trials-must-pass policy and
executes every trial with a fresh concrete session ID. Corpus and authored-suite
runs retain their exact `EvalSuiteTrialPolicyV1`; work-conserving scheduling may
fill the policy's bounded concurrency without changing deterministic trial
numbers or result order. `EvalCaseResult.trials` is an ordered tuple of
`EvalTrialResult` values; every trial retains its own status, session ID, final
output, assertion outcomes, exact-snapshot usage, assertion-specific cost
summary, duration, diagnostic, evidence-completeness flag, and optional
trajectory. Case and run aggregates are reproducible from those retained tuples.
There is no representative or implicit “last trial.” Published cases also
retain counts for passes, candidate failures, runtime errors, evaluator errors,
unavailable/cancelled trials, score range/mean, and an honest variability label.
When a structured AI judge runs on each trial, the distribution is labelled
end-to-end evaluation variability because candidate and judge variation cannot
be separated from those observations alone.

Assertion outcomes are `passed`, `failed`, `unavailable`, or `error`. Cases and
runs add `skipped` for a direct Python case with no assertions. Aggregate status
precedence is `error` → `unavailable` → `failed` → `skipped` → `passed`.
Unavailable and error results have `score = null`; aggregation never converts
them to zero or drops them from an average. A score is emitted only when every
contributing result is scored.

For fixed memory-intervention experiments, build a
`MemoryExperimentReportRequest` from exact intervention execution records,
published corpus results, execution profiles, and optional canonical
cost/overhead evidence, then call `build_memory_experiment_report(...)`. The
result preserves the complete baseline/candidate repetition matrix, including
missing and non-success outcomes, and calculates deltas only for comparable
paired rows. It reports per-pair, per-case, and per-experiment absolute and delta
distributions for latency, total tokens, memory-preparation duration, and memory
context tokens/bytes without adding them to ranking. Safety, privacy,
factual-support, hallucination, stale-memory,
false-memory, and unauthorized-memory gates run before declared task-quality
ranking. Each published assertion identity may populate exactly one declared
metric role, so a task-quality score cannot also stand in for independent
safety or evidence-gate authority. Canonical cost-quality evidence is scoped independently to each
candidate so a shared baseline is never counted twice in one aggregate. Safety
and evidence dimensions are gates, not ranking terms. Each baseline/candidate
accounting side uses `memory_experiment_accounting_task_id(experiment_id)` as its
shared paired cost-quality cohort identity, so the canonical aggregate covers all
cases and repetitions. Each side also uses
`memory_experiment_accounting_source_id(case_revision, repetition)` so row and
pair authority prevents evidence from moving between repetitions.
There is no universal aggregate score: the report recommends one fixed candidate
only when its complete evidence passes the configured gates and beats the
baseline under the declared deterministic ranking. Dispositions distinguish an
unavailable pair from an incomparable one, a baseline superseded by a selected
candidate, and an improving candidate that remained eligible but was not
selected.

Use `cayu eval memory-report REQUEST.json --format json|html` for local report
construction. A configured Control Plane exposes the equivalent protected
`POST /api/evals/memory-reports` and
`POST /api/evals/memory-reports/report.html` entrances. Those routes prove that
each supplied published result and its associated profile snapshot belongs to
the exact stored eval run. A result-less row retains its declared profile as
experiment-contract authority; it does not claim stored-run provenance. JSON is
the complete machine-readable contract. HTML renders a deterministic
human-readable summary of the primary identities, rows, pair classifications,
distributions, canonical accounting aggregates, gates, and recommendation
without launching another trial; consumers that require complete retained
authority use JSON.

The **Paired memory report** action on the Evals page exposes the same protected
boundary without application code. Select an exact schema-version-1 campaign
request JSON, review its experiment/case/variant/repetition dimensions, and
build the report against stored results. The browser shows the selected variant
and every disposition's comparable, incomparable, and unavailable pair counts,
then downloads the validated report JSON or server-rendered HTML. This action
validates and reports an already defined campaign; it does not silently invent
variants, interventions, reference truth, or missing trials.

### Credential-free causal-memory reference

The repository's `scripts/run_causal_memory_reference_campaign.py` composes
those existing contracts into one bounded reference. It loads the checked
standard Evals corpus, opens application-owned isolated knowledge overlays, and
submits each fixed `MemoryInterventionTrialRequest` to
`MemoryInterventionExecutor`. Cayu owns the snapshot verification, trial
identity, runtime execution, recall and context evidence, exact recovery,
portable publication, and paired report. The application owns the overlay
bytes, access scope, deterministic provider, evaluator mapping, and declared
outcome markers.

Recovery is trial-identity based: an ordinary rerun can recover completed work
and dispatch only the missing trials without changing how new provider requests
are audited. Terminal abnormal executions remain in the complete report matrix
with their exact execution availability even when portable assertion evidence
cannot be produced.

That is also the bounded integration seam for downstream systems such as Cayu
Compound: they may choose or rank already-declared candidates and then call
`run_causal_memory_reference_campaign(...)` for this reference fixture, or wire
their own application adapters to `MemoryInterventionExecutor` and
`build_memory_experiment_report(...)`. They do not bypass Cayu's execution and
evidence contracts, and Cayu does not take ownership of their search or ranking
policy.

The report language is deliberately narrow. A paired delta says that the
measured output changed under an exact declared memory intervention. It does not
claim that the model attended to a record, used it in hidden reasoning, or that
the result generalizes beyond the frozen corpus and candidate.

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
assertion evidence; corpus v4 does not silently infer replay input or an artifact
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
provider-specific boundaries cannot be represented exactly by corpus v4's single
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

The review sheet shows retained status, output, tool, structural, step, usage,
and cost evidence when each area was captured. Assertion quick-adds begin only
from complete observed facts; operators can
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
authoring even when runnable corpus-v2 conversion is unavailable. **Export eval
JSON** returns the same deterministic captured-only corpus without writing.

Runnable corpus-v2 conversion and scenario-v2 capture are independent
capabilities. Simple fresh invocations may satisfy
`build_promotion_candidate(...)`. Multi-stage sessions can instead produce an
ordered scenario preview from their retained initial, queued, resumed,
approval-checkpoint, and file-backed stimuli. Neither conversion controls
captured scoring or persistence: when exact source material is missing, the
captured evaluation remains usable and only the affected conversion reports why.

When that conversion is available, the same review sheet exposes **Run on current
app**. The default is one trial at concurrency one. Operators can contract the
published target's trial timeout and model-step ceiling, add run-scoped token,
tool-call, or elapsed-time limits, and—when the server owns a compatible
`PriceBook`—set an estimated-cost ceiling. The browser never supplies tools,
environments, credentials, pricing schedules, approval rules, or other execution
authority. It submits the reviewed expectation contract and bounded settings;
the server reconstructs runnable input from its current target baseline, scores
and saves the captured result, then admits the current-app run through the ordinary
durable worker.

Runtime and estimated-cost ceilings apply independently to each current-app trial;
they are not aggregate ceilings across the eval run. Trial count and concurrency
bound the run's aggregate scale. Generated project targets permit only one trial,
while an application-owned target that permits multiple trials must account for
that multiplication when choosing per-trial limits.

The current-app session uses the target's normal provider, tools, environment,
approval, and operator policy. Authenticated HTTP provenance and every requested
contraction are persisted with run admission, so a worker restart cannot silently
turn an operator launch into unattributed SDK work or recover with broader
limits. Before writing, the server resolves the target's normal provider/model
route and accepts a cost ceiling only when the current server-owned price book
has compatible pricing in the selected published currency. It then preflights
the complete effective request and enforces target ceilings again during
compilation and execution. Tool-effect metadata neither grants nor denies
authority.

Every fresh launch also carries the execution-profile revision shown by its
server preflight. HTTP can select that exact revision and narrow its published
ceilings; it cannot submit or alter provider, model, environment, fixture,
reset, effect, evidence, target request, bootstrap, or runtime identity.
Admission resolves the profile again and fails on a changed revision. The
durable run stores the full server-prepared runtime profile binding, and a
worker re-resolves and compares it before compilation or provider dispatch. A
deployment, registration, model route, environment, request base, bootstrap,
execution ceiling, evidence policy, or isolation change therefore produces a
visible readiness/compatibility conflict instead of silently running against a
different candidate. The operator refreshes the preview or catalog and makes a
new launch decision. Controlled scenarios repeat the same manifest and runtime
profile check before an approval, user-input, or manual-resume continuation can
dispatch more provider or tool work.

The profile carries an opaque target-material identity over the complete
request base, bootstrap messages, and `CorpusExecutionLimits` object used by
compilation. Public-safe material receives a restart-portable structural
SHA-256 identity. If any of that material crosses the application's configured
workload-secret boundary, Cayu publishes only a process-keyed HMAC and public
process-scope commitment. Raw target material is never returned. The same
private material remains stable within the serving process, while a restart
changes its scope and safely prevents old admitted work from running against an
unverifiable private configuration.

A scenario-selected registered environment is part of its exact profile. The
same environment is retained in durable launch identity, reconstructed by a
worker after restart, and used for the fresh session; it is never reduced to a
preflight-only label or replaced by the application's default.

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

Corpus v2 remains the supported assertion and execution contract. Scenario v2
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
provides the explicit corpus-v2 bridge for runnable cases, while captured-only
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
  routes for immutable persistence;
- `POST /api/evals/scenarios/artifacts/{requirement_id}/materialize` for the
  explicit idempotent fixture operation;
- `POST /api/evals/scenarios/{scenario_revision}/runs` to admit the exact saved
  revision and reviewed binding as durable work; and
- `POST /api/evals/runs/{run_id}/scenario-approval` to submit a fresh,
  actor-attributed approve or deny decision for one current trial checkpoint.

**Run scenario** executes the reviewed document through the target's ordinary
`CayuApp` runtime. Initial, queued, resumed, file, and structured inputs retain
their typed message shape; scenario approval decisions are never copied from a
source session. The worker records each trial's bounded cursor and phase on the
fenced eval claim. An approval pause, `ask_user` answer, or explicit session
resume survives coordinator and runtime-store restart and continues the same
durable session. All other interrupted stages begin a new attempt after claim
loss rather than being presented as exactly resumed.

The `manual_recovery` scenario-v2 wire value means an ordinary explicit
`CayuApp.resume(...)` interaction captured or authored by the operator. It does
not assert the outcome of an externally effectful tool whose result is unknown;
that runtime condition still requires the normal application-owned recovery
contract and is not portable scenario input.

Cancellation remains store-authoritative while a trial is running or awaiting
approval. Claim epochs fence progress and result publication, so a stale worker
cannot publish after ownership changes. This does not make provider billing or
external tool effects exactly once: work dispatched immediately before lease
loss may be repeated unless the provider or tool uses its own durable
idempotency/reconciliation contract.

Scenario runs publish the same `CorpusExecutionResult` shape as corpus-v2 runs.
Their JSON and HTML downloads therefore work with `cayu eval report` and
`cayu eval compare`, including the existing stable CI exit codes; no separate
scenario-only result format or execution engine exists.

### Durable eval catalog and run state

Use an `EvalStore` when promoted corpora, queued work, and published results
must survive beyond the promotion request. `SQLiteEvalStore` is restart-durable
for one embedded database; `PostgresEvalStore` supports shared multi-worker
claims; `InMemoryEvalStore` is intentionally process-local and is suitable for
tests and transient SDK workflows only. SQLite and PostgreSQL require storage
schema revision 50 for corpora and run state, and revision 53 for scenario
persistence. Session-backed production capture additionally requires revision
54. Controlled scenario progress and operator decisions require additive
revision 56. Typed queued messages, including file references used by a running
scenario, require breaking revision 57; stop older session workers before
migration because they would deliver only the text projection. Corpus and
scenario saves, run admission, and result publication require
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
provider, tool, environment, approval, and policy selection. Before publishing
the target catalog, Cayu resolves that runtime authority into a public execution
profile. The profile identifies the candidate agent, provider/model route,
environment, application release and manifest, evidence policy, runtime
execution fingerprint, exact target-material commitment, fixture/reset/effect
posture, and complete case, trial, concurrency, timeout, bootstrap, input,
compiled-input, model-step, and run-limit ceilings. It contains no credentials,
callback handles, raw bootstrap content, or private runtime objects. If current
runtime preparation fails, the target remains visible with an exact not-ready
state and cannot be launched. Catalog diagnostics use stable
`not_resolved`, `application_identity_changed`, or
`runtime_authority_unavailable` codes with bounded remediation copy, so clients
do not need to parse error prose.

Generated targets use the conservative profile automatically: no managed
fixture, a fresh Cayu session for each trial, ordinary application authority,
one trial, and concurrency one. The profile dimension is part of stable target
identity so applications can later publish deliberate fixture, isolation, or
authority changes without changing existing target keys. Catalog entries also
publish the server-enforced timeout and model-step ceilings plus the
target-compatible cost currencies, if any. A
non-empty currency list is the only condition that marks cost budgets available;
the browser cannot invent another currency and admission repeats compatibility
preflight against current pricing. An
explicit `EvalsConfig` remains the complete low-level contract
and takes precedence as one indivisible singleton registry; Cayu never merges its
target with the automatically assembled store. Arbitrary embedded
`create_server(...)` and `mount_cayu(...)` integrations continue to provide
trusted runtime objects explicitly.

A project may also declare one bounded default model judge without writing
Evals-specific Python:

```toml
[tool.cayu.evals]
price_book = "bundled-public"

[tool.cayu.evals.default_judge]
provider = "anthropic"
model = "claude-sonnet-4-6"
privacy_policy = "public-only"
allow_same_model = false
timeout_seconds = 120
max_input_tokens = 32768
max_output_tokens = 4096
max_total_tokens = 36864
max_estimated_cost = "0.1"
cost_currency = "USD"
```

The named provider must already be registered by the application. Cayu creates
a separate tool-free judge application, publishes only the declared route and
ceilings, and uses the provider's normal credential authority. It never infers
a judge from an API key or candidate route. `public-only` permits final output
and bounded public reference truth; `public-and-transcript` also permits the
retained transcript. Neither declaration permits private reference content.

`price_book = "bundled-public"` deliberately selects Cayu's packaged, dated
public-rate snapshot for generated candidate and judge cost enforcement; it is
never inferred. Applications with negotiated or gateway pricing must continue
to publish one complete application-owned `PriceBook` through server
configuration. A judge cost ceiling requires a configured project price book.

Author-first launch readiness can additionally accept a per-trial candidate
cost budget from Control Plane when the target publishes compatible pricing.
That value narrows only the reviewed launch, is retained in its accepted
exposure, and becomes a fail-closed observed-cost interruption threshold. As
with other runtime cost budgets, one provider completion can cross the
threshold before its usage is known, so the exposure remains labeled
`candidate_cost_not_hard_bounded` rather than misrepresenting it as a strict
pre-dispatch maximum.

When the declared route equals the candidate route, `allow_same_model = true`
is required to make that judge selectable for the candidate, and an operator
must still explicitly choose **Add same-model AI judge**. Cayu labels that
relation in the catalog, result, and report. Omit the declaration when no
project-wide judge decision has been made; deterministic evaluation and
retained-session promotion remain available.

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
from cayu import EvalExecutionCapacity, SQLiteEvalStore
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

The default explicit profile is also conservative and needs no additional
configuration. An application may enable repeated or concurrent trials only
when it owns a real reset boundary and publishes that promise explicitly:

```python
from cayu import EvalExecutionProfilePolicyV1
from cayu.server import EvalsConfig

evals = EvalsConfig(
    target=target,
    store=eval_store,
    execution_profile_policy=EvalExecutionProfilePolicyV1(
        fixture_strategy="application_managed",
        reset_strategy="application_managed",
        effect_posture="isolated_application_authority",
        # Change this content revision whenever the fixture/reset boundary changes.
        isolation_revision="sha256:" + resolved_isolation_digest,
        max_trials=5,
        max_concurrency=2,
    ),
)
```

`isolation_revision` is an application-supplied SHA-256 content identity, not a
secret and not a callback. Declaring application-managed behavior is a truthful
operational contract: Cayu does not create, reset, or validate an external test
tenant on the application's behalf. Any managed fixture, reset, or isolated
effect claim requires that revision. Repetition or concurrency additionally
requires an application-managed reset strategy, and the configured scale can
only narrow the attached target's own limits.

For a host-owned FastAPI application, pass the same complete wiring directly to the
mount. The `CorpusTarget.app` must be the exact `CayuApp` being mounted:

```python
from cayu import EvalExecutionCapacity, SQLiteEvalStore
from cayu.server import AuthenticatedAccess, EvalsConfig, mount_cayu
from cayu.storage.migrations import SchemaMode

eval_store = SQLiteEvalStore("cayu.db", schema_mode=SchemaMode.MIGRATE)
eval_capacity = EvalExecutionCapacity(max_active_trials=100)
mount_cayu(
    server,
    target.app,
    access=AuthenticatedAccess(dependency=require_operator),
    evals=EvalsConfig(
        target=target,
        store=eval_store,
        execution_capacity=eval_capacity,
    ),
)
```

An embedded mount must supply `access`; `OpenAccess` with Evals and an incomplete
`EvalsConfig` both fail construction. When `evals` is omitted, the dashboard keeps Evals
visible but marks execution and catalog operations not ready and names the missing target
or durable store wiring. An arbitrary embedded mount does not infer executable authority
from registered agents. `cayu serve --dev` is different: project assembly creates trusted
loopback access, a project-local durable store when needed, and server-published targets
for registered agents.

The target must reference the exact `CayuApp` attached to the server. Open
access, a disabled API, an in-memory store, incomplete wiring, or an unavailable
target identity rejects during construction and mounts no Evals execution
surface. `target`, `store`, and `execution_capacity` are excluded from
configuration serialization and diagnostics. `ServerSettings` does not deserialize application objects,
credentials, database handles, or executable targets from environment values;
applications resolve those trusted objects before constructing `EvalsConfig`.
`execution_capacity` is a process-local aggregate trial governor. Share one
`EvalExecutionCapacity` instance across every coordinator that should compete
for the same capacity domain. It defaults to 100 active trials, accepts larger
operator-selected values such as 10,000, and remains independent of each run's
`max_concurrency`. The per-run value bounds how much that run can request; the
shared capacity bounds how many trials cooperating runs may execute at once.
The default 100 is the N9 deployment target, not a power-of-two implementation
choice or a Runtime-wide maximum. Configure it from measured provider, session,
store, and memory capacity. Durable per-run `max_concurrency` fields have the
separate portable representation limit 2,147,483,647 so PostgreSQL, SQLite, and
browser clients reject an out-of-range value before storage.

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

Claim monitoring uses `EvalStore.load_run_observation(...)`, a bounded projection
of only run ID, status, attempt count, update time, and public ownership epoch and
lease expiry. Built-in in-memory, SQLite, and PostgreSQL stores implement that
projection without loading or parsing the immutable invocation, execution
profile, or scenario progress. Full `load_run(...)` validation remains in the
admission, explicit reconstruction, and publication paths where the complete
contract is required. The coordinator also uses
`heartbeat_run_observation(...)`; the existing full-record `heartbeat_run(...)`
API remains available to callers that need it. File-backed SQLite observations
use a dedicated read-only WAL connection, so status reads do not queue behind
that store instance's writer executor.

`EvalStore.wait_for_run_terminal(...)` is the application-facing bounded wait.
Each call is limited to at most 300 seconds, returns `None` immediately for an
unknown run, returns an `EvalRunObservation` for a terminal run, and raises
`TimeoutError` when its deadline expires. Its adaptive polling starts no faster
than 50 ms and backs off to a caller-selected ceiling of at most one second; the
default ceiling therefore bounds transition-observation latency to one second
plus one status-read operation. The coordinator applies the same exponential
backoff between heartbeats. A configured polling interval slower than one second
remains an explicit operator choice and is also the resulting observation-latency
bound. Callers load the immutable result once after observing `completed` instead
of repeatedly loading both the result and full run record.

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

One process may construct several `SQLiteEvalStore` instances over that embedded
database. Claim-heartbeat, trial-checkpoint, and terminal-result commits therefore
use an explicit `SQLiteEvalWriterContentionPolicy` instead of accidentally
inheriting the generic five-second SQLite timeout. The default gives each database
lock attempt 250 ms, retries with cancellation-aware exponential backoff from
10 ms through 250 ms, and stops after a cumulative 60 seconds. That cumulative
budget begins before a caller enters the store instance's writer queue, so local
queueing cannot multiply the bound. Applications may pass a narrower policy to
`SQLiteEvalStore(writer_contention_policy=...)`; every duration must be finite and
positive.

Each retry begins a fresh transaction. Claim ID, ownership epoch, lease expiry,
and run status are checked only after `BEGIN IMMEDIATE` acquires the writer lock,
so waiting never lets stale authority publish. Cancellation settles the current
lock attempt and rolls back before it is delivered. If the total contention
budget expires, the store raises `EvalStoreTransientContention`. The embedded
coordinator keeps a completed trial attached to its current execution and retries
that exact checkpoint after bounded, cancellation-aware pauses; it does not make
the run claimable while the completed result exists only in memory. A deferred
final-result publication may release the run for recovery because all of its trial
checkpoints are already durable. Neither case is marked `execution_failed`, while
a contended heartbeat keeps the current execution attached and retries. A competing claim cannot be published
while that SQLite writer lock is held; after the lock is acquired, the transaction
either renews the still-live exact claim or reports authoritative claim/lease loss.
Already committed trial checkpoints remain available, and a deferred final-result
publication can rebuild from them without rerunning those trials. Process loss
before a completed trial checkpoint commits still requires the target's normal
durable idempotency or reconciliation boundary, as described above; live contention
exhaustion alone does not manufacture that redispatch boundary.

Structured `cayu.storage.evals_sqlite` log records expose
`cayu_eval_store_event` values for `sqlite_writer.lock_wait`,
`sqlite_writer.retry`, `sqlite_writer.contention_exhausted`,
`sqlite_writer.claim_lost`, and `sqlite_writer.permanent_storage_failure`.
Records also carry the operation, bounded run ID, attempt count, accumulated
wait, and next retry delay. Other SQLite eval mutations retain the shared
five-second policy. Debug-level structured records separately identify
`run_status_read`, `full_run_rehydration`, `checkpoint_write`, and
`result_publication`; each includes elapsed duration, while worker-originated
checkpoint and publication records work consistently across store backends.

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

The same page also provides **New evaluation** for an author-first path that
does not depend on an imported corpus or captured production session. Its
bounded editor previews and saves immutable suite and scenario definitions,
preflights a full or explicit subset selection against current target
authority, and admits the selection through the existing durable workers. It
does not maintain a browser-only definition format or execution engine.

Launches use a cryptographically random `Idempotency-Key`. If a response is
ambiguous, retrying the unchanged launch reuses that key rather than duplicating
provider work. The dashboard follows queued, running, and cancelling records by
their durable run ID; cancellation is a server request, not a fabricated local
terminal state. Opaque catalog/run cursors and selected corpus, suite, run,
status, and comparison identities remain in bounded URL state. Superseded reads
are cancelled and a changed corpus never reuses another corpus's suite or case
projection.

The session-side **Run on current app** action uses the same retry registry and run
worker. It executes through the current server-published application target, not the
hermetic `[tool.cayu].eval_target`. Its request identity includes the captured candidate
revision and every execution setting, so changing a bound creates a new admission
identity while an ambiguous retry of unchanged work remains idempotent. Successful
launch opens the ordinary Evals run view; there is no dashboard-only execution engine.

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

Fresh authored-suite results retain both identities required for that decision.
The exact accepted-exposure revision remains the immutable preview, admission,
worker, and result identity. Comparison derives a separate compatibility revision
that excludes application-release and presentation metadata only; candidate and
runtime identity, target material, isolation, evidence, resource ceilings, selected
work, and pricing remain comparison-relevant and fail closed on drift.

The SDK, server, dashboard, JSON, HTML, and CLI share that same comparison
projection. `compare_eval_results(...)` accepts captured and fresh origins;
`eval_result_to_json(...)` and `render_eval_result_html(...)` report either
origin without reading an application or recomputing assertions.
`corpus_execution_comparison_to_json(...)` and
`render_corpus_execution_comparison_html(...)` render the shared comparison.
An incompatible comparison contains only typed mismatch reasons and result
summaries—never fabricated regressions.

### Browser acceptance profiles

Browser acceptance is a separate, versioned conformance layer above unit tests and
generic runtime Evals. The deterministic command owns its checked-in
`BrowserAcceptancePlanV1`, starts `BrowserAcceptanceFixtureV1`, and routes virtual
egress through that fixture. The repository command runs the pinned manifest through
the ordinary `CayuApp`, `WebBridge`,
`browser_session`, runner, virtual-egress, artifact, and durable-session surfaces:

```bash
python scripts/run_browser_acceptance.py \
  --mode deterministic --output-directory browser-acceptance-results
```

The command does not build or pull a container image, install Playwright, or download
Chromium. It requires Cayu's already prepared pinned browser workload. Missing
Docker, image, worker, or application prerequisites produce a content-free
`browser acceptance unavailable (...)` diagnostic and exit status 2. A complete
conforming report exits 0; failed or incomplete required evidence exits 1. Set
`CAYU_REQUIRE_STRONG_RUNTIME_BUILD_PROVENANCE=1` when the gate must reject an editable
source tree and require exact built-artifact provenance.

The command owns the deterministic fixture for the complete run and the scorecard
fails any relative-route case unless the fixture actually observed the route. The
deterministic profile is credential-free and uses only the two pinned local fixture
origins and Cayu's checked-in deterministic provider; no live model is called. It
covers navigation and redirect behavior; supported form and
action operations; delayed, replaced, hidden, detached, occluded, duplicate, stale,
scroll-dependent, frame, popup, truncation, artifact, denial, hostile-content,
capacity, cancellation, crash, acknowledgement-loss, replay, conflict, and cleanup
cases. History navigation, reload, upload, trace/video capture, multiple-page control,
and visual-only interaction are explicit `unsupported` rows in schema V1. Absence of
a required case, trial, semantic oracle, diagnostic, or terminal allocation
disposition can never be reported as a pass.

Redirect conformance binds the final browser-observed destination rather than only
the submitted navigation target. Stale-observation coverage retains a pre-action
revision across each admitted state-invalidating operation: click, fill, select,
press, wait, screenshot, and download. Missing, malformed, or contradictory browser
execution evidence is incomplete and cannot be inferred as a pre-dispatch intent.

Cancellation and process-loss rows are executable scenarios, not labels applied to
ordinary successful runs. Each row binds the injected lifecycle boundary, delivered
control signal, fresh-application recovery where applicable, and the number of browser
dispatches that crossed the boundary. Access observations use the closed
`available`/`blocked`/`unknown` vocabulary and the report aggregates each state; the
challenge-page oracle requires positive `blocked` evidence.
Browser-crash rows terminate the guest browser daemon before dispatch, during
execution, after an effect, and during cleanup while the Cayu owner remains alive;
they are distinct from the fresh-process recovery rows that terminate Cayu itself.

Public-web variability is invoked separately with the same model-facing
`browser_session` surface and strict allowlist, operation, concurrency, token, cost,
artifact, and wall-time ceilings:

```bash
python scripts/run_browser_acceptance.py myapp.browser_acceptance:build_live \
  --mode live_public --output-directory browser-acceptance-live-results
```

It records three immutable trials per case and reports variability instead of folding
an unavailable provider or site into deterministic conformance. Authenticated-site
acceptance is deliberately disabled in V1; selecting `live_authenticated` exits
unavailable before loading a target or resolving credentials.

Every trial is published to the output directory's `.trials` journal before the next
trial starts. Re-running an interrupted command with the same exact runtime and output
directory reconstructs completed trials and executes only rows that have no durable
dispatch intent. If a process disappears after that intent but before the terminal
receipt, recovery records the exact attempt as incomplete instead of dispatching it
again; another attempt requires an explicit retry. An intentional retry names both the
immutable source report and each exact case/trial pair:

```bash
python scripts/run_browser_acceptance.py myapp.browser_acceptance:build_live \
  --mode live_public --output-directory browser-acceptance-live-results \
  --resume-report browser-acceptance-live-results/sha256-REPORT.json \
  --retry live-iana-navigation:2
```

The retry creates another immutable report, retains the superseded attempt in
`prior_rows`, and never rewrites its trial receipt. If the retry receipt commits before
the command exits or loses acknowledgement, the next invocation reconstructs that
exact next attempt from the journal rather than dispatching it again. A journal entry
ahead of the source report's exact next attempt is a conflict. Live plans must supply an
exact `PriceBook` and currency selection. Every executable request must use one exact,
reserving app-wide budget authority whose currency, pricing, and cost ceiling equal the
manifest ceiling, so per-case budgets cannot partition the campaign cap. Journal-lock
waiting consumes the same campaign wall-time ceiling and expires without dispatching
when another owner does not release it. The scorecard binds the pricing fingerprint and
retains input, output, total-token, and estimated-cost evidence when available.

`BrowserAcceptanceReportV1` binds the case/corpus revisions to runtime build,
browser protocol and worker, Playwright/Chromium, runner/environment, workload,
egress, artifact store, execution profile, provider/model, platform, and execution
mode identities. Timestamps are evidence, not identity. Each immutable trial row
keeps expected versus observed state, separate fixture-semantic and agent-self-report
states, operations, artifacts, usage/cost/time, stable errors, truncation, and the
browser allocation disposition.

Portable JSON and standalone HTML contain bounded structural diagnostics only:
operation identities/states, observation revisions, counts and sizes, artifact
descriptors, and sanitized request summaries. They exclude raw page bodies, stderr,
cookies, browser storage/profile data, credentials, headers, query values, and raw
artifact bytes. Generic Evals remain the application behavior layer, existing
browser unit/process tests remain component evidence, the deterministic profile is
the release conformance gate, and the live-public profile is an opt-in variability
signal rather than a deterministic release gate.

Every later browser-capability slice must update the appropriate versioned manifest,
promote any newly supported expectation, and publish an exact-head acceptance report
before that capability is considered integrated. Component tests alone do not replace
the public application scorecard.

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
an actual OpenAI or Anthropic model. It adds an explicit project judge
declaration to `pyproject.toml`—but no Evals-specific Python—then runs `cayu
serve --dev`. In Control Plane it creates one source session, promotes and
baselines that retained result, launches one production-first current-app
trial, authors one same-model judged suite and trial, and verifies their
results, reports, comparisons, route labels, and stable CLI exits:

```bash
# Three candidate executions plus one bounded judge execution.
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

The separate judge-calibration live contract proves the evaluator authority
boundary with a real provider and a candidate sentinel that must receive zero
requests:

```bash
CAYU_PROVIDER=openai OPENAI_API_KEY=... \
  uv run python examples/evals_judge_calibration_live.py

CAYU_PROVIDER=anthropic ANTHROPIC_API_KEY=... \
  uv run python examples/evals_judge_calibration_live.py
```

It registers the selected provider only inside an explicit `ModelJudgeTarget`,
previews the immutable fixed evidence, executes one judge trial through the
protected HTTP contract, and fails if the candidate provider is called.
Nightly verification exposes it as `evals-judge-calibration-live`.

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

Saved `EvalRun` baselines use schema version `11`. Version 11 preserves the complete
ordered trial graph, explicit outcome/null-score contract, conclusive-evidence
state, the exact portable assertion revision behind each result, and the optional
portable execution contract a trusted executor fixes before dispatch. It retains
identity-free aggregate usage for every complete trial, canonical large counters,
durable-JSON validation, and the structural workspace/artifact assertion result
contract, including the exact memory-attribution observation used by a portable
memory assertion. A contracted run must retain exactly the requested number of
trials for every case.
`load_eval_run(...)` rejects missing versions and versions 1–10;
regenerate those baselines with the current Cayu version. No compatibility loader
or migration is used.

Standalone exports use a versioned document envelope. The current trajectory
schema version is `5`; `load_trajectory(...)` rejects files without that version
or with an unsupported version before validating the trajectory payload. This is an
intentional clean break from Cayu's earlier unversioned preview exports: they
are not migrated and must be regenerated. The trajectory schema version is
independent from `EvalRun.schema_version`. Version 5 retains immutable session
invocation provenance, typed memory attribution, and bounded structural workspace
and artifact probes in session-backed trajectories while preserving Cayu's
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

### Runtime-contract replay

Static assertion replay asks whether retained evidence satisfies assertions. It does not run
the application. **Runtime-contract replay** asks a different, narrower question: if recorded
model completions and tool results are treated as fixtures, does a candidate Cayu application
still construct the same model requests, authorize the same tool path, and reach the same
terminal outcome?

```python
from cayu import RuntimeReplayDisposition, RuntimeReplayRequest, trajectory_from_session

# Capture an exact runtime-attested trajectory from the durable source store.
trajectory = await trajectory_from_session(source_app, session_id)

# The candidate can be the current app or another app with the same public agent name.
report = await candidate_app.replay_session(
    RuntimeReplayRequest(trajectory=trajectory)
)
assert report.disposition is RuntimeReplayDisposition.MATCHED
```

The source run and its store are read-only. Replay creates a private in-memory session, sends
recorded assistant messages through a `ModelProvider` adapter, and substitutes every recorded
`ToolResult`, including artifact-bearing and error results, through the ordinary tool-round
machinery. Sequential and parallel calls in the recorded round are all matched by logical call
identity, tool name, exact model-authored arguments, effect, round linkage, and terminal result.
The candidate's context selection,
request construction, tool exposure, and tool policy still run. Registered provider dispatch,
registered tool implementations, environments, runners, workspaces, vaults, and external
effects do not. This includes tools declared `EXTERNAL`. Replay retains the source session's
resolved provider/model target rather than re-resolving a possibly different application
default.

Keyed request identities must have been enabled when the source ran, and the candidate must use
the same fingerprint key ID and secret. Missing source footprints report `unavailable`; a
different key reports `unavailable` because the HMACs are not comparable. The default report is
safe to serialize: it contains redacted execution-profile identities, keyed request identities,
hashed source/trajectory identities, typed reasons, and source event IDs, but no prompt text,
transcript content, tool arguments/results, metadata, provider error bodies, or credentials.
When contextual pricing participates in candidate budget admission, replay serves the exact
request/completion billing identities retained by source budget reservations. It never invokes
the provider's billing hook. A source without exact request billing evidence reports
`source_billing_evidence_unavailable` instead of guessing from completion-time accounting data.
Date-effective pricing is evaluated at the source session's creation time, keeping replay both
historically faithful and deterministic.

Completed-session evidence retains the resulting execution-profile identities, but it does not
retain every caller-owned invocation setting. If the source and candidate profiles differ in a
component that can be affected by unretained `RunRequest` settings—invocation loop policies,
thinking, request budget limits, run limits, retry policy, or `max_steps`—replay reports
`source_invocation_evidence_unavailable`. It never reconstructs those values from candidate
defaults and mislabels the resulting request/profile difference as candidate drift. Sources using
the candidate's matching defaults remain replayable.

Schema V1 is deliberately fail-closed and bounded. It accepts one completed fresh session with
exact runtime-owned text-only caller input, exactly two model steps, and one bounded tool round.
The default bound is 16 calls and callers may raise it to the hard limit of 256. Normal sequential
and parallel rounds retain exact finalized argument evidence only after every call's secret scope
has sealed. If argument publication is disabled, secret redaction changes an argument, or the
runtime cannot prove the final secret scope complete, replay returns
`source_tool_argument_evidence_unavailable`; it never executes with a redacted or guessed value.
Subagents,
forks, resumes, queued input, approvals, retries, compaction, hooks, custom loop/context/tool
policies, tool discovery, MCP/runtime/hosted tools, result projection, knowledge injection, and
registered environments report a typed `unavailable` result. A trajectory reloaded from a
standalone JSON export has no runtime-owned fresh-input attestation and therefore cannot be used
for runtime-contract replay; use that export for static assertion replay. Capture the replay
source directly from its durable session store instead.

These operations make different claims:

- `evaluate_assertions(...)` re-checks captured facts without running an application.
- `replay_session(...)` re-drives orchestration with recorded provider/tool outcomes and no live
  effects.
- A corpus run or **Run on current app** is a fresh evaluation and can call the configured live
  provider, tools, and environment under its eval limits.
- A session fork is a new live continuation from retained state, not replay.

A `matched` runtime replay is not model-behavior reproducibility. It does not claim that a model
would return the recorded completion today, that an external tool would return the recorded
result, or that two prompts are semantically equivalent beyond the versioned fingerprint
canonicalization contract.

## Interop

The default result format is JSON. It is intentionally simple so downstream
systems can consume it in CI or adapt it to external eval platforms.

Cayu should own the runtime-native view. External tools can own broader
experiment management, hosted dashboards, human review queues, and organization
level workflows.
