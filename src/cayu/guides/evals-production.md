# Evaluate production sessions, scenarios, and memory

Use this guide when a completed or failed retained session is the behavior you
want to understand. Cayu can score the evidence already captured, save it as a
baseline, and—when the session can be reconstructed under current published
authority—run a controlled fresh scenario through the same evaluator.

## Check a native process evaluation

For a run launched with `cayu eval run --processes ...`, use its recorded
`--process-directory`:

```bash
cayu eval status .cayu/evals/run-001 --sessions
cayu eval failures .cayu/evals/run-001 --json
cayu eval export .cayu/evals/run-001 --output run-001-receipts.zip
```

These commands inspect retained receipts and explicit read-only session links
without loading the target. Recorded results, provisional trial progress, and
recent session activity are separate facts. An admitted launch or a persisted
`running` session does not prove a live owner. Missing or truncated evidence is
reported explicitly; export never resumes work or promotes an incomplete run.
Use `--case CASE_ID` to focus failure inspection. The ZIP contains native
receipts and worker results; preserve application session stores separately.

## Evaluate a retained session

Open the finished session's action menu in the Developer dashboard and choose
**Add to evals**. Select an existing authored or corpus suite, or create a new one.
**Add as a case** saves runnable input in a new immutable suite revision;
**Score this session** records one captured case without changing an existing
suite's runnable membership. Review the exact addition before saving. A saved
request can be retried after a lost response without creating a second operation.

From a terminal or a coding agent, `cayu eval add-session` makes the same addition through
the same server API; add `--application NAME` for a Cayu Cloud Agent's session:

```console
cayu eval suites list
cayu eval add-session SESSION_ID --suite SUITE --mode case --preview
```

`--mode score` scores the session instead, and `--new-suite ID` creates a suite. The
preview prints the exact plan and `commands.publish`, the command that saves it; running
that command again after a lost response returns the original receipt, and a plan the
session or suite no longer matches is refused (`eval_addition_changed`) rather than saved.

For the detailed capture, export and fresh-run tools, expand **Captured evidence,
export and fresh execution**, then choose **Open captured evaluation details**.

1. Review **Captured evidence** before adding expectations. Missing, redacted,
   truncated, or unavailable evidence is not a failed assertion and must not be
   replaced with a guessed value.
2. Give the suite and case durable names. Use **Add observed** only for behavior
   that should remain true; incidental wording or a mistaken tool call does not
   become correct merely because it happened once.
3. Choose the assertion kind that matches the claim: final output, tool called,
   bounded tool arguments/result, process event/order, child status,
   workspace/artifact structure, usage/cost, or memory attribution.
4. Choose **Preview score** after every edit. Preview rescans the retained
   evidence with the same compiler and evaluator used for fresh runs.
5. **Save evaluation** records the exact captured result. **Approve baseline**
   makes that immutable revision the default comparison only after human
   review.

**Run on current app** is available only when the captured input can be
reconstructed safely. It uses the current registered specialist, provider, tools,
environment, policies, and limits. A readiness message that names a missing
tool, environment, secret, fixture, approval path, or evidence field is a real
current dependency—not a request to fabricate it in the browser.

If the launch response is lost, **Retry exact launch** sends the original evidence,
execution profile, limits, budget and operation key. Catalog updates do not turn
that retry into a different launch. The private request remains in this open
workbench; it is not stored in the URL or browser storage. Closing, navigating
away or previewing different evidence requires explicitly abandoning its recovery
once the request is no longer pending. Abandoning is not cancellation, and the
original work may already be running. Reloading cannot restore private evidence.

From a terminal or a coding agent, `cayu eval capture` does the same through the same server
API; add `--application NAME` for a Cayu Cloud Agent's session:

```console
cayu eval capture preview SESSION_ID
cayu eval capture save SESSION_ID --expected-candidate-revision REV
cayu eval baseline approve --result RESULT_REVISION
cayu eval capture run SESSION_ID --expected-candidate-revision REV
```

`capture preview` is **Preview score**: it returns the retained `evidence` (in `candidate`),
the suite and case the capture proposes, their `captured_score` over that evidence, and
whether the session can run afresh (`runnable_conversion`). `--suite-id`, `--suite-name`,
`--case-id`, `--case-name` and `--assertion JSON` (repeatable, in the corpus assertion format,
in place of the proposed checks) edit them as the details do. The preview prints
`commands.save` and, for a runnable session, `commands.run`, both for that exact candidate.
`capture save` is **Save evaluation** and prints `commands.baseline`, which is **Approve
baseline**. `capture run` is **Run on current app**, with `--trials`, `--max-concurrency`,
`--trial-timeout-seconds`, `--max-steps`, the per-trial limits and `--max-cost`; after a lost
response its `commands.continue` repeats that launch and answers with the run it admitted. A
deploy changes the captured evaluation, so after one the continuation lists the runs instead
of repeating the launch.

## Build a controlled multi-stage scenario

Use **Production scenario** from a captured session, or choose **Multi-stage
scenario** on a case’s **Input** tab in **New suite** or **Edit suite**.
**Evals → Settings → Production scenarios** provides the same editor, a paginated
library of saved revisions, **New scenario**, and exact-revision downloads.
Select a saved revision to edit it or run it on its own. Save or discard edits
before selecting another scenario.

- The initial event creates one fresh session.
- **Queue input** delivers ordered follow-up input on the next turn or when the
  session is idle.
- **Resume input** supplies a typed user-input answer or an ordinary session
  resume at the authored checkpoint.
- **Approval** requires a new decision for the named current tool occurrence.
  A production approval is never replayed into the fresh trial.
- **Add part** can add structured JSON. A file part must refer to a retained,
  digest-bound artifact requirement; a browser path or upload cannot create
  server-side file authority.

In a suite draft from `cayu eval suites draft`, a case runs a saved scenario with
`"stimulus": {"kind": "scenario", "scenario_id": "ID", "scenario_revision": "REVISION"}`.

Choose **Check readiness** before saving. The preview binds the exact current
target, execution profile, scenario revision, environment, fixtures, secrets,
effects, evidence policy, limits, trials, concurrency, timeout, and cost bound.
Prepare retained artifacts only through the displayed server-owned **Prepare
fixture** action. **Run scenario** admits that exact reviewed binding. If its
response is lost, retry **Run scenario** here before switching tools; the original
request identity is retained. If recovery remains unavailable, **Abandon scenario
recovery** releases only this tab's recovery handle after confirmation. It keeps
your draft and settings and requires a fresh readiness check before a new run.
It does not cancel or erase accepted server work; a new run can duplicate that
work. Abandonment is unavailable while a request is in flight. Closing the
browser does not cancel server work.

From a terminal or a coding agent, `cayu eval scenario` checks, saves and runs a scenario
through the same server API; add `--application NAME` for a Cayu Cloud Agent:

```console
cayu eval scenario check --session SESSION_ID
cayu eval scenario run --scenario REVISION
```

`--session` takes the scenario a finished session's capture proposes, `--file FILE` a
scenario JSON file, and `--scenario REVISION` a saved revision from `cayu eval scenario
list` (`cayu eval scenario show REVISION` prints it). The launch settings are options such as
`--environment`, `--trials` and `--max-cost`. `check` runs nothing; `--save` saves the
checked revision. A ready check prints `commands.run`, which runs that revision under an
operation ID and the binding it reviewed. `run` saves the revision and admits the run, and
its `commands.continue` repeats that admission after a lost response. Repeating either
command answers with the run it admitted, even after a deploy, without starting a second
one.

A check that names a file requirement with `artifact_binding_required` needs **Prepare
fixture**. From a terminal, name the requirement and the same scenario source:

```console
cayu eval scenario prepare-fixture REQUIREMENT_ID --session SESSION_ID
```

It copies the retained file into a reusable fixture of the scenario's environment (from the
requirement's own reference, or `--artifact REQUIREMENT_ID=ARTIFACT_ID`), saves the scenario
revision that uses it, and, once that revision is ready, prints `commands.run`. Preparing
again changes nothing.

During execution, the run page shows queued, running, awaiting-input,
awaiting-approval, cancelling, and terminal trial phases. Approve or deny only
the displayed fresh request. Cancellation is a durable server request, not a
browser-only status change. From a terminal, `cayu eval runs show RUN_ID` lists each trial's
phase, and for a trial awaiting approval the tool and the checkpoint (`pending_event_id`)
it waits on:

```console
cayu eval runs show RUN_ID
cayu eval scenario approve RUN_ID --event CHECKPOINT_ID
cayu eval scenario deny RUN_ID --event CHECKPOINT_ID --reason "Not the reviewed action."
cayu eval runs cancel RUN_ID
```

`approve` and `deny` decide the checkpoint the trial awaits (`--trial N` when several do).
`--event` decides only while the trial still awaits that checkpoint, and a checkpoint takes
one decision. Every trial of a run has the same checkpoints, so after a lost answer run the
error's `commands.continue`: it names the trial, its session and the checkpoint, and records
nothing new if the decision was recorded, where the bare command would decide whichever
trial awaits approval by then.

## Check tool and process behavior

Prefer the narrowest truthful assertion:

- **Tool arguments contain** and **Tool result contains** use bounded JSON
  subsets. Result checks are available only for explicitly retained public-safe
  fields or text.
- Process assertions use Cayu's closed event vocabulary. Use exact order only
  when intervening events should make the case fail.
- Workspace and artifact checks are structural by default: declared path or
  filename, media type, digest, and size. Raw workspace content is not portable
  evidence.
- Usage and cost checks depend on complete runtime evidence and an applicable
  price book. Unpriced cost remains unavailable rather than zero.

## Evaluate memory without overstating causality

Memory evaluation has three different questions:

1. **Structural exposure:** did reviewed items enter the candidate context, and
   were they exposed to the provider? Add **Require memory exposure** and set
   admitted-item and provider-exposure bounds. This needs complete attribution
   evidence for the selected source lifecycle.
2. **Semantic use:** did the answer apply the relevant remembered facts
   correctly? Add **Reference-backed judge**, replace the blank expected fact
   with trusted evaluator-only truth, and calibrate the rubric. This judges the
   output; it does not reveal private memory content to the candidate.
3. **Causal contribution:** did memory cause an improvement? An ordinary run
   cannot answer that. Use the protected paired-intervention report only for a
   validated campaign that controls the memory variant and retains comparable
   trial evidence.

**Evals → Settings → Memory reports** accepts the campaign request JSON (up to
32 MiB). **Build report** validates it against stored results and shows paired
comparisons, availability and variant dispositions. Download **JSON** or **HTML**,
or **Clear report** before choosing another request. This tool builds reports;
it does not launch a memory campaign or fill in missing trial evidence.

**Evals → Settings → Import / export** imports a portable corpus JSON file (up to
8 MiB) and lists saved corpora for the selected specialist. **Download** exports
the exact stored revision, without running trials.

From a terminal, the same server API imports, lists and downloads corpora; add `--application
NAME` for a Cayu Cloud Agent. `cayu eval memory-report` builds the memory report from the same
request without a server, so it doesn't check the request against stored results:

```console
cayu eval corpora import evals.json
cayu eval corpora list
cayu eval corpora download REVISION --json-output evals.json
cayu eval memory-report request.json --format html --output memory-report.html
```

Session-scoped memory aliases are diagnostic labels, not stable case identity
across fresh runs. Results retain bounded source/lifecycle evidence and safe
immutable references, never raw private memory identity or content.

## Objectives for Compass

Compass on Cayu Cloud watches how an Agent runs and proposes changes, and it
judges every finding by the objectives the Agent's Release declares. Cloud
refuses to enroll an Agent whose Release declares none (`no_objectives`).
Declare them in `pyproject.toml`, next to the evals that check a change:

```toml
[tool.cayu.improvement]
objectives = "cayu-objectives.toml"
eval_targets = ["evals.agent:build_eval"]
```

`objectives` is a project-relative path; keep it outside anything a change may
edit. Name at least one of `eval_targets` (the `module:attribute` form of
`[tool.cayu].eval_target`), `corpora` (portable corpus files) or
`benchmark_packages`; the missions Compass makes ready evaluate on
`eval_targets`. `evaluator_only_paths` lists material a candidate must never
see, such as reference answers.

A minimal objectives file, with the optional parts commented out:

```toml
schema_version = 1

# Required. The share of decisions granted autonomy that a person still
# reviews, in basis points from 0 to 10000 (500 = 5%).
sampled_review_rate_bps = 500

# Required, at least one. Suites whose compared results must not regress,
# each named once by the id its eval or corpus gives it.
[[correctness_floor]]
suite_id = "agent-output"
# metrics = ["status", "score", "reliability"]  # the default: all three

# Tools, or one approval gate on a tool, that are never removed or granted
# autonomy. Without approval_gate an entry covers every gate on the tool.
# [[mandatory_controls]]
# tool_name = "issue_refund"
# approval_gate = "refund-over-limit"

# Goals per verified completion. Recorded, not enforced.
# [targets]
# cost_per_verified_completion = { amount = "0.25", currency = "USD" }
# seconds_per_verified_completion = 90
# human_attention_requests_per_verified_completion = "0.2"

# What a change may touch. Nothing is mutable unless it is listed here.
[mutable_surfaces]
body_paths = ["prompts"]
# execution_profile = true  # may change the execution profile
# model_per_step = true     # may change the model each step uses

# Who approves each kind of change. Schema version 1 accepts only "person",
# which is the default, so the table can be left out.
# [approval_rules]
# body_paths = "person"
# execution_profile = "person"
# model_per_step = "person"
# autonomy_grant = "person"
```

Body paths must exist and must not overlap the protected paths:
`pyproject.toml`, the objectives file, the declared corpora, benchmark packages
and evaluator-only paths, and each eval target's module. `cayu new` writes this
file and table for presets with evals. Check the declaration before deploying:

```console
cayu objectives validate
cayu cloud deploy .
cayu cloud compass enroll my-agent --agent my-agent
cayu cloud compass verify my-agent --wait
cayu cloud compass findings list my-agent
```

`--agent` names a Cayu agent inside the Agent that Compass watches, and the
Agent must register it. Repeat it to watch several; each `enroll` sets the
whole list, so an agent left out is no longer watched. `cayu cloud compass
--help` lists the rest: `show`, `disable`, `pass`, `job`, `missions`, and
`findings show|accept|dismiss`. The full schema and the Python API are in the
repository's `docs/evals.md`.

## Compare and automate

Captured, fresh simple, and scenario results share the same result, baseline,
report, and comparison contracts. Download JSON/HTML from the Developer dashboard, or from a
terminal with `cayu eval runs report`, and use the files:

```console
cayu eval runs report RUN_ID --json-output result.eval-result.json --html-output report.html
cayu eval report result.eval-result.json --html --output report.html
cayu eval compare baseline.eval-result.json result.eval-result.json \
  --json --output comparison.json
```

Read incompatibility reasons instead of overriding them. Changed assertions,
scenario material, memory evidence policy, execution/judge identity, trial
policy, or pricing may make two apparently similar runs unsafe to compare.

Use `cayu guide evals-first` for the smallest author-first workflow and `cayu
guide evals-ai-quality` for rubric, reference, calibration, and judge semantics.

## Show CI results in the Developer dashboard

Opt in when running an ordinary direct, corpus, workflow or multiprocess eval:

```console
cayu eval run --output result.json \
  --report-to https://dashboard.example/api \
  --report-target YOUR_REGISTERED_TARGET_KEY --report-file report.json
```

Use a dedicated reporting credential in `CAYU_EVAL_REPORT_TOKEN`. The server’s
`CAYU_EVAL_REPORTERS` digest grants bind it to a stable publisher and explicit
targets; operator credentials cannot publish reports. Remote URLs require HTTPS.
For a local server, explicitly enable `--report-allow-loopback-http`.

Reported series show the publisher and submitted branch, tested commit and PR.
They are read-only external evidence: an unknown declared suite revision stays
unknown, and reported runs cannot become native baselines or comparisons.
`--report-associate-corpus` only accepts an exactly matching already-stored corpus.
Benchmark campaign reporting is rejected before execution.

The normal result remains available when reporting fails; the command exits 2.
Retain the credential-free `report.json` as a CI artifact, then use
`cayu eval publish report.json` to retry the exact publication without rerunning
the workload. GitHub metadata defaults come from environment/event fields; use
`--report-commit` when checkout tests a different SHA from `GITHUB_SHA`.
See `docs/recipes/ci-eval-reporting.md` for grants, metadata overrides, rotation
and the artifact-retention workflow.
