# Evaluate production sessions, scenarios, and memory

Use this guide when a completed or failed retained session is the behavior you
want to understand. Cayu can score the evidence already captured, save it as a
baseline, and—when the session can be reconstructed under current published
authority—run a controlled fresh scenario through the same evaluator.

## Evaluate a retained session

Open the session in Control Plane and choose **Evaluate**.

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
reconstructed safely. It uses the current registered agent, provider, tools,
environment, policies, and limits. A readiness message that names a missing
tool, environment, secret, fixture, approval path, or evidence field is a real
current dependency—not a request to fabricate it in the browser.

## Build a controlled multi-stage scenario

Use **Production scenario** from a captured session, or choose **Multi-stage
scenario** for a case in **New evaluation**.

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

Choose **Check readiness** before saving. The preview binds the exact current
target, execution profile, scenario revision, environment, fixtures, secrets,
effects, evidence policy, limits, trials, concurrency, timeout, and cost bound.
Prepare retained artifacts only through the displayed server-owned **Prepare
fixture** action. **Run scenario** admits that exact reviewed binding.

During execution, the Runs view shows queued, running, awaiting-input,
awaiting-approval, cancelling, and terminal trial phases. Approve or deny only
the displayed fresh request. Cancellation is a durable server request, not a
browser-only status change.

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

Session-scoped memory aliases are diagnostic labels, not stable case identity
across fresh runs. Results retain bounded source/lifecycle evidence and safe
immutable references, never raw private memory identity or content.

## Compare and automate

Captured, fresh simple, and scenario results share the same result, baseline,
report, and comparison contracts. Download JSON/HTML from Control Plane or use:

```console
cayu eval report result.eval-result.json --html --output report.html
cayu eval compare baseline.eval-result.json result.eval-result.json \
  --json --output comparison.json
```

Read incompatibility reasons instead of overriding them. Changed assertions,
scenario material, memory evidence policy, execution/judge identity, trial
policy, or pricing may make two apparently similar runs unsafe to compare.

Use `cayu guide evals-first` for the smallest author-first workflow and `cayu
guide evals-ai-quality` for rubric, reference, calibration, and judge semantics.
