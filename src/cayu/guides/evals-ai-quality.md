# Evaluate semantic quality with an AI judge

Use an AI judge for behavior such as correctness, groundedness, usefulness, or
tone that cannot be reduced to a stable runtime fact. Keep deterministic checks
for completion, required tools, approvals, artifacts, budgets, and evidence.
An AI judgment complements those facts; it does not replace them.

## Declare judge authority

A generated project never infers judge authority from an API key or from the
candidate's route. Declare one provider, model, privacy policy, and same-model
decision in `pyproject.toml`:

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

The provider must already be registered by the application and authenticated
through its normal credential path. `public-only` permits final output and
bounded public reference truth. Use `public-and-transcript` only after deciding
that the judge may receive the retained transcript. Neither option permits
private reference content; specialized private references remain explicit
application-owned server configuration.

The explicit `bundled-public` selection uses Cayu's packaged, dated public-rate
snapshot for candidate and judge budgets. It is convenient for public API
pricing, but it is not a substitute for an application-owned book when your
gateway, region, or negotiated rates differ. Cayu does not infer or merge
pricing. The judge's cost ceiling requires the selected book and appears in the
published profile.

If the declared judge provider and model equal the candidate route,
`allow_same_model = true` is also required to make that judge selectable for
the candidate. The user must still deliberately choose **Add same-model AI
judge** in Control Plane. The result and report retain that label. An
independent model is usually a stronger quality signal, while same-model
judging can be useful for an inexpensive first pass.

Restart `cayu serve` after changing the declaration. Evals should show
**Project default judge** with the exact public provider/model, privacy,
token/time ceilings, and candidate-route relation. A missing or mismatched
provider fails configuration instead of silently choosing another route.

## Author a rubric

Open **Evals → New evaluation**, select the case, and choose **Add AI judge** or
**Add same-model AI judge**.

1. Give the assertion and rubric stable IDs.
2. Define one to eight criteria. Each criterion needs a stable ID, a short name,
   a concrete description of good behavior, and a canonical weight. The weights
   must sum exactly to `1`.
3. Set the pass threshold between `0` and `1`. Cayu computes the weighted total;
   the model cannot choose the aggregation rule.
4. Choose **Final output only** unless the criterion genuinely needs retained
   transcript evidence and the declared privacy policy permits it.
5. Add a public expected answer or expected facts when the task has trusted
   reference truth. Reference material is sent only to the judge, never to the
   candidate.

Write criteria that a reviewer could score consistently. Separate factual
correctness, groundedness, and task completion rather than asking for one vague
"quality" score. Do not place secrets, credentials, private customer data, or
instructions for the candidate in a public reference.

## Calibrate before gating

Choose **Check suite** so the server compiles the exact rubric, reference,
profile, and evidence selection. The calibration panel then uses that reviewed
assertion.

Enter a fixed candidate output, the task it answered, and human scores for each
criterion. **Check calibration** shows the exact judge work and cost exposure;
**Run calibration** may repeat judge calls over the same retained evidence. It
does not rerun candidate tools or sessions.

Use clearly good, clearly bad, and difficult boundary examples. If repeated
judge scores move materially on identical evidence, refine the rubric or use a
different judge before treating its threshold as a release gate.

## Interpret a judged result

Every trial retains:

- each criterion score and bounded, redacted explanation state;
- Cayu's exact weighted contribution and aggregate;
- the rubric, reference, judge profile, and implementation revisions;
- candidate and judge route relation;
- observed judge usage and priced or unavailable cost; and
- evaluator errors separately from candidate behavior.

A timeout, provider failure, malformed typed response, missing required usage,
or unavailable reference produces an evaluator error with no candidate score.
It is never converted to numeric zero or reported as a candidate failure.

Repeated candidate runs with judging measure end-to-end evaluation variability.
Only fixed-evidence calibration isolates judge variability. Keep those claims
separate in reports and release decisions.

Use `cayu guide evals-first` for suite and baseline basics and `cayu guide
evals-production` for production-session, scenario, tool, and memory workflows.
