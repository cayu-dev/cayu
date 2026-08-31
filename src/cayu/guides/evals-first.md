# Run your first evaluation

Use this guide when you want to turn expected agent behavior into a reusable
regression suite. The ordinary path is entirely in Control Plane: the browser
authors bounded data, while the running Cayu application remains the only
source of provider, tool, environment, policy, and credential authority.

## Start the project

A conventional project created by `cayu new` already publishes one safe eval
target per registered agent. It also uses the project's normal SQLite or
PostgreSQL storage for suites, runs, results, and baselines.

For trusted local development:

```console
uv sync --extra dev
uv run --no-sync cayu check --fail-on warning --json
uv run --no-sync cayu serve --dev
```

Open `http://127.0.0.1:8000/cayu/evals`. `--dev` is loopback-only trusted-local
access. A deployed Control Plane needs its normal authentication and durable
store; do not expose `OpenAccess` publicly.

The Evals page names each operation that is ready and each missing dependency.
Fix a readiness diagnostic before running. The browser cannot create a missing
provider, credential, tool, environment, policy, or store.

## Create one deterministic suite

1. Choose **New evaluation**.
2. Give the suite and first case stable IDs and useful names. IDs identify the
   logical contract; editing content creates a new immutable revision.
3. Replace the sample input with a representative user request.
4. Keep the default completed-root expectation. Add **Final output contains**
   for one stable fact the answer must include. Prefer domain facts over exact
   prose when wording may vary.
5. Leave **Trials per case**, **Required passes**, and **Maximum concurrency**
   at `1` for the first run.
6. Choose **Check suite**. Review the canonical revision and every readiness
   diagnostic, then choose **Save revision**.
7. If the target publishes compatible pricing, optionally enter a **Maximum
   cost per trial**. Choose **Check launch** and review the maximum candidate
   work, the accepted interruption threshold, and whether cost is hard-bounded
   or observed-only. Choose **Run selected** only when that exposure is
   acceptable.

The result separates candidate status, deterministic assertions, runtime
health, evidence availability, usage, and observed or unavailable cost. Open a
case and trial before trusting the aggregate status.

## Grow and reuse the suite

- **Duplicate** copies the active case and its expectations under a new stable
  ID. Change only the stimulus or expected behavior you intend to vary.
- The checkboxes beside cases define an explicit launch subset. Select every
  case for a full-suite run; clear unrelated cases for a focused run.
- Editing a loaded suite never mutates the saved revision. Check and save again
  to create the next immutable revision.
- A repeated-trial policy retains every outcome. Required passes are not a way
  to hide evaluator errors or unavailable evidence; those still fail closed.

After a representative passing result, choose **Approve baseline**. Future
compatible results compare against that exact result revision. Cayu permits a
new application release to compare, but changed cases, assertions, evidence,
execution authority, judge semantics, trial policy, or applicable pricing may
make results explicitly incomparable.

Download JSON for automation and standalone HTML for review. The same files are
accepted by the package CLI:

```console
cayu eval report result.eval-result.json --html --output report.html
cayu eval compare baseline.eval-result.json result.eval-result.json \
  --json --output comparison.json
```

`cayu eval compare` exits `0` for a compatible comparison with no regression,
`1` for a compatible regression, and `2` when no conclusive decision is safe.

## Choose the next guide

- Use `cayu guide evals-ai-quality` when deterministic facts do not capture
  correctness, groundedness, helpfulness, or tone.
- Use `cayu guide evals-production` to evaluate a retained production session,
  a controlled multi-stage scenario, tool behavior, or memory use.
- Use the repository's `docs/evals.md` for complete SDK, schema, HTTP, storage,
  and embedding contracts.
