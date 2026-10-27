# Run your first evaluation

Use this guide when you want to turn expected agent behavior into a reusable
regression suite. The ordinary path is entirely in the Developer dashboard: the browser
authors bounded data, while the running Cayu application remains the only
source of provider, tool, environment, policy, and credential authority.

## Start the project

A conventional project created by `cayu new` already publishes one safe eval
target per registered specialist. It also uses the project's normal SQLite or
PostgreSQL storage for suites, runs, results, and baselines.

For trusted local development:

```console
uv sync --extra dev
uv run --no-sync cayu check --fail-on warning --json
uv run --no-sync cayu serve --dev
```

Open `http://127.0.0.1:8000/cayu/evals`. `--dev` is loopback-only trusted-local
access. A deployed Developer dashboard needs its normal authentication and durable
store; do not expose `OpenAccess` publicly.

Evals lists your suites and their latest runs. Open **Settings** for readiness,
execution profiles, judges, limits, corpus import and advanced tools.
Fix a readiness diagnostic before running. The browser cannot create a missing
provider, credential, tool, environment, policy, or store.

## Create one deterministic suite

1. Open **Evals → New suite** and select the specialist.
2. Name the suite and first case. **Suite details** and **Case details** contain
   their stable IDs and optional descriptions. Editing content creates a new
   immutable revision; the saved suite ID stays fixed.
3. In the case’s **Input** tab, replace the sample with a representative user request.
4. Open **Checks** and keep the completed-session check. Choose **Final output
   contains** in **Check type**, then **Add check**. Expand its sentence to edit
   the expected fact. Prefer domain facts over exact prose when wording may vary.
   **Final output equals** also offers exact text or JSON-value comparison.
5. Expand **Suite settings**. Leave **Trials per case**, **Required passes**,
   and **Maximum concurrency** at `1` for the first run.
6. Choose **Check suite**. Review the canonical revision and every readiness
   diagnostic, then choose **Save revision**.
7. Choose **Run saved revision**. If the specialist publishes compatible pricing,
   optionally enter a **Maximum cost per trial**. Choose **Check launch** and review the maximum candidate
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
- Open a suite and choose **Edit suite**, or edit a specific case. Check and save
  to create the next immutable revision. The old revision stays unchanged.
  Captured cases retain their protected source-session links only when their
  case ID and content remain unchanged.
- Saving does not run the specialist. If the save response is lost, use **Retry
  exact save** to confirm the original revision before editing again.
- A repeated-trial policy retains every outcome. Required passes are not a way
  to hide evaluator errors or unavailable evidence; those still fail closed.

After a representative passing result, choose **Approve as baseline** on its run
page. For authored suites, approval applies to the exact execution group; there is no
aggregate baseline across different case selections. Future
compatible results compare against that exact result revision. Cayu permits a
new application release to compare, but changed cases, assertions, evidence,
execution authority, judge semantics, trial policy, or applicable pricing may
make results explicitly incomparable.

The run’s **Details** disclosure retains lifecycle, all six result dimensions,
manifest, evidence, pricing and trial policies, revisions and raw JSON. Download
JSON for automation and standalone HTML for review. The same files are
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
