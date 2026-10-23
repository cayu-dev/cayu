/**
 * Proposed (backend item 14): suite-centred reads for the Evals pages. Today the server
 * has the parts (targets, corpora, authored suites, runs, results, baselines, comparisons;
 * see `src/lib/api.ts`) but no read that answers "how is each suite doing against its
 * baseline" in one call, and CLI runs never reach the eval store.
 */

/** A run's outcome against the suite's approved baseline, case by case. */
export type EvalVerdict = "improvement" | "regression" | "tradeoff" | "tie"

export type EvalRunStatus = "queued" | "running" | "completed" | "failed" | "cancelled"

/** Where a run came from: the dashboard, CI (`cayu eval run` reporting in), or a session. */
export type EvalRunOrigin =
  | { kind: "dashboard"; by: string }
  | { kind: "ci"; branch: string; commit: string; pr: number | null }
  | { kind: "session"; session_id: string }

export type EvalRunSummary = {
  id: string
  suite_id: string
  status: EvalRunStatus
  origin: EvalRunOrigin
  started_at: string
  finished_at: string | null
  /** Cases counted once trials settle; a flaky case passed its minimum but not every trial. */
  cases: { total: number; passed: number; failed: number; flaky: number }
  /** Mean model-judge score from 0 to 1, when the suite has judges. */
  score: number | null
  /** Against the baseline; null for the baseline itself or when there isn't one. */
  verdict: EvalVerdict | null
  change: { improved: number; regressed: number } | null
  usage: { tokens: number; cost: number | null }
  is_baseline: boolean
}

export type EvalSuiteSummary = {
  id: string
  name: string
  description: string | null
  specialist: string
  /** Written in the dashboard, imported as a corpus, or captured from sessions. */
  source: "authored" | "imported" | "captured"
  case_count: number
  trials: { per_case: number; min_passes: number }
  baseline: {
    run_id: string
    approved_by: string
    approved_at: string
    cases: EvalRunSummary["cases"]
  } | null
  last_run: EvalRunSummary | null
}

/** One check on a case, in words: "Final output contains “reconciled”". */
export type EvalCheck = {
  kind: string
  label: string
  /** Model judges score against a rubric instead of passing or failing outright. */
  judge: boolean
}

export type EvalCase = {
  id: string
  name: string
  input: string
  checks: EvalCheck[]
  /** The session it was captured from, if any. */
  source_session_id: string | null
}

export type EvalCheckResult = EvalCheck & {
  passed: boolean
  /** What was found, or why it failed. */
  detail: string | null
  /** For judges: score from 0 to 1 and the judge's reason. */
  score: number | null
  reason: string | null
}

export type EvalTrial = {
  n: number
  passed: boolean
  session_id: string | null
  output: string
  checks: EvalCheckResult[]
  usage: { tokens: number; cost: number | null }
  seconds: number
}

export type EvalCaseOutcome = "passed" | "failed" | "flaky" | "error"

export type EvalCaseResult = {
  case: EvalCase
  outcome: EvalCaseOutcome
  trials: EvalTrial[]
  score: number | null
  /** The same case in the run it's compared with. */
  compared: { outcome: EvalCaseOutcome; trials_passed: number; output: string } | null
  change: "improved" | "regressed" | "unchanged" | "new"
}

export type EvalRunDetail = EvalRunSummary & {
  suite: { id: string; name: string; specialist: string }
  /** The run these results are compared with: the baseline unless another was picked. */
  compared_with: EvalRunSummary | null
  candidate: { provider: string | null; model: string; environment: string | null }
  results: EvalCaseResult[]
}

export type EvalSuiteDetail = EvalSuiteSummary & {
  cases: (EvalCase & {
    last: { outcome: EvalCaseOutcome; trials_passed: number; trials: number } | null
  })[]
  runs: EvalRunSummary[]
}

/** What a fresh trial runs with, and which eval features this deployment has. */
export type EvalSettings = {
  candidate: {
    specialist: string
    provider: string | null
    model: string
    environment: string | null
  }[]
  judge: { provider: string; model: string; same_model_allowed: boolean } | null
  limits: {
    max_trials: number
    max_concurrency: number
    timeout_seconds: number
    max_steps: number
  }
  /** Readiness from the server contract, in words; only what isn't ready needs a look. */
  features: { name: string; ready: boolean; note: string | null }[]
}

export type NewEvalSuite = {
  name: string
  specialist: string
  trials: { per_case: number; min_passes: number }
  cases: { name: string; input: string; expect_contains: string }[]
}

/** Score a session as it is (captured evaluation), or add it as a case that reruns. */
export type AddToEvals = {
  session_id: string
  mode: "score" | "rerun"
  /** An existing suite, or a new one by name. */
  suite: { id: string } | { name: string }
}
