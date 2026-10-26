/** How eval runs read: pass counts, verdicts, where a run came from. */
import type { EvalCaseOutcome, EvalRunOrigin, EvalRunSummary, EvalVerdict } from "../api/eval-types"

/** Variability overlaps settled outcomes and must never be added to them. */
export const passing = (cases: EvalRunSummary["cases"]) => cases.passed

export function passRate(cases: EvalRunSummary["cases"]): string {
  return cases.total ? `${passing(cases)}/${cases.total}` : "–"
}

export function percent(cases: EvalRunSummary["cases"]): number | null {
  return cases.total ? Math.round((passing(cases) / cases.total) * 100) : null
}

export const VERDICT_LABEL: Record<EvalVerdict, string> = {
  improvement: "Improvement",
  regression: "Regression",
  tradeoff: "Tradeoff",
  tie: "No change",
}

export const VERDICT_TONE: Record<EvalVerdict, string> = {
  improvement: "text-success",
  regression: "text-destructive",
  tradeoff: "text-warning",
  tie: "text-muted-foreground",
}

export const OUTCOME_LABEL: Record<EvalCaseOutcome, string> = {
  passed: "Passed",
  failed: "Failed",
  flaky: "Flaky",
  error: "Error",
}

export const OUTCOME_TONE: Record<EvalCaseOutcome, string> = {
  passed: "text-success",
  failed: "text-destructive",
  flaky: "text-warning",
  error: "text-destructive",
}

/** "CI · main · 4f2a9c1 · PR #482", "j.okafor", "From a session". */
export function originText(origin: EvalRunOrigin): string {
  if (origin.kind === "ci")
    return ["CI", origin.branch, origin.commit.slice(0, 7), origin.pr ? `PR #${origin.pr}` : null]
      .filter(Boolean)
      .join(" · ")
  if (origin.kind === "session") return "From a session"
  return origin.by
}

/** "▲2 ▼1"; empty when nothing changed or there's nothing to compare with. */
export function changeText(change: EvalRunSummary["change"]): string {
  if (!change) return ""
  return [
    change.improved ? `▲${change.improved}` : "",
    change.regressed ? `▼${change.regressed}` : "",
  ]
    .filter(Boolean)
    .join(" ")
}

export const isLive = (status: EvalRunSummary["status"]) =>
  status === "queued" || status === "running"
