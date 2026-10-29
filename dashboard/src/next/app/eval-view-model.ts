import type {
  EvalReportedRunDetail,
  EvalReportedRunView,
  EvalReportedSuiteDetail,
  EvalRunViewDetail,
  EvalRunViewMetadata,
  EvalSuiteViewDetail,
  EvalSuiteViewIdentity,
  EvalViewComparison,
} from "@/lib/generated/server-api"
import type { EvalCaseFilter } from "../api/eval-views"

/** Canonical selectors locate records; the server alone resolves their authority. */
export function evalViewSelector(value: unknown): string {
  const ascii = JSON.stringify(value).replace(
    /[\u007f-\uffff]/g,
    (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`,
  )
  return btoa(ascii).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "")
}
export const nativeRunSelector = (id: string) => evalViewSelector(["run", id])
export const resultSelector = (revision: string) => evalViewSelector(["result", revision])
export const suiteSelector = (identity: EvalSuiteViewIdentity) =>
  evalViewSelector([
    "suite",
    {
      source: identity.source,
      target_key: identity.target_key,
      revision: identity.revision,
      suite_id: identity.suite_id ?? null,
    },
  ])
export const suiteViewRoute = (id: string) =>
  ({ to: "/evals/suites/$suiteId", params: { suiteId: id } }) as const
export const runViewRoute = (id: string) =>
  ({ to: "/evals/runs/$runId", params: { runId: id } }) as const
export function isReportedRun(value: unknown): value is EvalReportedRunView {
  return (
    value !== null && typeof value === "object" && "source" in value && value.source === "reported"
  )
}
export function isReportedRunDetail(
  value: EvalRunViewDetail | EvalReportedRunDetail | undefined,
): value is EvalReportedRunDetail {
  return value !== undefined && isReportedRun(value.run)
}
export function isReportedSuiteDetail(
  value: EvalSuiteViewDetail | EvalReportedSuiteDetail | undefined,
): value is EvalReportedSuiteDetail {
  return value?.suite.source === "reported"
}
export function runOrigin(
  run: Pick<EvalRunViewMetadata, "origin" | "actor"> | EvalReportedRunView,
): string {
  if (isReportedRun(run))
    return [
      `Reported ${run.origin.kind === "ci" ? "CI" : "CLI"}`,
      run.publisher_id,
      run.origin.branch,
      run.origin.commit?.slice(0, 7),
      run.origin.pull_request == null ? null : `PR #${run.origin.pull_request}`,
    ]
      .filter(Boolean)
      .join(" · ")
  const origins: Record<string, string> = {
    captured_session: "Captured session",
    authored_suite: "Authored suite",
    fresh_execution: "Fresh execution",
    dashboard: "Dashboard",
    sdk: "SDK",
    http: "HTTP",
    ci: "CI",
  }
  return (
    [origins[run.origin] ?? run.origin.replaceAll("_", " "), run.actor]
      .filter(Boolean)
      .join(" · ") || "Origin unavailable"
  )
}
export function comparisonLabel(comparison: EvalViewComparison): string {
  if (comparison.state === "compared" && comparison.verdict)
    return {
      improvement: "Improvement",
      regression: "Regression",
      tradeoff: "Tradeoff",
      tie: "No change",
    }[comparison.verdict]
  return {
    pending: "Comparison pending",
    unavailable: "Comparison unavailable",
    no_baseline: "No baseline to compare",
    incompatible: "Incompatible runs",
    compared: "Comparison unavailable",
  }[comparison.state]
}
export function evalCaseFilter(value: unknown): EvalCaseFilter | undefined {
  return value === "regressed" || value === "improved" || value === "failing" || value === "flaky"
    ? value
    : undefined
}
export function evidenceText(value: unknown): string {
  return typeof value === "string"
    ? value
    : value == null
      ? "Not retained"
      : JSON.stringify(value, null, 2)
}
export function checkLabel(check: unknown): string {
  if (!check || typeof check !== "object") return evidenceText(check)
  const value = check as Record<string, unknown>
  if (typeof value.label === "string") return value.label
  const kind = String(value.kind ?? "Check").replaceAll("_", " ")
  return typeof value.value === "string"
    ? `${kind}: ${value.value}`
    : typeof value.text === "string"
      ? `${kind}: ${value.text}`
      : kind
}

export function isNativeRunDetail(
  value: EvalRunViewDetail | EvalReportedRunDetail | undefined,
): value is EvalRunViewDetail {
  return value !== undefined && "progress" in value && typeof value.run.origin === "string"
}
export function isNativeSuiteDetail(
  value: EvalSuiteViewDetail | EvalReportedSuiteDetail | undefined,
): value is EvalSuiteViewDetail {
  return (
    value !== undefined && (value.suite.source === "authored" || value.suite.source === "corpus")
  )
}

/** Display names never replace the opaque target key used for requests. */
export function evalTargetLabel(target: {
  label?: string | null
  agent_name?: string | null
}): string {
  return target.label?.trim() || target.agent_name?.trim() || "Specialist unavailable"
}

export function baselineApprovalLabel(actor: string): string {
  return actor === "cayu:trusted-local-development" ? "Approved locally" : `Approved by ${actor}`
}
