import { useQuery } from "@tanstack/react-query"
import { type ReactNode, useState } from "react"
import { useDashboardCapability, useServerContract } from "@/components/dashboard/server-contract"
import { useViewer } from "@/components/dashboard/viewer"
import { Button } from "@/components/ui/button"
import type {
  EvalCasePresentationV2,
  EvalRunViewMetadata,
  EvalRunViewSummary,
  EvalViewCaseCounts,
} from "@/lib/generated/server-api"
import { cn } from "@/lib/utils"
import { fetchEvalCaseSource } from "../../api/eval-additions"
import { comparisonLabel, runOrigin, runViewPath, suiteSelector } from "../../app/eval-view-model"
import { NavLink, ROW_CLASS, ROW_LINK_CLASS, sessionPath, useRowClick } from "../common"

export const FIELD_CLASS =
  "h-8 min-w-0 rounded-lg border border-input bg-background px-2.5 text-sm focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
/** Browser back/forward cannot reuse a continuation from another filter or comparison. */
export function useEvalCursor(scope: string) {
  const [page, setPage] = useState<{ scope: string; cursor?: string }>({ scope })
  return [
    page.scope === scope ? page.cursor : undefined,
    (cursor: string | undefined) => setPage({ scope, cursor }),
  ] as const
}

export function useEvalAccess() {
  const viewer = useViewer()
  const mutate = useDashboardCapability({ kind: "surface", surface: "evals", operation: "mutate" })
  const readiness = useServerContract().capabilities.evals_readiness
  return {
    canReadDetails: viewer?.full_access === true,
    canWrite: viewer?.full_access === true && mutate.enabled,
    readiness,
  }
}
export function LoadState({
  error,
  pending,
  retry,
}: {
  error: Error | null
  pending: boolean
  retry: () => unknown
}) {
  if (error)
    return (
      <div
        role="alert"
        className="my-4 space-y-2 rounded-lg border border-destructive/30 p-3 text-sm break-words"
      >
        <p>{error.message}</p>
        <Button variant="outline" size="sm" onClick={() => void retry()}>
          Retry
        </Button>
      </div>
    )
  if (pending)
    return (
      <p role="status" className="py-6 text-sm text-muted-foreground">
        Loading evals…
      </p>
    )
  return null
}
export function Pages({
  hasMore,
  cursor,
  busy,
  onNext,
  onFirst,
  label,
}: {
  hasMore: boolean
  cursor?: string
  busy: boolean
  onNext: () => void
  onFirst: () => void
  label: string
}) {
  if (!hasMore && !cursor) return null
  return (
    <nav aria-label={`${label} pages`} className="mt-3 flex justify-end gap-2">
      <Button variant="ghost" size="sm" disabled={!cursor || busy} onClick={onFirst}>
        First page
      </Button>
      <Button variant="outline" size="sm" disabled={!hasMore || busy} onClick={onNext}>
        Next {label.toLowerCase()}
      </Button>
    </nav>
  )
}
export function PassingBar({ cases }: { cases: EvalViewCaseCounts | null }) {
  if (!cases) return <span className="text-muted-foreground">Not available</span>
  const percent = cases.total ? (100 * cases.passed) / cases.total : 0
  return (
    <span className="inline-flex items-center gap-2 tabular-nums">
      <span aria-hidden="true" className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
        <span
          className={cn("block h-full rounded-full", percent === 100 ? "bg-success" : "bg-warning")}
          style={{ width: `${percent}%` }}
        />
      </span>
      {cases.passed}/{cases.total}
    </span>
  )
}
const OUTCOME_TONE = {
  passed: "text-success",
  failed: "text-destructive",
  error: "text-destructive",
  unavailable: "text-muted-foreground",
} as const

/** Dots describe retained trials, never an inferred order from aggregate counts. */
export function CaseOutcome({ presentation }: { presentation: EvalCasePresentationV2 | null }) {
  if (!presentation)
    return <span className="text-xs text-muted-foreground">Outcome unavailable</span>
  const { reliability, trials, status } = presentation
  return (
    <span className="inline-flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
      <span
        className="inline-flex items-center gap-1"
        role="img"
        aria-label="Trial outcomes"
        title={trials
          .slice(0, 20)
          .map((trial, index) => `Trial ${trial.trial_number ?? index + 1}: ${trial.status}`)
          .join("; ")}
      >
        {trials.slice(0, 20).map((trial, index) => (
          <span
            key={trial.trial_number ?? index}
            aria-hidden="true"
            title={`Trial ${trial.trial_number ?? index + 1}: ${trial.status}`}
            className={cn("size-1.5 rounded-full bg-current", OUTCOME_TONE[trial.status])}
          />
        ))}
        {trials.length > 20 && <span aria-hidden="true">+{trials.length - 20}</span>}
      </span>
      <span className="tabular-nums text-muted-foreground">
        {reliability.passed_trials}/{reliability.total_trials} passed
      </span>
      <span className={OUTCOME_TONE[status]}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    </span>
  )
}

export function Verdict({ run }: { run: EvalRunViewSummary }) {
  if (run.status !== "completed")
    return (
      <span className={run.status === "failed" ? "text-destructive" : "text-muted-foreground"}>
        {run.status === "running"
          ? "Running…"
          : run.status.charAt(0).toUpperCase() + run.status.slice(1)}
      </span>
    )
  if (run.is_baseline) return <span className="text-muted-foreground">Baseline</span>
  const c = run.comparison
  return (
    <span
      className={cn(
        c.verdict === "regression"
          ? "text-destructive"
          : c.verdict === "improvement"
            ? "text-success"
            : c.verdict === "tradeoff"
              ? "text-warning"
              : "text-muted-foreground",
      )}
    >
      <span>{comparisonLabel(c)}</span>
      {c.state === "compared" && (
        <span className="ml-1.5 whitespace-nowrap tabular-nums">
          {c.improved_cases ? `▲${c.improved_cases} ` : ""}
          {c.regressed_cases ? `▼${c.regressed_cases}` : ""}
        </span>
      )}
    </span>
  )
}
export function RunUsage({ run }: { run: EvalRunViewMetadata }) {
  if (run.usage.cost != null)
    return (
      <>
        {run.usage.currency ?? ""} {run.usage.cost}
      </>
    )
  if (run.usage.tokens != null) return <>{run.usage.tokens} tokens</>
  if (run.usage.observed_tokens != null)
    return <>{run.usage.observed_tokens} observed tokens (partial)</>
  return <>Not retained</>
}
export function DateText({ value }: { value: string | null }) {
  return <>{value ? new Date(value).toLocaleString() : "Not retained"}</>
}
export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mt-8">
      <h2 className="mb-2 text-sm font-medium">{title}</h2>
      {children}
    </section>
  )
}
export function RunsTable({ runs }: { runs: EvalRunViewSummary[] }) {
  const rowClick = useRowClick()
  if (!runs.length)
    return <p className="text-sm text-muted-foreground">No runs recorded for this suite yet.</p>
  return (
    <div className="overflow-x-auto rounded-xl border border-border">
      <table className="w-full min-w-[640px] text-sm">
        <thead>
          <tr className="border-b border-border text-left text-xs text-muted-foreground">
            {["When", "From", "Passing", "Against baseline", "Usage"].map((h) => (
              <th key={h} className="px-3 py-2 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr
              key={run.id}
              className={ROW_CLASS}
              onClick={rowClick(
                runViewPath(run.id),
                run.placement ? { placement: suiteSelector(run.placement.destination) } : undefined,
              )}
            >
              <td className="px-3 py-2">
                <NavLink
                  to={runViewPath(run.id)}
                  search={
                    run.placement
                      ? { placement: suiteSelector(run.placement.destination) }
                      : undefined
                  }
                  className={ROW_LINK_CLASS}
                >
                  <DateText value={run.started_at ?? run.created_at} />
                </NavLink>
              </td>
              <td className="max-w-64 px-3 py-2 break-words text-muted-foreground">
                {run.placement ? "Scored session · one captured case" : runOrigin(run)}
              </td>
              <td className="px-3 py-2">
                <PassingBar cases={run.cases} />
              </td>
              <td className="px-3 py-2">
                <Verdict run={run} />
              </td>
              <td className="px-3 py-2 text-muted-foreground">
                <RunUsage run={run} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** The private source boundary is queried only on explicit demand and by full-access callers. */
export function EvalSourceLink({
  suite,
  caseId,
  resultRevision,
}: {
  suite: string
  caseId: string
  resultRevision?: string
}) {
  const [requested, setRequested] = useState(false)
  const query = useQuery({
    queryKey: ["evals", "source", suite, caseId, resultRevision],
    queryFn: ({ signal }) => fetchEvalCaseSource(suite, caseId, resultRevision, signal),
    enabled: requested,
    retry: false,
  })
  if (query.data?.state === "available" && query.data.session_id)
    return (
      <NavLink to={sessionPath(query.data.session_id)} className="text-xs underline">
        Source session
      </NavLink>
    )
  if (query.data?.state === "unavailable")
    return <span className="text-xs text-muted-foreground">Source session unavailable</span>
  return (
    <span className="inline-flex flex-wrap items-center gap-2">
      <Button
        size="xs"
        variant="ghost"
        disabled={query.isFetching}
        onClick={() => (requested ? void query.refetch() : setRequested(true))}
      >
        {query.isFetching
          ? "Finding source…"
          : query.error
            ? "Retry source lookup"
            : "Find source session"}
      </Button>
      {query.error && (
        <span role="alert" className="max-w-xs text-xs text-destructive break-words">
          {query.error.message}
        </span>
      )}
    </span>
  )
}
