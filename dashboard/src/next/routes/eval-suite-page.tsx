/** One suite: its baseline, its runs (newest first) and its cases with their latest outcome. */
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useParams } from "@tanstack/react-router"
import { ChevronRight, Copy, Ellipsis, Play, Settings2 } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { EvalCaseOutcome } from "../api/types"
import { OUTCOME_LABEL, OUTCOME_TONE, originText, passRate } from "../app/eval-model"
import { PrototypeLink as NavLink, usePrototypeGo as useGo } from "../app/prototype-navigation"
import { useEvalSuite } from "../app/queries"
import { useApi, useNow } from "../app/services"
import { when } from "../app/task-model"
import {
  ErrorState,
  PageBody,
  ROW_CLASS,
  ROW_LINK_CLASS,
  sessionRoute,
  TopBar,
  useRowClick,
} from "./common"
import { PassBar, runRoute, SOURCE_LABEL, VerdictText } from "./evals-page"

/** One dot per trial: filled when it passed. */
export function TrialDots({
  passed,
  total,
  outcome,
}: {
  passed: number
  total: number
  outcome: EvalCaseOutcome
}) {
  return (
    <span
      className={cn("inline-flex items-center gap-0.5", OUTCOME_TONE[outcome])}
      title={`${passed} of ${total} trials passed`}
    >
      {Array.from({ length: total }, (_, i) => (
        <span
          // biome-ignore lint/suspicious/noArrayIndexKey: trials are positional
          key={i}
          className={cn(
            "size-2 rounded-full border border-current",
            i < passed ? "bg-current" : "bg-transparent",
          )}
        />
      ))}
      <span className="sr-only">
        {passed} of {total} trials passed
      </span>
    </span>
  )
}

export function EvalSuitePage() {
  const { suiteId } = useParams({ strict: false }) as { suiteId: string }
  const api = useApi()
  const go = useGo()
  const rowClick = useRowClick()
  const now = useNow()
  const queryClient = useQueryClient()
  const suite = useEvalSuite(suiteId)
  const [menuOpen, setMenuOpen] = useState(false)
  const run = useMutation({
    mutationFn: () => api.runEvalSuite(suiteId),
    onSuccess: ({ run_id }) => {
      void queryClient.invalidateQueries({ queryKey: ["evals"] })
      go(runRoute(run_id))
    },
  })

  if (suite.error)
    return (
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar>Suite</TopBar>
        <div className="p-6">
          <ErrorState error={suite.error} />
        </div>
      </div>
    )
  const s = suite.data
  if (!s)
    return (
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar>Suite</TopBar>
      </div>
    )

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <TopBar
        actions={
          <>
            <Button
              size="sm"
              disabled={run.isPending || s.case_count === 0}
              onClick={() => run.mutate()}
            >
              <Play />
              {run.isPending ? "Starting…" : "Run suite"}
            </Button>
            <div className="relative">
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label="More actions"
                aria-expanded={menuOpen}
                onClick={() => setMenuOpen(!menuOpen)}
              >
                <Ellipsis />
              </Button>
              {menuOpen && (
                <div className="absolute top-9 right-0 z-20 w-60 rounded-lg border border-border bg-popover p-1 text-sm shadow-lg">
                  <button
                    type="button"
                    onClick={() => {
                      void navigator.clipboard?.writeText(s.id).catch(() => undefined)
                      setMenuOpen(false)
                    }}
                    className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-muted"
                  >
                    <Copy className="size-4" aria-hidden="true" />
                    Copy suite ID
                  </button>
                  <NavLink
                    to="/evals/settings"
                    className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 hover:bg-muted"
                  >
                    <Settings2 className="size-4" aria-hidden="true" />
                    Judge, export and memory reports
                  </NavLink>
                </div>
              )}
            </div>
          </>
        }
      >
        <span className="flex min-w-0 items-center gap-1">
          <NavLink to="/evals" className="shrink-0 text-muted-foreground hover:text-foreground">
            Evals
          </NavLink>
          <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
          <span className="truncate">{s.name}</span>
        </span>
      </TopBar>
      <PageBody width="max-w-5xl">
        <h1 className="text-xl font-semibold">{s.name}</h1>
        {s.description && <p className="mt-1 text-sm text-muted-foreground">{s.description}</p>}
        <p className="mt-1 text-xs text-muted-foreground">
          {s.specialist} · {SOURCE_LABEL[s.source]} ·{" "}
          {s.trials.per_case > 1
            ? `${s.trials.per_case} trials per case, passes with ${s.trials.min_passes}`
            : "1 trial per case"}
        </p>
        {run.error && <ErrorState error={run.error} />}

        <div className="mt-5 rounded-xl border border-border bg-card px-4 py-3 text-sm">
          {s.baseline ? (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <span className="font-medium">Baseline</span>
              <span className="tabular-nums">{passRate(s.baseline.cases)} passing</span>
              <span className="text-muted-foreground">
                approved by {s.baseline.approved_by},{" "}
                {when(Date.parse(s.baseline.approved_at), now())}
              </span>
              <NavLink {...runRoute(s.baseline.run_id)} className="ml-auto text-xs underline">
                Open run
              </NavLink>
            </div>
          ) : (
            <p className="text-muted-foreground">
              No baseline yet. Approve a run you're happy with, and later runs are compared with it.
            </p>
          )}
        </div>

        <section className="mt-8">
          <h2 className="mb-2 text-sm font-medium">Runs</h2>
          {s.runs.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Not run yet. Run it here, or from CI with{" "}
              <code className="font-mono text-xs">cayu eval run</code>.
            </p>
          ) : (
            <div className="overflow-x-auto rounded-xl border border-border">
              <table className="w-full min-w-[640px] text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs text-muted-foreground">
                    <th className="px-3 py-2 font-medium">When</th>
                    <th className="px-3 py-2 font-medium">From</th>
                    <th className="px-3 py-2 font-medium">Passing</th>
                    <th className="px-3 py-2 font-medium">Against baseline</th>
                    <th className="px-3 py-2 text-right font-medium">Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {s.runs.map((r) => (
                    <tr
                      key={r.id}
                      onClick={rowClick(runRoute(r.id))}
                      className={cn(ROW_CLASS, r.is_baseline && "bg-muted/40")}
                    >
                      <td className="px-3 py-2 whitespace-nowrap">
                        <NavLink {...runRoute(r.id)} className={ROW_LINK_CLASS}>
                          {when(Date.parse(r.started_at), now())}
                        </NavLink>
                        {r.is_baseline && (
                          <span className="ml-2 rounded-full border border-border px-1.5 text-[11px] text-muted-foreground">
                            Baseline
                          </span>
                        )}
                      </td>
                      <td className="max-w-[16rem] truncate px-3 py-2 text-muted-foreground">
                        {originText(r.origin)}
                      </td>
                      <td className="px-3 py-2">
                        {r.status === "completed" ? <PassBar cases={r.cases} /> : "–"}
                      </td>
                      <td className="px-3 py-2">
                        <VerdictText run={r} />
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                        {r.usage.cost != null ? `$${r.usage.cost.toFixed(2)}` : "–"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="mt-8">
          <h2 className="mb-2 text-sm font-medium">Cases ({s.cases.length})</h2>
          <div className="overflow-hidden rounded-xl border border-border">
            {s.cases.map((c) => (
              <div
                key={c.id}
                className="flex min-w-0 items-start gap-3 border-b border-border px-3 py-2.5 text-sm last:border-0"
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate">{c.name}</div>
                  <div className="mt-0.5 line-clamp-2 text-xs text-muted-foreground">
                    {c.checks.map((k) => k.label).join(" · ")}
                  </div>
                </div>
                {c.source_session_id && (
                  <NavLink
                    {...sessionRoute(c.source_session_id)}
                    className="shrink-0 text-xs text-muted-foreground hover:text-foreground hover:underline"
                  >
                    From a session
                  </NavLink>
                )}
                {c.last ? (
                  <span className="flex shrink-0 items-center gap-2 text-xs">
                    <TrialDots
                      passed={c.last.trials_passed}
                      total={c.last.trials}
                      outcome={c.last.outcome}
                    />
                    <span className={OUTCOME_TONE[c.last.outcome]}>
                      {OUTCOME_LABEL[c.last.outcome]}
                    </span>
                  </span>
                ) : (
                  <span className="shrink-0 text-xs text-muted-foreground">Not run</span>
                )}
              </div>
            ))}
          </div>
          {s.runs.some((r) => r.status === "completed") && (
            <p className="mt-2 text-xs text-muted-foreground">
              Outcomes are from the latest finished run.
            </p>
          )}
        </section>
      </PageBody>
    </div>
  )
}
