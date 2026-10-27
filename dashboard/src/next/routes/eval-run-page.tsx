/**
 * One run, case by case against the suite's baseline (or another run). The header answers
 * "did this change make things better"; the table filters to what regressed; a case opens a
 * drawer with each trial's checks, the judge's reasons, both outputs and the trial sessions.
 */
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useParams, useSearch } from "@tanstack/react-router"
import { Check, ChevronRight, RotateCw, ShieldCheck, X } from "lucide-react"
import { type ReactNode, useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { EvalCaseResult, EvalRunDetail } from "../api/types"
import {
  isLive,
  OUTCOME_LABEL,
  OUTCOME_TONE,
  originText,
  passRate,
  VERDICT_LABEL,
  VERDICT_TONE,
} from "../app/eval-model"
import { PrototypeLink as NavLink, usePrototypeGo as useGo } from "../app/prototype-navigation"
import { useEvalRun, useEvalSuite } from "../app/queries"
import { useApi, useNow } from "../app/services"
import { formatDuration, formatTokens } from "../app/session-model"
import { when } from "../app/task-model"
import { ErrorState, PageBody, sessionRoute, TopBar } from "./common"
import { TrialDots } from "./eval-suite-page"
import { PassBar, runRoute, suiteRoute } from "./evals-page"

type Filter = "all" | "regressed" | "improved" | "failing" | "flaky"

const FILTERS: { id: Filter; label: string; match: (r: EvalCaseResult) => boolean }[] = [
  { id: "all", label: "All", match: () => true },
  { id: "regressed", label: "Regressed", match: (r) => r.change === "regressed" },
  { id: "improved", label: "Improved", match: (r) => r.change === "improved" },
  {
    id: "failing",
    label: "Failing",
    match: (r) => r.outcome === "failed" || r.outcome === "error",
  },
  { id: "flaky", label: "Flaky", match: (r) => r.outcome === "flaky" },
]

const CHANGE: Record<EvalCaseResult["change"], { text: string; tone: string }> = {
  improved: { text: "▲ Improved", tone: "text-success" },
  regressed: { text: "▼ Regressed", tone: "text-destructive" },
  unchanged: { text: "Same", tone: "text-muted-foreground" },
  new: { text: "New case", tone: "text-muted-foreground" },
}

const passedTrials = (r: EvalCaseResult) => r.trials.filter((t) => t.passed).length

function Stat({ label, children, note }: { label: string; children: ReactNode; note?: ReactNode }) {
  return (
    <div className="rounded-xl border border-border bg-card px-4 py-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <div className="mt-1 text-lg font-medium tabular-nums">{children}</div>
      {note && <p className="mt-0.5 text-xs text-muted-foreground">{note}</p>}
    </div>
  )
}

function CaseDrawer({
  result,
  run,
  onClose,
}: {
  result: EvalCaseResult
  run: EvalRunDetail
  onClose: () => void
}) {
  const [trialIndex, setTrialIndex] = useState(() => {
    const failing = result.trials.findIndex((t) => !t.passed)
    return failing === -1 ? 0 : failing
  })
  const trial = result.trials[trialIndex] ?? result.trials[0]
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <button
        type="button"
        aria-label="Close case"
        className="absolute inset-0 bg-black/20"
        onClick={onClose}
      />
      <aside
        aria-label={result.case.name}
        className="relative flex h-full w-full max-w-2xl flex-col border-l border-border bg-background shadow-xl"
      >
        <header className="flex items-start gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0 flex-1">
            <p className="truncate font-medium">{result.case.name}</p>
            <p className="mt-0.5 flex items-center gap-2 text-xs">
              <TrialDots
                passed={passedTrials(result)}
                total={result.trials.length}
                outcome={result.outcome}
              />
              <span className={OUTCOME_TONE[result.outcome]}>{OUTCOME_LABEL[result.outcome]}</span>
              <span className={CHANGE[result.change].tone}>{CHANGE[result.change].text}</span>
            </p>
          </div>
          <Button variant="ghost" size="icon-sm" aria-label="Close" onClick={onClose}>
            <X />
          </Button>
        </header>
        <div className="min-h-0 flex-1 space-y-6 overflow-y-auto px-4 py-4 text-sm">
          <section>
            <h3 className="mb-1 text-xs font-medium text-muted-foreground">Input</h3>
            <p className="whitespace-pre-wrap rounded-lg bg-muted px-3 py-2">{result.case.input}</p>
            {result.case.source_session_id && (
              <NavLink
                {...sessionRoute(result.case.source_session_id)}
                className="mt-1 inline-block text-xs text-muted-foreground underline"
              >
                Captured from a session
              </NavLink>
            )}
          </section>

          {result.trials.length > 1 && (
            <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Trials">
              {result.trials.map((t, i) => (
                <button
                  key={t.n}
                  type="button"
                  role="tab"
                  aria-selected={i === trialIndex}
                  onClick={() => setTrialIndex(i)}
                  className={cn(
                    "inline-flex h-7 items-center gap-1.5 rounded-full border px-3 text-[13px]",
                    i === trialIndex
                      ? "border-foreground bg-foreground text-background"
                      : "border-border hover:bg-muted",
                  )}
                >
                  {t.passed ? <Check className="size-3.5" /> : <X className="size-3.5" />}
                  Trial {t.n}
                </button>
              ))}
            </div>
          )}

          {trial && (
            <>
              <section>
                <h3 className="mb-2 text-xs font-medium text-muted-foreground">Checks</h3>
                <ul className="space-y-2">
                  {trial.checks.map((c) => (
                    <li key={c.label} className="flex gap-2">
                      {c.passed ? (
                        <Check
                          className="mt-0.5 size-4 shrink-0 text-success"
                          aria-label="Passed"
                        />
                      ) : (
                        <X
                          className="mt-0.5 size-4 shrink-0 text-destructive"
                          aria-label="Failed"
                        />
                      )}
                      <div className="min-w-0">
                        <p>
                          {c.label}
                          {c.score != null && (
                            <span className="ml-2 text-xs text-muted-foreground tabular-nums">
                              score {c.score.toFixed(2)}
                            </span>
                          )}
                        </p>
                        {(c.reason ?? c.detail) && (
                          <p className="mt-0.5 text-xs text-muted-foreground">
                            {c.reason ?? c.detail}
                          </p>
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              </section>

              <section>
                <h3 className="mb-2 text-xs font-medium text-muted-foreground">
                  Output{result.compared ? " against the run it's compared with" : ""}
                </h3>
                <div className={cn("grid gap-3", result.compared && "sm:grid-cols-2")}>
                  <div>
                    <p className="mb-1 text-xs text-muted-foreground">This run, trial {trial.n}</p>
                    <p className="whitespace-pre-wrap rounded-lg border border-border px-3 py-2 text-[13px]">
                      {trial.output || "No output."}
                    </p>
                  </div>
                  {result.compared && (
                    <div>
                      <p className="mb-1 text-xs text-muted-foreground">
                        {run.compared_with?.is_baseline ? "Baseline" : "Compared run"} ·{" "}
                        <span className={OUTCOME_TONE[result.compared.outcome]}>
                          {OUTCOME_LABEL[result.compared.outcome]}
                        </span>
                      </p>
                      <p className="whitespace-pre-wrap rounded-lg border border-dashed border-border px-3 py-2 text-[13px] text-muted-foreground">
                        {result.compared.output || "No output."}
                      </p>
                    </div>
                  )}
                </div>
              </section>

              <section className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
                <span>{formatDuration(trial.seconds)}</span>
                <span>{formatTokens(trial.usage.tokens)} tokens</span>
                {trial.usage.cost != null && <span>${trial.usage.cost.toFixed(3)}</span>}
                {trial.session_id && (
                  <NavLink
                    {...sessionRoute(trial.session_id)}
                    search={{ view: "trace" }}
                    className="ml-auto text-foreground underline"
                  >
                    Open the trial's trace
                  </NavLink>
                )}
              </section>
            </>
          )}
        </div>
      </aside>
    </div>
  )
}

export function EvalRunPage() {
  const { runId } = useParams({ strict: false }) as { runId: string }
  const search = useSearch({ strict: false }) as {
    compare?: string
    filter?: string
    case?: string
  }
  const api = useApi()
  const go = useGo()
  const now = useNow()
  const queryClient = useQueryClient()
  const run = useEvalRun(runId, search.compare)
  const r = run.data
  const suite = useEvalSuite(r?.suite.id ?? "")
  const set = (patch: Partial<typeof search>) =>
    go({ ...runRoute(runId), search: { ...search, ...patch } })
  const refresh = () => void queryClient.invalidateQueries({ queryKey: ["evals"] })
  const approve = useMutation({ mutationFn: () => api.approveBaseline(runId), onSuccess: refresh })
  const again = useMutation({
    mutationFn: () => api.runEvalSuite(r?.suite.id ?? ""),
    onSuccess: ({ run_id }) => {
      refresh()
      go(runRoute(run_id))
    },
  })

  if (run.error)
    return (
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar>Run</TopBar>
        <div className="p-6">
          <ErrorState error={run.error} />
        </div>
      </div>
    )
  if (!r)
    return (
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar>Run</TopBar>
      </div>
    )

  const filter = (FILTERS.find((f) => f.id === search.filter) ??
    FILTERS[0]) as (typeof FILTERS)[number]
  const rows = r.results.filter(filter.match)
  const open = r.results.find((x) => x.case.id === search.case)
  const others = (suite.data?.runs ?? []).filter((x) => x.id !== r.id && x.status === "completed")
  const c = r.compared_with
  const seconds = r.finished_at
    ? (Date.parse(r.finished_at) - Date.parse(r.started_at)) / 1000
    : null
  const flaky = r.results.filter((x) => x.outcome === "flaky").length

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <TopBar
        actions={
          <>
            {r.status === "completed" && !r.is_baseline && (
              <Button
                size="sm"
                variant="outline"
                disabled={approve.isPending}
                onClick={() => approve.mutate()}
              >
                <ShieldCheck />
                {approve.isPending ? "Approving…" : "Approve as baseline"}
              </Button>
            )}
            <Button
              size="sm"
              variant="outline"
              disabled={again.isPending}
              onClick={() => again.mutate()}
            >
              <RotateCw />
              Run suite again
            </Button>
          </>
        }
      >
        <span className="flex min-w-0 items-center gap-1">
          <NavLink to="/evals" className="shrink-0 text-muted-foreground hover:text-foreground">
            Evals
          </NavLink>
          <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
          <NavLink
            {...suiteRoute(r.suite.id)}
            className="truncate text-muted-foreground hover:text-foreground"
          >
            {r.suite.name}
          </NavLink>
          <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
          <span className="shrink-0">Run</span>
        </span>
      </TopBar>
      <PageBody width="max-w-5xl">
        <h1 className="flex flex-wrap items-baseline gap-x-3 text-xl font-semibold">
          {isLive(r.status) ? (
            <span className="text-info">{r.status === "queued" ? "Queued" : "Running…"}</span>
          ) : r.status !== "completed" ? (
            <span className="text-destructive">
              {r.status === "failed" ? "Didn't finish" : "Cancelled"}
            </span>
          ) : r.is_baseline ? (
            <span>Baseline run</span>
          ) : r.verdict ? (
            <span className={VERDICT_TONE[r.verdict]}>{VERDICT_LABEL[r.verdict]}</span>
          ) : (
            <span>Finished</span>
          )}
          {r.change && (r.change.improved > 0 || r.change.regressed > 0) && (
            <span className="text-sm font-normal text-muted-foreground">
              {r.change.regressed > 0 && `${r.change.regressed} regressed`}
              {r.change.regressed > 0 && r.change.improved > 0 && ", "}
              {r.change.improved > 0 && `${r.change.improved} improved`}
            </span>
          )}
        </h1>
        <p className="mt-1 text-xs text-muted-foreground">
          {when(Date.parse(r.started_at), now())} · {originText(r.origin)} · {r.candidate.model}
          {r.candidate.provider ? ` via ${r.candidate.provider}` : ""}
          {r.is_baseline && " · approved as the suite's baseline"}
        </p>
        {(approve.error || again.error) && <ErrorState error={approve.error ?? again.error} />}

        <div className="mt-5 grid gap-3 sm:grid-cols-4">
          <Stat label="Passing" note={c ? `was ${passRate(c.cases)}` : undefined}>
            {r.status === "completed" ? (
              <PassBar cases={r.cases} className="text-lg" />
            ) : isLive(r.status) ? (
              <span className="text-base">
                {r.cases.passed + r.cases.flaky + r.cases.failed} of {r.cases.total} cases done
              </span>
            ) : (
              "–"
            )}
          </Stat>
          <Stat label="Flaky" note="passed, but not every trial">
            {flaky}
          </Stat>
          <Stat
            label="Judge score"
            note={c?.score != null ? `was ${c.score.toFixed(2)}` : undefined}
          >
            {r.score != null ? r.score.toFixed(2) : "–"}
          </Stat>
          <Stat
            label="Cost"
            note={`${formatTokens(r.usage.tokens)} tokens · ${formatDuration(seconds)}`}
          >
            {r.usage.cost != null ? `$${r.usage.cost.toFixed(2)}` : "–"}
          </Stat>
        </div>

        <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
          <fieldset className="flex min-w-0 flex-wrap gap-1.5">
            <legend className="sr-only">Filter cases</legend>
            {FILTERS.map((f) => {
              const n = r.results.filter(f.match).length
              const on = filter.id === f.id
              if (f.id !== "all" && n === 0) return null
              return (
                <button
                  key={f.id}
                  type="button"
                  aria-pressed={on}
                  onClick={() => set({ filter: f.id === "all" ? undefined : f.id })}
                  className={cn(
                    "inline-flex h-7 items-center gap-1.5 rounded-full border px-3 text-[13px]",
                    on
                      ? "border-foreground bg-foreground text-background"
                      : "border-border hover:bg-muted",
                  )}
                >
                  {f.label}
                  <span
                    className={cn(
                      "rounded-full px-1.5 text-xs font-medium tabular-nums",
                      on
                        ? "bg-background/20"
                        : f.id === "regressed"
                          ? "bg-destructive/10 text-destructive"
                          : "bg-muted text-muted-foreground",
                    )}
                  >
                    {n}
                  </span>
                </button>
              )
            })}
          </fieldset>
          {!r.is_baseline && (
            <label className="flex items-center gap-2 text-[13px] text-muted-foreground">
              Compared with
              <select
                id="eval-compare"
                value={search.compare ?? ""}
                onChange={(e) => set({ compare: e.target.value || undefined })}
                className="h-8 rounded-lg border border-input bg-background px-2 text-sm text-foreground outline-none focus:border-ring"
              >
                <option value="">The baseline</option>
                {others
                  .filter((o) => !o.is_baseline)
                  .map((o) => (
                    <option key={o.id} value={o.id}>
                      {when(Date.parse(o.started_at), now())} · {originText(o.origin)}
                    </option>
                  ))}
              </select>
            </label>
          )}
        </div>

        <div className="mt-3 overflow-x-auto rounded-xl border border-border">
          <table className="w-full min-w-[640px] text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs text-muted-foreground">
                <th className="px-3 py-2 font-medium">Case</th>
                <th className="px-3 py-2 font-medium">Trials</th>
                <th className="px-3 py-2 font-medium">Outcome</th>
                {c && (
                  <th className="px-3 py-2 font-medium">
                    Against {c.is_baseline ? "baseline" : "that run"}
                  </th>
                )}
                <th className="px-3 py-2 text-right font-medium">Judge</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((x) => (
                <tr
                  key={x.case.id}
                  tabIndex={0}
                  onClick={() => set({ case: x.case.id })}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") set({ case: x.case.id })
                  }}
                  className="cursor-pointer border-b border-border last:border-0 hover:bg-muted/60 focus-visible:bg-muted/60 focus-visible:outline-none"
                >
                  <td className="max-w-[24rem] px-3 py-2">
                    <div className="truncate">{x.case.name}</div>
                    <div className="truncate text-xs text-muted-foreground">{x.case.input}</div>
                  </td>
                  <td className="px-3 py-2">
                    <TrialDots
                      passed={passedTrials(x)}
                      total={x.trials.length}
                      outcome={x.outcome}
                    />
                  </td>
                  <td className={cn("px-3 py-2", OUTCOME_TONE[x.outcome])}>
                    {OUTCOME_LABEL[x.outcome]}
                  </td>
                  {c && (
                    <td className={cn("px-3 py-2 whitespace-nowrap", CHANGE[x.change].tone)}>
                      {CHANGE[x.change].text}
                      {x.compared && x.change !== "unchanged" && (
                        <span className="ml-1.5 text-xs text-muted-foreground">
                          was {OUTCOME_LABEL[x.compared.outcome].toLowerCase()}
                        </span>
                      )}
                    </td>
                  )}
                  <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                    {x.score != null ? x.score.toFixed(2) : "–"}
                  </td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-3 py-6 text-center text-muted-foreground">
                    {isLive(r.status) ? "Results appear as cases finish." : "No cases match."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {!c && r.status === "completed" && !r.is_baseline && (
          <p className="mt-2 text-xs text-muted-foreground">
            This suite has no baseline yet, so there's nothing to compare with. Approve this run to
            make it the baseline.
          </p>
        )}
      </PageBody>
      {open && (
        <CaseDrawer
          key={open.case.id}
          result={open}
          run={r}
          onClose={() => set({ case: undefined })}
        />
      )}
    </div>
  )
}
