/**
 * Evals: one list of suites, each with its last run against the approved baseline. A suite
 * opens its cases and run history; a run opens the case-by-case comparison. Settings holds
 * the candidate, the judge and the advanced tools.
 */
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Plus, Settings2, Trash2, X } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { EvalRunSummary, EvalSuiteSummary } from "../api/types"
import {
  changeText,
  isLive,
  originText,
  passRate,
  percent,
  VERDICT_LABEL,
  VERDICT_TONE,
} from "../app/eval-model"
import { useEvalSettings, useEvalSuites, useSpecialists } from "../app/queries"
import { useApi, useNow } from "../app/services"
import { relativeTime } from "../app/session-model"
import {
  EmptyState,
  ErrorState,
  NavLink,
  PageBody,
  ROW_CLASS,
  ROW_LINK_CLASS,
  TopBar,
  useGo,
  useRowClick,
} from "./common"

export const suitePath = (id: string) => `/evals/suites/${encodeURIComponent(id)}`
export const runPath = (id: string) => `/evals/runs/${encodeURIComponent(id)}`

export const SOURCE_LABEL: Record<EvalSuiteSummary["source"], string> = {
  authored: "Written here",
  imported: "Imported",
  captured: "From sessions",
}

/** "11/14" with a bar; green when everything passed, amber otherwise. */
export function PassBar({
  cases,
  className,
}: {
  cases: EvalRunSummary["cases"]
  className?: string
}) {
  const pct = percent(cases)
  return (
    <span className={cn("inline-flex items-center gap-2 tabular-nums", className)}>
      <span className="h-1.5 w-16 overflow-hidden rounded-full bg-muted" aria-hidden="true">
        <span
          className={cn("block h-full rounded-full", pct === 100 ? "bg-success" : "bg-warning")}
          style={{ width: `${pct ?? 0}%` }}
        />
      </span>
      {passRate(cases)}
    </span>
  )
}

/** A run's verdict against what it's compared with, or its status while it runs. */
export function VerdictText({ run }: { run: EvalRunSummary }) {
  if (isLive(run.status))
    return <span className="text-info">{run.status === "queued" ? "Queued" : "Running…"}</span>
  if (run.status !== "completed")
    return (
      <span className="text-destructive">
        {run.status === "failed" ? "Didn't finish" : "Cancelled"}
      </span>
    )
  if (run.is_baseline) return <span className="text-muted-foreground">Baseline</span>
  if (!run.verdict) return <span className="text-muted-foreground">No baseline to compare</span>
  return (
    <span className={cn("whitespace-nowrap", VERDICT_TONE[run.verdict])}>
      {VERDICT_LABEL[run.verdict]}
      {changeText(run.change) && (
        <span className="ml-1.5 tabular-nums">{changeText(run.change)}</span>
      )}
    </span>
  )
}

type DraftCase = { key: number; name: string; input: string; expect_contains: string }
let draftKey = 0
const emptyCase = (): DraftCase => ({ key: ++draftKey, name: "", input: "", expect_contains: "" })

function NewSuite({ onClose }: { onClose: () => void }) {
  const api = useApi()
  const go = useGo()
  const queryClient = useQueryClient()
  const specialists = useSpecialists()
  const names = specialists.data?.agents.map((a) => a.name) ?? []
  const [name, setName] = useState("")
  const [specialist, setSpecialist] = useState("")
  const [trials, setTrials] = useState("1")
  const [cases, setCases] = useState<DraftCase[]>([emptyCase()])
  const chosen = specialist || names[0] || ""
  const ready =
    name.trim() && chosen && cases.every((c) => c.input.trim() && c.expect_contains.trim())
  const create = useMutation({
    mutationFn: () =>
      api.createEvalSuite({
        name: name.trim(),
        specialist: chosen,
        trials: trials === "3" ? { per_case: 3, min_passes: 2 } : { per_case: 1, min_passes: 1 },
        cases: cases.map(({ key: _, ...c }, i) => ({
          ...c,
          name: c.name.trim() || `Case ${i + 1}`,
        })),
      }),
    onSuccess: ({ suite_id }) => {
      void queryClient.invalidateQueries({ queryKey: ["evals"] })
      go(suitePath(suite_id))
    },
  })
  const field =
    "h-8 w-full rounded-lg border border-input bg-background px-2.5 text-sm outline-none focus:border-ring"
  const set = (i: number, patch: Partial<DraftCase>) =>
    setCases(cases.map((c, j) => (i === j ? { ...c, ...patch } : c)))

  return (
    <form
      className="mb-6 space-y-4 rounded-xl border border-border bg-card p-4"
      onSubmit={(e) => {
        e.preventDefault()
        if (ready) create.mutate()
      }}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-medium">New suite</p>
          <p className="text-xs text-muted-foreground">
            Start with cases and the text each answer should contain. Judges and other checks can be
            added on the suite.
          </p>
        </div>
        <Button type="button" variant="ghost" size="icon-sm" aria-label="Close" onClick={onClose}>
          <X />
        </Button>
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        <label className="space-y-1 text-sm sm:col-span-1">
          <span className="text-muted-foreground">Name</span>
          <input
            id="suite-name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className={field}
          />
        </label>
        <label className="space-y-1 text-sm">
          <span className="text-muted-foreground">Specialist</span>
          <select
            id="suite-specialist"
            value={chosen}
            onChange={(e) => setSpecialist(e.target.value)}
            className={field}
          >
            {names.map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
        <label className="space-y-1 text-sm">
          <span className="text-muted-foreground">Trials per case</span>
          <select
            id="suite-trials"
            value={trials}
            onChange={(e) => setTrials(e.target.value)}
            className={field}
          >
            <option value="1">1 trial</option>
            <option value="3">3 trials, 2 must pass</option>
          </select>
        </label>
      </div>
      <div className="space-y-3">
        {cases.map((c, i) => (
          <fieldset
            key={c.key}
            className="grid gap-2 rounded-lg border border-border p-3 sm:grid-cols-[1fr_1fr_auto]"
          >
            <legend className="sr-only">Case {i + 1}</legend>
            <input
              aria-label={`Case ${i + 1} name`}
              value={c.name}
              onChange={(e) => set(i, { name: e.target.value })}
              placeholder={`Case ${i + 1} name (optional)`}
              className={cn(field, "sm:col-span-2")}
            />
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              aria-label={`Remove case ${i + 1}`}
              disabled={cases.length === 1}
              onClick={() => setCases(cases.filter((_, j) => j !== i))}
            >
              <Trash2 />
            </Button>
            <textarea
              aria-label={`Case ${i + 1} input`}
              rows={2}
              value={c.input}
              onChange={(e) => set(i, { input: e.target.value })}
              placeholder="Input, as the agent would receive it"
              className="rounded-lg border border-input bg-background px-2.5 py-1.5 text-sm outline-none focus:border-ring"
            />
            <input
              aria-label={`Case ${i + 1} expected text`}
              value={c.expect_contains}
              onChange={(e) => set(i, { expect_contains: e.target.value })}
              placeholder="The final answer contains…"
              className={cn(field, "self-start sm:col-span-2")}
            />
          </fieldset>
        ))}
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => setCases([...cases, emptyCase()])}
        >
          <Plus />
          Add case
        </Button>
      </div>
      {create.error && <ErrorState error={create.error} />}
      <div className="flex justify-end">
        <Button type="submit" size="sm" disabled={!ready || create.isPending}>
          {create.isPending ? "Creating…" : "Create suite"}
        </Button>
      </div>
    </form>
  )
}

export function EvalsPage() {
  const suites = useEvalSuites()
  const settings = useEvalSettings()
  const now = useNow()
  const rowClick = useRowClick()
  const [creating, setCreating] = useState(false)
  const notReady = settings.data?.features.filter((f) => !f.ready) ?? []
  const rows = suites.data ?? []

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <TopBar
        actions={
          <>
            <NavLink
              to="/evals/settings"
              aria-label="Eval settings"
              className="inline-flex size-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground"
            >
              <Settings2 className="size-4" />
            </NavLink>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCreating(!creating)}
              aria-expanded={creating}
            >
              <Plus />
              New suite
            </Button>
          </>
        }
      >
        Evals
      </TopBar>
      <PageBody width="max-w-6xl">
        {creating && <NewSuite onClose={() => setCreating(false)} />}
        <p className="text-sm text-muted-foreground">
          Suites of cases the agent should keep getting right. Each run is compared with the suite's
          approved baseline, so a change that breaks something shows up as a regression.
        </p>
        {notReady.length > 0 && (
          <p className="mt-2 text-xs text-warning">
            Not available in this deployment: {notReady.map((f) => f.name).join(", ")}.{" "}
            <NavLink to="/evals/settings" className="underline">
              Settings
            </NavLink>
          </p>
        )}
        {suites.error && (
          <div className="mt-3">
            <ErrorState error={suites.error} />
          </div>
        )}
        {suites.isSuccess && rows.length === 0 ? (
          <div className="mt-4">
            <EmptyState title="No eval suites yet">
              Create a suite with a few cases, or open a finished session and choose Add to evals to
              keep it as a case.
            </EmptyState>
          </div>
        ) : (
          <div className="mt-4 overflow-x-auto rounded-xl border border-border">
            <table className="w-full min-w-[760px] text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs text-muted-foreground">
                  <th className="px-3 py-2 font-medium">Suite</th>
                  <th className="px-3 py-2 text-right font-medium">Cases</th>
                  <th className="px-3 py-2 font-medium">Last run</th>
                  <th className="px-3 py-2 font-medium">Against baseline</th>
                  <th className="px-3 py-2 font-medium">Baseline</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((s) => (
                  <tr key={s.id} onClick={rowClick(suitePath(s.id))} className={ROW_CLASS}>
                    <td className="max-w-[22rem] px-3 py-2">
                      <NavLink
                        to={suitePath(s.id)}
                        className={cn("block truncate", ROW_LINK_CLASS)}
                      >
                        {s.name}
                      </NavLink>
                      <div className="mt-0.5 truncate text-xs text-muted-foreground">
                        {s.specialist} · {SOURCE_LABEL[s.source]}
                        {s.trials.per_case > 1 && ` · ${s.trials.per_case} trials`}
                      </div>
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">{s.case_count}</td>
                    <td className="px-3 py-2">
                      {s.last_run ? (
                        <div>
                          {s.last_run.status === "completed" ? (
                            <PassBar cases={s.last_run.cases} />
                          ) : (
                            <VerdictText run={s.last_run} />
                          )}
                          <div className="mt-0.5 truncate text-xs text-muted-foreground">
                            {relativeTime(Date.parse(s.last_run.started_at), now())} ·{" "}
                            {originText(s.last_run.origin)}
                          </div>
                        </div>
                      ) : (
                        <span className="text-muted-foreground">Never run</span>
                      )}
                    </td>
                    <td className="px-3 py-2">
                      {s.last_run?.status === "completed" ? <VerdictText run={s.last_run} /> : "–"}
                    </td>
                    <td className="px-3 py-2 text-xs text-muted-foreground">
                      {s.baseline ? (
                        <>
                          <span className="text-foreground tabular-nums">
                            {passRate(s.baseline.cases)}
                          </span>{" "}
                          · {s.baseline.approved_by},{" "}
                          {relativeTime(Date.parse(s.baseline.approved_at), now())}
                        </>
                      ) : (
                        "None yet"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </PageBody>
    </div>
  )
}
