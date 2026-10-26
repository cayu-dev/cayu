import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useParams, useRouter } from "@tanstack/react-router"
import { Download, Play } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { EvalSuiteAuthoringAction } from "@/components/dashboard/eval-suite-authoring"
import { Button } from "@/components/ui/button"
import { ApiClientError, createEvalRun, downloadEvalCorpus, fetchEvalTargets } from "@/lib/api"
import { dashboardConfig } from "@/lib/config"
import {
  EVAL_TARGET_QUERY_KEY,
  EvalLaunchIdempotencyRegistry,
  evalLaunchFailureIsDefinitive,
  evalLaunchRequestIdentity,
  evalTargetCatalogMayBeStale,
} from "@/lib/evals-dashboard"
import { fetchSuiteView } from "../../api/eval-views"
import { checkLabel, nativeRunSelector, runViewPath } from "../../app/eval-view-model"
import { ActionsMenu, NavLink, PageBody, sessionPath, TopBar, useGo } from "../common"
import {
  CaseOutcome,
  DateText,
  EvalSourceLink,
  FIELD_CLASS,
  LoadState,
  Pages,
  PassingBar,
  RunsTable,
  Section,
  useEvalAccess,
} from "./shared"
import { downloadBlob } from "./technical-details"

export function EvalSuiteViewPage() {
  const { suiteId } = useParams({ strict: false }) as { suiteId: string }
  const [casesCursor, setCasesCursor] = useState<string>()
  const [runsCursor, setRunsCursor] = useState<string>()
  const [toolsOpen, setToolsOpen] = useState(false)
  const [concurrency, setConcurrency] = useState("1")
  const { canWrite, canReadDetails, readiness } = useEvalAccess()
  const client = useQueryClient()
  const go = useGo()
  const router = useRouter()
  const navigationOrigin = useRef(router.state.location.href)
  const alive = useRef(true)
  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])
  const registry = useRef<EvalLaunchIdempotencyRegistry | null>(null)
  const query = useQuery({
    queryKey: ["evals", "suite-view", suiteId, casesCursor, runsCursor],
    queryFn: ({ signal }) =>
      fetchSuiteView(suiteId, { cases_cursor: casesCursor, runs_cursor: runsCursor }, signal),
  })
  const targets = useQuery({
    queryKey: EVAL_TARGET_QUERY_KEY,
    queryFn: ({ signal }) => fetchEvalTargets(signal),
    enabled: canWrite,
  })
  const s = query.data?.suite
  const target = targets.data?.items.find((t) => t.target_key === s?.identity.target_key)
  const parsedConcurrency = Number(concurrency)
  const validConcurrency =
    Number.isInteger(parsedConcurrency) &&
    parsedConcurrency >= 1 &&
    parsedConcurrency <= (target?.max_concurrency ?? 1)
  const launch = useMutation({
    mutationFn: async () => {
      if (
        !canWrite ||
        !validConcurrency ||
        !s ||
        s.source !== "corpus" ||
        !s.identity.suite_id ||
        !target?.execution_profile_ready ||
        !target.execution_profile
      )
        throw new Error(
          "The current execution profile is unavailable. Refresh settings before launching.",
        )
      const identity = evalLaunchRequestIdentity(
        s.identity.revision,
        s.identity.suite_id,
        parsedConcurrency,
        target.execution_profile.revision,
      )
      const owned =
        registry.current ??
        new EvalLaunchIdempotencyRegistry(window.sessionStorage, dashboardConfig.apiBaseUrl)
      registry.current = owned
      try {
        const run = await createEvalRun(
          {
            corpus_revision: s.identity.revision,
            suite_id: s.identity.suite_id,
            max_concurrency: parsedConcurrency,
            expected_execution_profile_revision: target.execution_profile.revision,
          },
          owned.keyFor(identity),
        )
        owned.resolve(identity)
        return run.spec.run_id
      } catch (error) {
        if (evalTargetCatalogMayBeStale(error))
          void client.invalidateQueries({ queryKey: EVAL_TARGET_QUERY_KEY })
        if (error instanceof ApiClientError && evalLaunchFailureIsDefinitive(error.status))
          owned.resolve(identity)
        throw error
      }
    },
    onSuccess: async (id) => {
      await client.invalidateQueries({ queryKey: ["evals"] })
      if (alive.current && router.state.location.href === navigationOrigin.current)
        await go(runViewPath(nativeRunSelector(id)))
    },
  })
  const download = useMutation({
    mutationFn: async () => {
      if (!canReadDetails || !s) return
      const file = await downloadEvalCorpus(s.identity.revision)
      downloadBlob(file.blob, file.filename)
    },
  })
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar
        actions={
          <>
            {s &&
              canWrite &&
              (s.source === "authored" ? (
                <EvalSuiteAuthoringAction
                  targetKey={s.identity.target_key}
                  initialRevision={s.identity.revision}
                  label="Run suite"
                  disabled={readiness.catalog_write.state !== "ready"}
                  onLaunched={async (ids) => {
                    if (alive.current && ids[0]) await go(runViewPath(nativeRunSelector(ids[0])))
                  }}
                />
              ) : (
                <Button
                  size="sm"
                  disabled={
                    launch.isPending ||
                    !validConcurrency ||
                    !target?.execution_profile_ready ||
                    readiness.fresh_launch.state !== "ready"
                  }
                  onClick={() => launch.mutate()}
                >
                  <Play />
                  {launch.isPending ? "Starting…" : "Run suite"}
                </Button>
              ))}
            <ActionsMenu open={toolsOpen} onOpenChange={setToolsOpen}>
              {canReadDetails && s?.source === "corpus" && (
                <Button
                  size="sm"
                  variant="ghost"
                  className="w-full justify-start"
                  disabled={download.isPending}
                  onClick={() => download.mutate()}
                >
                  <Download />
                  {download.isPending ? "Downloading…" : "Download corpus"}
                </Button>
              )}
              <NavLink
                to="/evals/settings"
                className="block rounded-md px-2 py-1.5 hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
              >
                {canWrite ? "Import corpus and advanced tools" : "Settings"}
              </NavLink>
            </ActionsMenu>
          </>
        }
      >
        <NavLink to="/evals" className="mr-2 text-muted-foreground">
          Evals /
        </NavLink>
        <span className="truncate">{s?.name ?? "Suite"}</span>
      </TopBar>
      <PageBody width="max-w-5xl">
        <LoadState error={query.error} pending={query.isPending} retry={query.refetch} />
        {s && query.data && (
          <>
            <h1 className="text-xl font-semibold break-words">{s.name}</h1>
            {s.description && (
              <p className="mt-1 text-sm text-muted-foreground break-words">{s.description}</p>
            )}
            <p className="mt-1 text-xs text-muted-foreground break-words">
              {s.specialist} · {s.trials.trial_count} trials per case,{" "}
              {s.trials.minimum_passed_trials} must pass
            </p>
            {canReadDetails && (
              <LoadState
                error={launch.error ?? download.error}
                pending={false}
                retry={() => {
                  launch.reset()
                  download.reset()
                }}
              />
            )}
            <div className="mt-5 rounded-xl border border-border bg-card px-4 py-3 text-sm">
              {s.baseline ? (
                <div className="flex flex-wrap items-center gap-3">
                  <span className="font-medium">Baseline</span>
                  {s.baseline_result && (
                    <>
                      <PassingBar cases={s.baseline_result.cases} />
                      <NavLink to={runViewPath(s.baseline_result.id)} className="underline">
                        Open run
                      </NavLink>
                    </>
                  )}
                  <span className="text-muted-foreground">
                    Approved by {s.baseline.updated_by}, <DateText value={s.baseline.updated_at} />
                  </span>
                </div>
              ) : (
                <p className="text-muted-foreground">
                  {s.baseline_scope === "execution_group"
                    ? "Baselines belong to each execution group. Open a run to see or approve the baseline for its exact cases and configuration."
                    : "No baseline yet. Approve a finished run to compare later runs with it."}
                </p>
              )}
            </div>
            <Section title="Runs">
              <RunsTable runs={query.data.runs.items} />
              <Pages
                label="Runs"
                cursor={runsCursor}
                busy={query.isFetching}
                hasMore={query.data.runs.has_more ?? false}
                onFirst={() => setRunsCursor(undefined)}
                onNext={() => setRunsCursor(query.data.runs.next_cursor ?? undefined)}
              />
            </Section>
            <Section title={`Cases (${s.case_count})`}>
              {query.data.cases.items.length === 0 && (
                <p className="py-3 text-sm text-muted-foreground">No cases in this suite.</p>
              )}
              <div className="overflow-hidden rounded-xl border border-border">
                {query.data.cases.items.map((c) => (
                  <div
                    key={c.id}
                    className="flex flex-wrap items-start gap-3 border-b border-border px-3 py-2.5 text-sm last:border-0"
                  >
                    <div className="min-w-40 flex-1">
                      <p className="break-words">{c.name}</p>
                      <p className="mt-0.5 text-xs text-muted-foreground break-words">
                        {c.checks.map(checkLabel).join(" · ")}
                      </p>
                    </div>
                    {canReadDetails && !c.source_session_id && (
                      <EvalSourceLink suite={suiteId} caseId={c.id} />
                    )}
                    {canReadDetails && c.source_session_id && (
                      <NavLink to={sessionPath(c.source_session_id)} className="text-xs underline">
                        Source session
                      </NavLink>
                    )}
                    <CaseOutcome presentation={c.last ?? null} />
                  </div>
                ))}
              </div>
              <Pages
                label="Cases"
                cursor={casesCursor}
                busy={query.isFetching}
                hasMore={query.data.cases.has_more ?? false}
                onFirst={() => setCasesCursor(undefined)}
                onNext={() => setCasesCursor(query.data.cases.next_cursor ?? undefined)}
              />
            </Section>
            {canWrite && (
              <Section title="Suite tools">
                {s.source === "corpus" && (
                  <label className="mb-4 flex flex-wrap items-center gap-2 text-sm">
                    Maximum concurrency
                    <input
                      type="number"
                      min={1}
                      max={target?.max_concurrency ?? 1}
                      value={concurrency}
                      onChange={(e) => setConcurrency(e.target.value)}
                      className={`${FIELD_CLASS} w-24`}
                    />
                  </label>
                )}
                <div className="flex flex-wrap gap-2">
                  <EvalSuiteAuthoringAction
                    targetKey={s.identity.target_key}
                    initialRevision={s.source === "authored" ? s.identity.revision : undefined}
                    label="Edit suites and scenarios"
                    disabled={readiness.catalog_write.state !== "ready"}
                    onLaunched={async (ids) => {
                      if (alive.current && ids[0]) await go(runViewPath(nativeRunSelector(ids[0])))
                    }}
                  />
                </div>
              </Section>
            )}
          </>
        )}
      </PageBody>
    </div>
  )
}
