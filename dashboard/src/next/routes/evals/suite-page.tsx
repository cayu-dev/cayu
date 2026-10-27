import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useParams, useRouter } from "@tanstack/react-router"
import { Download, Play } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { EvalSuiteAuthoringAction } from "@/components/dashboard/eval-suite-authoring"
import { Button } from "@/components/ui/button"
import { createEvalRun, downloadEvalCorpus, fetchEvalTargets } from "@/lib/api"
import { dashboardConfig } from "@/lib/config"
import {
  type CorpusLaunchAttempt,
  clearCorpusLaunchAttempt,
  readCorpusLaunchAttempt,
  saveCorpusLaunchAttempt,
} from "@/lib/eval-corpus-launch-attempt"
import {
  createEvalIdempotencyKey,
  EVAL_TARGET_QUERY_KEY,
  evalTargetCatalogMayBeStale,
} from "@/lib/evals-dashboard"
import { fetchSuiteView } from "../../api/eval-views"
import {
  checkLabel,
  isNativeSuiteDetail,
  isReportedSuiteDetail,
  nativeRunSelector,
  runViewPath,
} from "../../app/eval-view-model"
import { ActionsMenu, NavLink, PageBody, sessionPath, TopBar, useGo } from "../common"
import { ReportedSuitePage } from "./reported-pages"
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
  const [recovery, setRecovery] = useState<{
    suite: string
    attempt: CorpusLaunchAttempt | null
    error: Error | null
  } | null>(null)
  useEffect(() => {
    if (!canWrite) return
    try {
      const attempt = readCorpusLaunchAttempt(
        window.sessionStorage,
        dashboardConfig.apiBaseUrl,
        suiteId,
      )
      setRecovery({ suite: suiteId, attempt, error: null })
      if (attempt) setConcurrency(String(attempt.request.max_concurrency))
    } catch (error) {
      setRecovery({ suite: suiteId, attempt: null, error: error as Error })
    }
  }, [canWrite, suiteId])
  const recoveryReady = recovery?.suite === suiteId
  const retained = recoveryReady ? recovery.attempt : null
  const recoveryError = recoveryReady ? recovery.error : null
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
  const data = query.data
  const reported = isReportedSuiteDetail(data) ? data : undefined
  const detail = isNativeSuiteDetail(data) ? data : undefined
  const s = detail?.suite
  const target = targets.data?.items.find((t) => t.target_key === s?.identity.target_key)
  const parsedConcurrency = Number(concurrency)
  const validConcurrency =
    Number.isInteger(parsedConcurrency) &&
    parsedConcurrency >= 1 &&
    parsedConcurrency <= (target?.max_concurrency ?? 1)
  const launch = useMutation({
    mutationFn: async () => {
      if (!canWrite || !recoveryReady || recoveryError)
        throw new Error("Launch recovery is unavailable.")
      // Recovery owns the original envelope; current metadata only prepares a new intent.
      let attempt = retained
      if (!attempt) {
        if (
          !validConcurrency ||
          !s ||
          s.source !== "corpus" ||
          !s.identity.suite_id ||
          !target?.execution_profile_ready ||
          !target.execution_profile ||
          readiness.fresh_launch.state !== "ready"
        )
          throw new Error(
            "The current execution profile is unavailable. Refresh settings before launching.",
          )
        attempt = {
          key: createEvalIdempotencyKey(),
          request: {
            corpus_revision: s.identity.revision,
            suite_id: s.identity.suite_id,
            max_concurrency: parsedConcurrency,
            expected_execution_profile_revision: target.execution_profile.revision,
          },
        }
        saveCorpusLaunchAttempt(window.sessionStorage, dashboardConfig.apiBaseUrl, suiteId, attempt)
        setRecovery({ suite: suiteId, attempt, error: null })
      }
      try {
        const run = await createEvalRun(attempt.request, attempt.key)
        if (
          clearCorpusLaunchAttempt(
            window.sessionStorage,
            dashboardConfig.apiBaseUrl,
            suiteId,
            attempt,
          )
        )
          setRecovery((current) =>
            current?.attempt?.key === attempt.key
              ? { suite: suiteId, attempt: null, error: null }
              : current,
          )
        return run.spec.run_id
      } catch (error) {
        if (evalTargetCatalogMayBeStale(error))
          void client.invalidateQueries({ queryKey: EVAL_TARGET_QUERY_KEY })
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
  if (reported)
    return (
      <ReportedSuitePage
        detail={reported}
        cursor={runsCursor}
        setCursor={setRunsCursor}
        busy={query.isFetching}
        error={query.error}
        retry={query.refetch}
      />
    )
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
                  reviewRun
                  disabled={readiness.catalog_write.state !== "ready"}
                />
              ) : (
                <Button
                  size="sm"
                  disabled={
                    launch.isPending ||
                    !recoveryReady ||
                    !!recoveryError ||
                    (!retained &&
                      (!validConcurrency ||
                        !target?.execution_profile_ready ||
                        readiness.fresh_launch.state !== "ready"))
                  }
                  onClick={() => launch.mutate()}
                >
                  <Play />
                  {launch.isPending ? "Starting…" : retained ? "Retry exact launch" : "Run suite"}
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
        {canWrite && recoveryError && <p role="alert">{recoveryError.message}</p>}
        {canWrite && retained && (
          <section
            aria-label="Unconfirmed launch"
            className="space-y-2 rounded-xl border border-border p-4 text-sm"
          >
            <p>
              The previous launch may already be running. Retry its exact request to recover the
              result.
            </p>
            <p>
              Concurrency: {retained.request.max_concurrency}. The original execution profile is
              retained.
            </p>
            <Button
              variant="outline"
              disabled={launch.isPending}
              onClick={() => {
                if (
                  !window.confirm(
                    "The previous run may already be executing. Abandon its recovery and review a separate new launch?",
                  )
                )
                  return
                if (
                  clearCorpusLaunchAttempt(
                    window.sessionStorage,
                    dashboardConfig.apiBaseUrl,
                    suiteId,
                    retained,
                  )
                ) {
                  setRecovery({ suite: suiteId, attempt: null, error: null })
                  launch.reset()
                }
              }}
            >
              Abandon launch recovery
            </Button>
          </section>
        )}
        {s && detail && (
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
              <RunsTable runs={detail.runs.items} />
              <Pages
                label="Runs"
                cursor={runsCursor}
                busy={query.isFetching}
                hasMore={detail.runs.has_more ?? false}
                onFirst={() => setRunsCursor(undefined)}
                onNext={() => setRunsCursor(detail.runs.next_cursor ?? undefined)}
              />
            </Section>
            <Section title={`Cases (${s.case_count})`}>
              {detail.cases.items.length === 0 && (
                <p className="py-3 text-sm text-muted-foreground">No cases in this suite.</p>
              )}
              <div className="overflow-hidden rounded-xl border border-border">
                {detail.cases.items.map((c) => (
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
                    {canWrite &&
                      s.source === "authored" &&
                      readiness.catalog_write.state === "ready" && (
                        <NavLink
                          to={`/evals/suites/${encodeURIComponent(suiteId)}/edit`}
                          search={{ case: c.id }}
                          className="text-xs underline"
                          aria-label={`Edit ${c.name}`}
                        >
                          Edit case
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
                hasMore={detail.cases.has_more ?? false}
                onFirst={() => setCasesCursor(undefined)}
                onNext={() => setCasesCursor(detail.cases.next_cursor ?? undefined)}
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
                      disabled={!!retained || !!recoveryError || launch.isPending}
                      onChange={(e) => setConcurrency(e.target.value)}
                      className={`${FIELD_CLASS} w-24`}
                    />
                  </label>
                )}
                <div className="flex flex-wrap gap-2">
                  <EvalSuiteAuthoringAction
                    targetKey={s.identity.target_key}
                    initialRevision={s.source === "authored" ? s.identity.revision : undefined}
                    label={s.source === "authored" ? "Edit suite" : "New suite"}
                    disabled={readiness.catalog_write.state !== "ready"}
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
