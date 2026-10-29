import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useParams, useSearch } from "@tanstack/react-router"
import { useRef, useState } from "react"
import { PayloadViewer } from "@/components/dashboard/layout"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import {
  cancelEvalRun,
  compareEvalResults,
  downloadCatalogEvalResultHtml,
  downloadCatalogEvalResultJson,
  downloadEvalResultHtml,
  downloadEvalResultJson,
  fetchEvalResult,
  fetchEvalResultDetail,
  fetchEvalRun,
  selectEvalBaseline,
  submitEvalScenarioApproval,
} from "@/lib/api"
import {
  evalProcessLocalIdentityText,
  evalRunCanCancel,
  evalRunIsActive,
} from "@/lib/evals-dashboard"
import { evalResultRevisionIsValid } from "@/lib/evals-search"
import type { EvalCaseResultView, EvalScenarioTrialProgress } from "@/lib/generated/server-api"
import { fetchRunView, fetchSuiteView } from "../../api/eval-views"
import {
  comparisonLabel,
  evalCaseFilter,
  evidenceText,
  isNativeRunDetail,
  isReportedRun,
  isReportedRunDetail,
  resultSelector,
  runOrigin,
  runViewRoute,
  suiteSelector,
  suiteViewRoute,
} from "../../app/eval-view-model"
import { NavLink, PageBody, sessionRoute, TopBar, useGo } from "../common"
import { ComparisonSummary } from "./comparison-details"
import { ReportedRunPage } from "./reported-pages"
import {
  CaseOutcome,
  DateText,
  EvalSourceLink,
  FIELD_CLASS,
  LoadState,
  Pages,
  PassingBar,
  RunUsage,
  useEvalAccess,
  useEvalCursor,
  Verdict,
} from "./shared"
import {
  downloadBlob,
  OutcomeDimensionsGrid,
  PresentedAssertions,
  ResultInspector,
  ResultPresentationInspector,
  RunLifecycleCard,
} from "./technical-details"

function record(value: unknown): Record<string, unknown> | null {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}
function trialEvidence(value: unknown, number: number | null | undefined) {
  const trials = record(value)?.trials
  return Array.isArray(trials) ? trials.map(record).find((t) => t?.trial_number === number) : null
}
function Output({ evidence }: { evidence: Record<string, unknown> | null | undefined }) {
  const preview = record(evidence?.output)
  return (
    <div className="mt-2 rounded-lg border border-border p-3">
      <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words text-xs">
        {typeof preview?.text === "string" ? preview.text : "Output unavailable"}
      </pre>
      {preview?.preview_truncated === true && (
        <p className="mt-2 text-xs text-muted-foreground">Retained output is truncated.</p>
      )}
    </div>
  )
}
function CaseDrawer({
  item,
  traceLinks,
  onClose,
}: {
  item: EvalCaseResultView
  traceLinks: {
    case_id: string
    trial_number: number
    source_trial_revision: string
    session_id: string
  }[]
  onClose: () => void
}) {
  const [index, setIndex] = useState(0)
  const trial = item.presentation.trials[index]
  const current = trialEvidence(item.evidence, trial?.trial_number)
  const previous = trialEvidence(item.compared_evidence, trial?.trial_number)
  const link = traceLinks.find(
    (l) =>
      l.case_id === item.case.id &&
      l.trial_number === trial?.trial_number &&
      l.source_trial_revision === current?.source_trial_revision,
  )
  return (
    <Sheet
      open
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
    >
      <SheetContent className="overflow-y-auto data-[side=right]:w-[min(96vw,72rem)] data-[side=right]:sm:max-w-none">
        <SheetHeader className="pr-12">
          <SheetTitle className="break-words">{item.case.name}</SheetTitle>
          <SheetDescription>
            {item.presentation.status} · {item.change.replaceAll("_", " ")}
          </SheetDescription>
        </SheetHeader>
        <div className="space-y-5 p-4">
          <div>
            <h3 className="text-sm font-medium">Input</h3>
            <pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-muted p-3 text-xs">
              {evidenceText(item.case.input)}
            </pre>
          </div>
          <div className="flex flex-wrap gap-2" role="tablist" aria-label="Trials">
            {item.presentation.trials.map((t, i) => (
              <button
                key={t.trial_number ?? i}
                type="button"
                role="tab"
                id={`eval-trial-${i}`}
                aria-controls="eval-trial-panel"
                aria-selected={i === index}
                tabIndex={i === index ? 0 : -1}
                className={`rounded-lg border px-3 py-1.5 text-sm ${i === index ? "border-foreground bg-muted" : "border-border"}`}
                onClick={() => setIndex(i)}
                onKeyDown={(e) => {
                  if (["ArrowRight", "ArrowLeft", "Home", "End"].includes(e.key)) {
                    e.preventDefault()
                    const next =
                      e.key === "Home"
                        ? 0
                        : e.key === "End"
                          ? item.presentation.trials.length - 1
                          : (i +
                              (e.key === "ArrowRight" ? 1 : -1) +
                              item.presentation.trials.length) %
                            item.presentation.trials.length
                    setIndex(next)
                    ;(e.currentTarget.parentElement?.children[next] as HTMLElement)?.focus()
                  }
                }}
              >
                Trial {t.trial_number ?? i + 1} · {t.status}
              </button>
            ))}
          </div>
          {trial && (
            <div
              role="tabpanel"
              id="eval-trial-panel"
              aria-labelledby={`eval-trial-${index}`}
              className="space-y-4"
            >
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <h3 className="text-sm font-medium">This run</h3>
                  <Output evidence={current} />
                </div>
                <div>
                  <h3 className="text-sm font-medium">Compared run</h3>
                  <Output evidence={previous} />
                </div>
              </div>
              {link ? (
                <NavLink
                  {...sessionRoute(link.session_id)}
                  search={{ view: "trace" }}
                  className="inline-block text-sm underline"
                >
                  Open trial trace
                </NavLink>
              ) : (
                <p className="text-xs text-muted-foreground">Trial trace link unavailable.</p>
              )}
              <PresentedAssertions assertions={trial.assertions} />
              <details>
                <summary className="cursor-pointer text-sm">Trial details</summary>
                <OutcomeDimensionsGrid dimensions={trial.dimensions} />
                <PayloadViewer value={current ?? trial} />
              </details>
            </div>
          )}
          {(item.structured_judgments?.length ?? 0) > 0 && (
            <details>
              <summary className="cursor-pointer text-sm">Judge comparison evidence</summary>
              <PayloadViewer value={item.structured_judgments} />
            </details>
          )}
          {(item.tool_json_assertions?.length ?? 0) > 0 && (
            <details>
              <summary className="cursor-pointer text-sm">Tool comparison evidence</summary>
              <PayloadViewer value={item.tool_json_assertions} />
            </details>
          )}
        </div>
      </SheetContent>
    </Sheet>
  )
}

export function EvalRunViewPage() {
  const { runId } = useParams({ strict: false }) as { runId: string }
  const search = useSearch({ strict: false }) as {
    compare?: string
    placement?: string
    filter?: ReturnType<typeof evalCaseFilter>
    case?: string
  }
  const go = useGo()
  const client = useQueryClient()
  const { canWrite, canReadDetails } = useEvalAccess()
  const filter = evalCaseFilter(search.filter)
  const [cursor, setCursor] = useEvalCursor(
    JSON.stringify([runId, search.compare ?? null, filter ?? null]),
  )
  const [compareCursor, setCompareCursor] = useState<string>()
  const [metadataOpen, setMetadataOpen] = useState(false)
  const [comparisonOpen, setComparisonOpen] = useState(false)
  const [comparisonRevision, setComparisonRevision] = useState("")
  const baselineOperations = useRef(new Map<string, string>())
  const query = useQuery({
    queryKey: ["evals", "run-view", runId, search.compare, search.placement, filter, cursor],
    queryFn: ({ signal }) =>
      fetchRunView(
        runId,
        {
          compared_with: search.compare,
          placement: search.placement,
          cases_cursor: cursor,
          case_filter: filter,
        },
        signal,
      ),
    refetchInterval: (q) =>
      ["running", "queued", "cancelling"].includes(q.state.data?.run.status ?? "") ? 1500 : false,
    refetchIntervalInBackground: false,
  })
  const data = query.data
  const reported = isReportedRunDetail(data) ? data : undefined
  const detail = isNativeRunDetail(data) ? data : undefined
  const r = detail?.run
  const comparedRevision = detail?.compared_run?.result_revision
  const technicalComparison = useQuery({
    queryKey: ["evals", "technical-comparison", comparedRevision, r?.result_revision],
    queryFn: ({ signal }) => compareEvalResults(comparedRevision!, r!.result_revision!, 0, signal),
    enabled: canReadDetails && comparisonOpen && !!comparedRevision && !!r?.result_revision,
  })
  const native = useQuery({
    queryKey: ["evals", "run", r?.native_run_id],
    queryFn: ({ signal }) => fetchEvalRun(r?.native_run_id ?? "", signal),
    enabled: canReadDetails && !!r?.native_run_id,
    refetchInterval: (q) => (evalRunIsActive(q.state.data) ? 1500 : false),
    refetchIntervalInBackground: false,
  })
  const result = useQuery({
    queryKey: ["evals", "result-detail", r?.result_revision],
    queryFn: ({ signal }) => fetchEvalResultDetail(r?.result_revision ?? "", signal),
    enabled: canReadDetails && !!r?.result_revision,
  })
  const nativeResult = useQuery({
    queryKey: ["evals", "result", r?.native_run_id],
    queryFn: ({ signal }) => fetchEvalResult(r?.native_run_id ?? "", signal),
    enabled: canReadDetails && !!r?.native_run_id && !!r.result_revision,
  })
  const history = useQuery({
    queryKey: ["evals", "compare-options", r?.suite, compareCursor],
    queryFn: ({ signal }) =>
      fetchSuiteView(suiteSelector(r!.suite), { runs_cursor: compareCursor }, signal),
    enabled: !!r,
  })
  const refresh = async () => {
    setCursor(undefined)
    await client.invalidateQueries({ queryKey: ["evals"] })
  }
  const action = useMutation({
    mutationFn: async (
      operation:
        | { kind: "cancel" }
        | { kind: "baseline" }
        | { kind: "scenario"; trial: EvalScenarioTrialProgress; decision: "approve" | "deny" }
        | { kind: "download"; format: "json" | "html" },
    ) => {
      if (!r) throw new Error("Run unavailable")
      if (operation.kind === "download") {
        if (!canReadDetails) throw new Error("Reports require developer access.")
        if (!r.result_revision) throw new Error("No published result")
        const file = r.native_run_id
          ? await (operation.format === "json" ? downloadEvalResultJson : downloadEvalResultHtml)(
              r.native_run_id,
            )
          : await (operation.format === "json"
              ? downloadCatalogEvalResultJson
              : downloadCatalogEvalResultHtml)(r.result_revision)
        downloadBlob(file.blob, file.filename)
        return
      }
      if (!canWrite) throw new Error("Your role can read evals but cannot change them.")
      if (operation.kind === "baseline") {
        if (!r.result_revision) throw new Error("No published result")
        const expectedGeneration = r.baseline?.generation ?? 0
        const identity = JSON.stringify([r.result_revision, expectedGeneration])
        let operationId = baselineOperations.current.get(identity)
        if (!operationId) {
          operationId = `sha256:${Array.from(crypto.getRandomValues(new Uint8Array(32)), (b) => b.toString(16).padStart(2, "0")).join("")}`
          baselineOperations.current.set(identity, operationId)
        }
        await selectEvalBaseline(r.result_revision, {
          result_revision: r.result_revision,
          expected_generation: expectedGeneration,
          operation_id: operationId,
        })
        baselineOperations.current.delete(identity)
      } else if (operation.kind === "cancel") {
        if (!native.data || !evalRunCanCancel(native.data))
          throw new Error("This run cannot be cancelled")
        await cancelEvalRun(native.data.spec.run_id)
      } else {
        const progress = native.data?.scenario_progress
        if (
          !native.data ||
          !progress ||
          !operation.trial.pending_event_id ||
          operation.trial.approval
        )
          throw new Error("This approval is no longer pending")
        await submitEvalScenarioApproval(native.data.spec.run_id, {
          expected_progress_revision: progress.revision,
          trial_number: operation.trial.trial_number,
          event_id: operation.trial.pending_event_id,
          decision: operation.decision,
        })
      }
    },
    onSuccess: refresh,
    onError: () => void refresh(),
  })
  const changeSearch = (patch: Partial<typeof search>) => {
    if (
      ("compare" in patch && patch.compare !== search.compare) ||
      ("filter" in patch && patch.filter !== search.filter)
    )
      setCursor(undefined)
    void go({ ...runViewRoute(runId), search: { ...search, ...patch }, resetScroll: false })
  }
  const selected = detail?.results.items.find((item) => item.case.id === search.case)
  if (reported)
    return (
      <ReportedRunPage
        detail={reported}
        cursor={cursor}
        setCursor={setCursor}
        busy={query.isFetching}
        filter={filter}
        changeSearch={changeSearch}
        selectedId={search.case}
        error={query.error}
        retry={query.refetch}
      />
    )
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar
        actions={
          <>
            {canWrite && r?.result_revision && !r.is_baseline && (
              <Button
                size="sm"
                disabled={action.isPending}
                onClick={() => action.mutate({ kind: "baseline" })}
              >
                Approve as baseline
              </Button>
            )}
            {canWrite && native.data && evalRunCanCancel(native.data) && (
              <Button
                variant="outline"
                size="sm"
                disabled={action.isPending}
                onClick={() => action.mutate({ kind: "cancel" })}
              >
                Cancel run
              </Button>
            )}
          </>
        }
      >
        <NavLink to="/evals" className="mr-2 text-muted-foreground">
          Evals /
        </NavLink>
        {r?.placement && (
          <NavLink
            {...suiteViewRoute(suiteSelector(r.placement.destination))}
            className="mr-2 truncate text-muted-foreground"
          >
            Destination suite /
          </NavLink>
        )}
        <span className="truncate">{r?.suite_name ?? "Run"}</span>
      </TopBar>
      <PageBody width="max-w-6xl">
        <LoadState
          error={query.error}
          pending={query.isPending}
          retry={() => (cursor ? setCursor(undefined) : void query.refetch())}
        />
        {r && detail && (
          <>
            {r.placement && (
              <div className="mb-4 flex flex-wrap items-center gap-2 rounded-lg border border-border p-3 text-sm">
                <p className="min-w-0 flex-1">
                  Scored session · one captured case. This result keeps its own execution suite and
                  baseline; it does not represent a full run of the destination suite.
                </p>
                {canReadDetails && (
                  <EvalSourceLink
                    suite={suiteSelector(r.placement.destination)}
                    caseId={r.placement.case_id}
                    resultRevision={r.result_revision ?? undefined}
                  />
                )}
              </div>
            )}
            {!r.placement &&
              canReadDetails &&
              r.origin === "captured_session" &&
              r.result_revision &&
              detail.results.items[0] && (
                <div className="mb-3">
                  <EvalSourceLink
                    suite={suiteSelector(r.suite)}
                    caseId={detail.results.items[0].case.id}
                    resultRevision={r.result_revision}
                  />
                </div>
              )}
            <NavLink
              {...suiteViewRoute(suiteSelector(r.suite))}
              className="text-sm text-muted-foreground underline"
            >
              {r.suite_name}
            </NavLink>
            <h1 className="mt-2 text-2xl font-semibold">
              <Verdict run={r} />
            </h1>
            <p className="mt-2 text-sm text-muted-foreground">
              {runOrigin(r)} · <DateText value={r.started_at ?? r.created_at} />
            </p>
            {canReadDetails && (
              <LoadState error={action.error} pending={false} retry={() => action.reset()} />
            )}
            <div className="mt-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
              {[
                {
                  name: "Passing",
                  value: <PassingBar cases={r.cases} />,
                  baseline: detail.compared_run?.cases
                    ? `${detail.compared_run.cases.passed}/${detail.compared_run.cases.total}`
                    : null,
                },
                {
                  name: "Flaky",
                  value: r.cases?.flaky ?? "Not available",
                  baseline: detail.compared_run?.cases?.flaky,
                },
                {
                  name: "Judge score",
                  value:
                    r.judge_score == null
                      ? r.judge_score_state.replaceAll("_", " ")
                      : r.judge_score.toFixed(2),
                  baseline: detail.compared_run?.judge_score,
                },
                {
                  name: "Usage",
                  value: <RunUsage run={r} />,
                  baseline: detail.compared_run ? <RunUsage run={detail.compared_run} /> : null,
                },
              ].map((stat) => (
                <div key={stat.name} className="min-w-0 rounded-xl border border-border p-3">
                  <p className="text-xs text-muted-foreground">{stat.name}</p>
                  <p className="mt-1 text-lg tabular-nums break-words">{stat.value}</p>
                  {stat.baseline != null && (
                    <p className="mt-1 text-xs text-muted-foreground">Compared: {stat.baseline}</p>
                  )}
                </div>
              ))}
            </div>
            {["queued", "running", "cancelling"].includes(r.status) && (
              <p className="mt-4 text-sm" role="status">
                {detail.progress.completed_trials ?? "Unknown"} of {detail.progress.total_trials}{" "}
                trials complete · {detail.progress.completed_cases ?? "Unknown"} of{" "}
                {detail.progress.total_cases} cases
              </p>
            )}
            <p className="mt-2 text-xs text-muted-foreground">
              Flaky describes variability and overlaps the outcome counts. Usage covers candidate
              execution; cost may not be retained.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-2">
              <label htmlFor="compare-run" className="text-sm">
                Compared with
              </label>
              <select
                id="compare-run"
                className={`${FIELD_CLASS} max-w-full flex-1 sm:max-w-md`}
                value={search.compare ?? ""}
                onChange={(e) =>
                  changeSearch({ compare: e.target.value || undefined, case: undefined })
                }
              >
                <option value="">Approved baseline</option>
                {search.compare &&
                  !history.data?.runs.items.some((item) => item.id === search.compare) && (
                    <option value={search.compare}>Selected run</option>
                  )}
                {history.data?.runs.items
                  .filter((item) => !isReportedRun(item) && item.id !== r.id)
                  .map((item) => (
                    <option key={item.id} value={item.id}>
                      {new Date(item.created_at).toLocaleString()} · {runOrigin(item)}
                    </option>
                  ))}
              </select>
              {history.data?.runs.has_more && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setCompareCursor(history.data.runs.next_cursor ?? undefined)}
                >
                  More runs
                </Button>
              )}
              {compareCursor && (
                <Button variant="ghost" size="sm" onClick={() => setCompareCursor(undefined)}>
                  First runs
                </Button>
              )}
            </div>
            <LoadState error={history.error} pending={false} retry={history.refetch} />
            <p className="mt-2 text-xs text-muted-foreground">
              {comparisonLabel(detail.comparison)}
              {detail.comparison.compatibility?.process_local_components?.length ||
              detail.comparison.compatibility?.process_local_components_truncated ? (
                <span className="mt-2 block break-words">
                  {evalProcessLocalIdentityText(
                    detail.comparison.compatibility.process_local_components ?? [],
                    detail.comparison.compatibility.process_local_components_truncated,
                  )}
                </span>
              ) : null}
              {detail.comparison.unavailable_reasons?.length
                ? ` · ${detail.comparison.unavailable_reasons.join(" · ").replaceAll("_", " ")}`
                : ""}
              {detail.compared_run && (
                <>
                  {" "}
                  ·{" "}
                  <NavLink {...runViewRoute(detail.compared_run.id)} className="underline">
                    Open compared run
                  </NavLink>
                </>
              )}
            </p>
            <fieldset className="mt-5 flex flex-wrap gap-2" aria-label="Filter cases">
              {["all", "regressed", "improved", "failing", "flaky"].map((filter) => (
                <Button
                  key={filter}
                  size="sm"
                  variant={(search.filter ?? "all") === filter ? "secondary" : "ghost"}
                  aria-pressed={(search.filter ?? "all") === filter}
                  onClick={() => changeSearch({ filter: evalCaseFilter(filter), case: undefined })}
                >
                  {filter.charAt(0).toUpperCase() + filter.slice(1)}
                </Button>
              ))}
            </fieldset>
            <div className="mt-3 overflow-hidden rounded-xl border border-border">
              {detail.results.items.map((item) => (
                <button
                  type="button"
                  key={item.case.id}
                  className="flex w-full flex-wrap items-center gap-3 border-b border-border px-3 py-3 text-left text-sm last:border-0 hover:bg-muted/60 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-ring"
                  onClick={() => changeSearch({ case: item.case.id })}
                >
                  <span className="min-w-40 flex-1 break-words">{item.case.name}</span>
                  <CaseOutcome presentation={item.presentation} />
                  <span
                    className={`text-xs ${item.change === "regressed" ? "text-destructive" : item.change === "improved" ? "text-success" : item.change === "tradeoff" ? "text-warning" : "text-muted-foreground"}`}
                  >
                    {item.change === "unchanged"
                      ? "No change"
                      : item.change === "uncompared"
                        ? "Not compared"
                        : item.change.charAt(0).toUpperCase() + item.change.slice(1)}
                  </span>
                  {item.judge_score != null && (
                    <span className="text-xs tabular-nums">
                      Judge {item.judge_score.toFixed(2)}
                    </span>
                  )}
                </button>
              ))}
            </div>
            {detail.results.items.length === 0 && (
              <p className="py-6 text-sm text-muted-foreground">No matching cases.</p>
            )}
            <Pages
              label="Cases"
              cursor={cursor}
              busy={query.isFetching}
              hasMore={detail.results.has_more ?? false}
              onFirst={() => setCursor(undefined)}
              onNext={() => setCursor(detail.results.next_cursor ?? undefined)}
            />
            {canReadDetails && native.data?.scenario_progress && (
              <section
                aria-label="Scenario approvals"
                data-testid="scenario-run-progress"
                className="mt-6 space-y-3"
              >
                {native.data.scenario_progress.trials
                  .filter((t) => t.phase === "awaiting_approval" && !t.approval)
                  .map((trial) => (
                    <div
                      key={trial.trial_number}
                      className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border p-4"
                    >
                      <div className="min-w-0 text-sm">
                        <p className="font-medium">Trial {trial.trial_number} awaiting approval</p>
                        <p className="break-words text-muted-foreground">
                          Fresh approval required for {trial.pending_tool_name}.
                        </p>
                      </div>
                      {canWrite && (
                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            disabled={action.isPending}
                            onClick={() =>
                              action.mutate({ kind: "scenario", trial, decision: "deny" })
                            }
                          >
                            Deny
                          </Button>
                          <Button
                            size="sm"
                            disabled={action.isPending}
                            onClick={() =>
                              action.mutate({ kind: "scenario", trial, decision: "approve" })
                            }
                          >
                            Approve
                          </Button>
                        </div>
                      )}
                    </div>
                  ))}
              </section>
            )}
            <span
              className="sr-only"
              role="status"
              aria-live="polite"
              aria-atomic="true"
              data-testid="eval-run-status-announcement"
            >
              Eval run status: {r.status}.
            </span>
            <details className="mt-8 rounded-xl border border-border p-4">
              <summary className="cursor-pointer text-sm font-medium">Details</summary>
              <div className="mt-4 space-y-5">
                {!canReadDetails && (
                  <p className="text-sm text-muted-foreground">
                    Your role can read suite comparisons. Lifecycle, reports and retained trace
                    links require developer access.
                  </p>
                )}
                {canReadDetails && (
                  <LoadState
                    error={native.error ?? result.error ?? nativeResult.error}
                    pending={false}
                    retry={() => {
                      void native.refetch()
                      void result.refetch()
                      void nativeResult.refetch()
                    }}
                  />
                )}
                {canReadDetails && native.data && <RunLifecycleCard run={native.data} />}
                {canReadDetails && r.result_revision && !nativeResult.data && (
                  <div className="flex gap-2">
                    {(["json", "html"] as const).map((format) => (
                      <Button
                        key={format}
                        size="sm"
                        variant="outline"
                        disabled={action.isPending}
                        onClick={() => action.mutate({ kind: "download", format })}
                      >
                        Download {format.toUpperCase()}
                      </Button>
                    ))}
                  </div>
                )}
                {canReadDetails &&
                  (nativeResult.data ? (
                    <ResultInspector
                      result={nativeResult.data}
                      pendingAction={action.isPending ? "download" : null}
                      download={(format) => action.mutate({ kind: "download", format })}
                    />
                  ) : (
                    result.data && (
                      <>
                        <ResultPresentationInspector
                          presentation={result.data.presentation}
                          result={result.data.result}
                        />
                        <details>
                          <summary className="cursor-pointer text-sm">Raw result JSON</summary>
                          <PayloadViewer value={result.data.result} />
                        </details>
                      </>
                    )
                  ))}
                {canReadDetails && r.result_revision && (
                  <details>
                    <summary className="cursor-pointer text-sm">
                      Compare another result revision
                    </summary>
                    <form
                      className="mt-3 flex flex-wrap items-end gap-2"
                      onSubmit={(event) => {
                        event.preventDefault()
                        const revision = comparisonRevision.trim()
                        if (evalResultRevisionIsValid(revision) && revision !== r.result_revision)
                          changeSearch({ compare: resultSelector(revision), case: undefined })
                      }}
                    >
                      <label className="min-w-0 flex-1 text-xs">
                        Result revision
                        <input
                          className={`${FIELD_CLASS} mt-1 w-full`}
                          value={comparisonRevision}
                          maxLength={71}
                          onChange={(event) => setComparisonRevision(event.target.value)}
                          placeholder="sha256:…"
                        />
                      </label>
                      <Button
                        size="sm"
                        type="submit"
                        disabled={
                          !evalResultRevisionIsValid(comparisonRevision.trim()) ||
                          comparisonRevision.trim() === r.result_revision
                        }
                      >
                        Compare
                      </Button>
                    </form>
                  </details>
                )}
                {canReadDetails && comparedRevision && r.result_revision && (
                  <details onToggle={(event) => setComparisonOpen(event.currentTarget.open)}>
                    <summary className="cursor-pointer text-sm">Full comparison details</summary>
                    {comparisonOpen && (
                      <>
                        <LoadState
                          error={technicalComparison.error}
                          pending={technicalComparison.isPending}
                          retry={technicalComparison.refetch}
                        />
                        {technicalComparison.data && (
                          <ComparisonSummary comparison={technicalComparison.data} />
                        )}
                      </>
                    )}
                  </details>
                )}
                <details onToggle={(event) => setMetadataOpen(event.currentTarget.open)}>
                  <summary className="cursor-pointer text-sm">Run and comparison metadata</summary>
                  {metadataOpen && <PayloadViewer value={detail} />}
                </details>
              </div>
            </details>
            {selected && (
              <CaseDrawer
                key={`${r.id}:${selected.case.id}`}
                item={selected}
                traceLinks={canReadDetails ? (nativeResult.data?.trial_evidence_links ?? []) : []}
                onClose={() => changeSearch({ case: undefined })}
              />
            )}
          </>
        )}
      </PageBody>
    </div>
  )
}
