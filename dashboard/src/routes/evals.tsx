import { useQuery, useQueryClient } from "@tanstack/react-query"
import { useNavigate, useSearch } from "@tanstack/react-router"
import {
  Ban,
  CheckCircle2,
  Database,
  Download,
  FileJson,
  FlaskConical,
  LoaderCircle,
  Play,
  RotateCcw,
  Upload,
} from "lucide-react"
import {
  type ChangeEvent,
  type KeyboardEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react"
import { EvalSuiteAuthoringAction } from "@/components/dashboard/eval-suite-authoring"
import {
  DataCard,
  Page,
  PageHeader,
  PayloadViewer,
  StateMessage,
} from "@/components/dashboard/layout"
import { ScenarioAuthoring } from "@/components/dashboard/scenario-authoring"
import { useDashboardCapability, useServerContract } from "@/components/dashboard/server-contract"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  ApiClientError,
  cancelEvalRun,
  compareEvalResults,
  createEvalRun,
  downloadCatalogEvalResultHtml,
  downloadCatalogEvalResultJson,
  downloadEvalCorpus,
  downloadEvalResultHtml,
  downloadEvalResultJson,
  downloadEvalScenario,
  type EvalCorpusEntry,
  type EvalResult,
  type EvalResultComparison,
  type EvalResultDetail,
  type EvalResultSummary,
  type EvalRun,
  type EvalScenarioEntry,
  type EvalStatus,
  type EvalTarget,
  fetchEvalCases,
  fetchEvalCorpora,
  fetchEvalResult,
  fetchEvalResultDetail,
  fetchEvalResults,
  fetchEvalRun,
  fetchEvalRuns,
  fetchEvalScenario,
  fetchEvalScenarios,
  fetchEvalSuites,
  fetchEvalTargets,
  importEvalCorpus,
  selectEvalBaseline,
  submitEvalScenarioApproval,
} from "@/lib/api"
import { dashboardConfig } from "@/lib/config"
import { dashboardCapabilityUnavailableText } from "@/lib/dashboard-capabilities"
import {
  EVAL_RESULT_QUERY_RETENTION,
  EVAL_TARGET_QUERY_KEY,
  EVAL_TARGET_STALE_TIME_MS,
  EvalLaunchIdempotencyRegistry,
  evalCancellationNotice,
  evalComparisonReasonText,
  evalErrorMessage,
  evalLaunchFailureIsDefinitive,
  evalLaunchNotice,
  evalLaunchRequestIdentity,
  evalRunCanCancel,
  evalRunHasResult,
  evalRunIsActive,
  evalTargetCatalogMayBeStale,
  evalTrialCostSummary,
  preflightEvalCorpusFile,
  retryEvalQuery,
  shortEvalIdentity,
} from "@/lib/evals-dashboard"
import {
  EVALS_READINESS_OPERATIONS,
  evalsReadinessReasonText,
  evalsReadinessStateLabel,
} from "@/lib/evals-readiness"
import { type EvalsSearch, evalResultRevisionIsValid, evalsSearchWithout } from "@/lib/evals-search"
import { PROCESS_EVENT_OPTIONS } from "@/lib/evaluation-promotion"
import { formatBytes, formatCount, formatDateTime } from "@/lib/format"
import type {
  EvalAssertionPresentationV1,
  EvalResultOutcomeDimensionsV1,
  EvalResultPresentationV1,
  EvalScenarioTrialProgress,
  EvalStructuredJudgeComparisonV1,
  EvalStructuredJudgePresentationV1,
  EvalsReadiness,
  EvalToolJsonAssertionComparisonV1,
  PublishedProcessEventsInOrderDetail,
} from "@/lib/generated/server-api"

const PAGE_LIMIT = 25
type UpdateEvalsSearch = (next: (current: EvalsSearch) => EvalsSearch) => Promise<void>

export function EvalsPage() {
  const search = useSearch({ from: "/evals" })
  const navigate = useNavigate({ from: "/evals" })
  const queryClient = useQueryClient()
  const readiness = useServerContract().capabilities.evals_readiness
  const catalogReady = readiness.catalog_read.state === "ready"
  const resultsReady = readiness.captured_result_persistence.state === "ready"
  const targets = useQuery({
    queryKey: EVAL_TARGET_QUERY_KEY,
    queryFn: ({ signal }) => fetchEvalTargets(signal),
    enabled: catalogReady,
    staleTime: EVAL_TARGET_STALE_TIME_MS,
  })
  const selectedTargetKey = targets.data?.items.some(
    (target) => target.target_key === search.target,
  )
    ? search.target
    : targets.data?.default_target_key
  const selectedTarget = targets.data?.items.find(
    (target) => target.target_key === selectedTargetKey,
  )
  const mutateCapability = useDashboardCapability({
    kind: "surface",
    surface: "evals",
    operation: "mutate",
  })
  const mutationUnavailable = dashboardCapabilityUnavailableText(mutateCapability)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const catalogTabRef = useRef<HTMLButtonElement>(null)
  const resultsTabRef = useRef<HTMLButtonElement>(null)
  const runsTabRef = useRef<HTMLButtonElement>(null)
  const actionControllerRef = useRef<AbortController | null>(null)
  const [pendingAction, setPendingAction] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [actionNotice, setActionNotice] = useState<string | null>(null)

  const cancelAction = useCallback(() => {
    actionControllerRef.current?.abort()
    actionControllerRef.current = null
  }, [])
  useEffect(() => cancelAction, [cancelAction])

  const runAction = useCallback(
    async (name: string, action: (signal: AbortSignal) => Promise<string | undefined>) => {
      cancelAction()
      const controller = new AbortController()
      actionControllerRef.current = controller
      setPendingAction(name)
      setActionError(null)
      setActionNotice(null)
      try {
        const notice = await action(controller.signal)
        if (!controller.signal.aborted && notice) setActionNotice(notice)
      } catch (error) {
        if (!controller.signal.aborted) {
          setActionError(evalErrorMessage(error, "The Evals operation failed."))
        }
      } finally {
        if (actionControllerRef.current === controller) {
          actionControllerRef.current = null
          setPendingAction(null)
        }
      }
    },
    [cancelAction],
  )

  const updateSearch = useCallback(
    (next: (current: EvalsSearch) => EvalsSearch) => navigate({ search: next, resetScroll: false }),
    [navigate],
  )

  const importCorpus = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (!file || pendingAction !== null) return
    void runAction("import", async (signal) => {
      await preflightEvalCorpusFile(file)
      if (signal.aborted) return
      const imported = await importEvalCorpus(file, signal)
      if (signal.aborted) return
      await queryClient.invalidateQueries({ queryKey: ["evals", "corpora"] })
      if (signal.aborted) return
      await updateSearch((current) => ({
        ...evalsSearchWithout(current, "suite", "suites_cursor", "cases_cursor", "corpora_cursor"),
        tab: "catalog",
        corpus: imported.revision,
        target: imported.target_key,
      }))
      return `Imported corpus ${shortEvalIdentity(imported.revision)}.`
    })
  }

  const activeTab = search.tab ?? "catalog"
  const selectTarget = (targetKey: string) =>
    updateSearch((current) => ({
      ...evalsSearchWithout(
        current,
        "target",
        "corpus",
        "suite",
        "run",
        "result",
        "baseline",
        "status",
        "corpora_cursor",
        "suites_cursor",
        "cases_cursor",
        "runs_cursor",
        "results_cursor",
      ),
      target: targetKey,
    }))
  const showCatalog = () =>
    updateSearch((current) => ({
      ...evalsSearchWithout(
        current,
        "run",
        "result",
        "baseline",
        "runs_cursor",
        "results_cursor",
        "status",
      ),
      tab: "catalog",
    }))
  const showResults = () =>
    updateSearch((current) => ({
      ...evalsSearchWithout(
        current,
        "suite",
        "run",
        "baseline",
        "suites_cursor",
        "cases_cursor",
        "corpora_cursor",
        "runs_cursor",
        "status",
      ),
      tab: "results",
    }))
  const showRuns = () =>
    updateSearch((current) => ({
      ...evalsSearchWithout(
        current,
        "suite",
        "result",
        "suites_cursor",
        "cases_cursor",
        "corpora_cursor",
        "results_cursor",
      ),
      tab: "runs",
    }))
  const moveTabFocus = (event: KeyboardEvent<HTMLDivElement>) => {
    const focusedTab =
      document.activeElement === runsTabRef.current
        ? "runs"
        : document.activeElement === resultsTabRef.current
          ? "results"
          : "catalog"
    const nextTab =
      event.key === "Home"
        ? "catalog"
        : event.key === "End"
          ? "runs"
          : event.key === "ArrowLeft"
            ? focusedTab === "runs"
              ? "results"
              : focusedTab === "results"
                ? "catalog"
                : "runs"
            : event.key === "ArrowRight"
              ? focusedTab === "catalog"
                ? "results"
                : focusedTab === "results"
                  ? "runs"
                  : "catalog"
              : null
    if (nextTab === null) return
    event.preventDefault()
    if (nextTab === "catalog") {
      catalogTabRef.current?.focus()
      void showCatalog()
    } else if (nextTab === "results") {
      resultsTabRef.current?.focus()
      void showResults()
    } else {
      runsTabRef.current?.focus()
      void showRuns()
    }
  }

  if (!catalogReady) {
    return (
      <Page>
        <PageHeader
          title="Evals"
          description="Evaluate captured production behavior and build reusable regression protection."
        />
        <EvalsReadinessOverview readiness={readiness} />
        <StateMessage className="rounded-lg border border-border bg-muted/30 py-12" role="status">
          <div className="font-medium text-foreground">The Evals catalog is not ready yet</div>
          <div className="mt-1">
            This page remains available so the deployment&apos;s exact readiness and planned
            capabilities are visible.
          </div>
        </StateMessage>
      </Page>
    )
  }

  return (
    <Page>
      <PageHeader
        title="Evals"
        description="Manage portable regression corpora and durable current-app evaluation runs."
        actions={
          <>
            <EvalTargetSelector
              targets={targets.data?.items ?? []}
              selectedTargetKey={selectedTargetKey}
              loading={targets.isLoading}
              selectTarget={selectTarget}
            />
            {selectedTargetKey && (
              <EvalSuiteAuthoringAction
                key={selectedTargetKey}
                targetKey={selectedTargetKey}
                disabled={!mutateCapability.enabled}
                onLaunched={async (runIds) => {
                  const firstRunId = runIds[0]
                  if (!firstRunId) return
                  await updateSearch((current) => ({
                    ...evalsSearchWithout(
                      current,
                      "suite",
                      "result",
                      "baseline",
                      "suites_cursor",
                      "cases_cursor",
                      "corpora_cursor",
                      "results_cursor",
                      "runs_cursor",
                      "status",
                    ),
                    tab: "runs",
                    run: firstRunId,
                  }))
                }}
              />
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="application/json,.json"
              className="sr-only"
              data-testid="eval-import-file"
              onChange={importCorpus}
            />
            <Button
              type="button"
              variant="outline"
              disabled={!mutateCapability.enabled || pendingAction !== null}
              title={mutationUnavailable ?? undefined}
              onClick={() => fileInputRef.current?.click()}
            >
              {pendingAction === "import" ? <LoaderCircle className="animate-spin" /> : <Upload />}
              {pendingAction === "import" ? "Importing..." : "Import corpus"}
            </Button>
          </>
        }
      />

      <EvalsReadinessOverview readiness={readiness} />

      {selectedTarget && (
        <EvalExecutionProfileSummary
          target={selectedTarget}
          refreshing={targets.isFetching}
          refresh={() => void targets.refetch()}
        />
      )}

      <div
        className="flex gap-2 border-b border-border"
        role="tablist"
        aria-label="Evals views"
        onKeyDown={moveTabFocus}
      >
        <Button
          ref={catalogTabRef}
          id="evals-tab-catalog"
          role="tab"
          aria-controls="evals-panel-catalog"
          aria-selected={activeTab === "catalog"}
          tabIndex={activeTab === "catalog" ? 0 : -1}
          variant="ghost"
          className="rounded-b-none"
          onClick={() => void showCatalog()}
        >
          <Database /> Catalog
        </Button>
        <Button
          ref={resultsTabRef}
          id="evals-tab-results"
          role="tab"
          aria-controls="evals-panel-results"
          aria-selected={activeTab === "results"}
          tabIndex={activeTab === "results" ? 0 : -1}
          variant="ghost"
          className="rounded-b-none"
          onClick={() => void showResults()}
        >
          <CheckCircle2 /> Results
        </Button>
        <Button
          ref={runsTabRef}
          id="evals-tab-runs"
          role="tab"
          aria-controls="evals-panel-runs"
          aria-selected={activeTab === "runs"}
          tabIndex={activeTab === "runs" ? 0 : -1}
          variant="ghost"
          className="rounded-b-none"
          onClick={() => void showRuns()}
        >
          <FlaskConical /> Runs
        </Button>
      </div>

      {mutationUnavailable && (
        <StateMessage className="rounded-lg border border-border bg-muted/30 py-3 text-left">
          Evals remain readable, but imports, launches, and cancellation are unavailable.{" "}
          {mutationUnavailable}
        </StateMessage>
      )}
      {actionError && (
        <StateMessage
          tone="danger"
          className="rounded-lg border border-destructive/30 py-3"
          role="alert"
        >
          {actionError}
        </StateMessage>
      )}
      {actionNotice && (
        <div
          className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-4 py-3 text-sm text-emerald-700 dark:text-emerald-300"
          role="status"
        >
          {actionNotice}
        </div>
      )}

      {targets.isError && (
        <QueryError
          message="Could not load the server-owned eval targets."
          retry={() => void targets.refetch()}
        />
      )}

      {targets.isLoading ? (
        <LoadingState label="Loading eval targets..." />
      ) : selectedTargetKey ? (
        <>
          <div
            id="evals-panel-catalog"
            role="tabpanel"
            aria-labelledby="evals-tab-catalog"
            hidden={activeTab !== "catalog"}
          >
            {activeTab === "catalog" && (
              <div className="space-y-6">
                <ScenarioCatalog
                  key={selectedTargetKey}
                  targetKey={selectedTargetKey}
                  pendingAction={pendingAction}
                  runAction={runAction}
                  mutateEnabled={mutateCapability.enabled}
                />
                <CatalogView
                  search={search}
                  target={selectedTarget}
                  updateSearch={updateSearch}
                  pendingAction={pendingAction}
                  runAction={runAction}
                  mutateEnabled={mutateCapability.enabled}
                />
              </div>
            )}
          </div>
          <div
            id="evals-panel-results"
            role="tabpanel"
            aria-labelledby="evals-tab-results"
            hidden={activeTab !== "results"}
          >
            {activeTab === "results" && resultsReady && (
              <ResultsView
                search={search}
                targetKey={selectedTargetKey}
                updateSearch={updateSearch}
                pendingAction={pendingAction}
                runAction={runAction}
                mutateEnabled={mutateCapability.enabled}
              />
            )}
            {activeTab === "results" && !resultsReady && (
              <StateMessage
                className="rounded-lg border border-border bg-muted/30 py-12"
                role="status"
              >
                <div className="font-medium text-foreground">
                  The evaluation result catalog is unavailable
                </div>
                <div className="mt-1">
                  {evalsReadinessReasonText(readiness.captured_result_persistence)}
                </div>
              </StateMessage>
            )}
          </div>
          <div
            id="evals-panel-runs"
            role="tabpanel"
            aria-labelledby="evals-tab-runs"
            hidden={activeTab !== "runs"}
          >
            {activeTab === "runs" && (
              <RunsView
                search={search}
                targetKey={selectedTargetKey}
                updateSearch={updateSearch}
                pendingAction={pendingAction}
                runAction={runAction}
                mutateEnabled={mutateCapability.enabled}
              />
            )}
          </div>
        </>
      ) : null}
    </Page>
  )
}

function EvalTargetSelector({
  targets,
  selectedTargetKey,
  loading,
  selectTarget,
}: {
  targets: EvalTarget[]
  selectedTargetKey: string | undefined
  loading: boolean
  selectTarget: (targetKey: string) => Promise<void>
}) {
  return (
    <label className="flex items-center gap-2 text-xs text-muted-foreground">
      Current app target
      <select
        value={selectedTargetKey ?? ""}
        disabled={loading || targets.length === 0}
        className="h-9 max-w-72 rounded-md border border-input bg-background px-2 text-sm text-foreground"
        aria-label="Current application eval target"
        title="Server-published target for the mounted app, including its current provider, tools, environment, approvals, and policy."
        onChange={(event) => void selectTarget(event.target.value)}
      >
        {loading && <option value="">Loading...</option>}
        {targets.map((target) => (
          <option key={target.target_key} value={target.target_key}>
            {target.agent_name} · {target.profile_id}
          </option>
        ))}
      </select>
    </label>
  )
}

function EvalExecutionProfileSummary({
  target,
  refreshing,
  refresh,
}: {
  target: EvalTarget
  refreshing: boolean
  refresh: () => void
}) {
  const profile = target.execution_profile
  return (
    <DataCard
      title="Current execution profile"
      description="Server-published candidate identity and authority that fresh Evals will freeze at launch."
      actions={
        <Button type="button" size="sm" variant="outline" disabled={refreshing} onClick={refresh}>
          <RotateCcw className={refreshing ? "animate-spin" : undefined} />
          Refresh profile
        </Button>
      }
      contentClassName="p-4"
    >
      {profile ? (
        <div className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
          <ProfileFact label="Candidate" value={profile.candidate.agent_name} />
          <ProfileFact
            label="Provider / model"
            value={`${profile.candidate.provider_name} / ${profile.candidate.model}`}
          />
          <ProfileFact
            label="Environment"
            value={profile.candidate.environment_name ?? "Application default"}
          />
          <ProfileFact
            label="Fixture / reset"
            value={`${executionProfileLabel(profile.fixture_strategy)} / ${executionProfileLabel(profile.reset_strategy)}`}
          />
          <ProfileFact label="Effects" value={executionProfileLabel(profile.effect_posture)} />
          <ProfileFact
            label="Maximum scale"
            value={`${profile.ceilings.max_cases} case${profile.ceilings.max_cases === 1 ? "" : "s"}, ${profile.ceilings.max_trials} trial${profile.ceilings.max_trials === 1 ? "" : "s"}, concurrency ${profile.ceilings.max_concurrency}`}
          />
          <ProfileFact
            label="Execution ceilings"
            value={`${profile.ceilings.max_timeout_seconds}s timeout · ${profile.ceilings.max_steps} steps · ${profile.ceilings.max_compiled_input_chars.toLocaleString()} compiled chars`}
          />
          <ProfileFact
            label="Evidence"
            value={`Public runtime projection · output up to ${(profile.evidence_policy.max_final_output_chars ?? 65_536).toLocaleString()} chars`}
          />
          <ProfileFact
            label="Runtime identity"
            value={profile.candidate.runtime_execution_profile_fingerprint.slice(0, 12)}
            mono
          />
          <ProfileFact
            label="Target material"
            value={`${profile.target_material.fingerprint.slice(0, 12)} · ${profile.target_material.kind === "structural_sha256" ? "structural" : "process-local"}`}
            mono
          />
          <ProfileFact label="Profile revision" value={shortEvalIdentity(profile.revision)} mono />
        </div>
      ) : (
        <div className="text-sm text-amber-700 dark:text-amber-300">
          {target.execution_profile_diagnostics?.[0]?.message ??
            "The current runtime execution profile is unavailable."}
        </div>
      )}
    </DataCard>
  )
}

function ProfileFact({
  label,
  value,
  mono = false,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div className="min-w-0 rounded-md border border-border bg-muted/20 p-3">
      <div className="text-muted-foreground">{label}</div>
      <div className={`mt-1 truncate text-foreground ${mono ? "font-mono" : "font-medium"}`}>
        {value}
      </div>
    </div>
  )
}

function executionProfileLabel(value: string): string {
  return value.replaceAll("_", " ")
}

function EvalsReadinessOverview({ readiness }: { readiness: EvalsReadiness }) {
  return (
    <DataCard
      title="Readiness"
      description="Embedded mounts need AuthenticatedAccess and EvalsConfig(target=..., store=...); cayu serve --dev assembles trusted local wiring. Routes still enforce runtime policy."
      contentClassName="p-4"
    >
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {EVALS_READINESS_OPERATIONS.map(([operationName, label]) => {
          const operation = readiness[operationName]
          return (
            <div
              key={operationName}
              className="min-w-0 rounded-md border border-border bg-muted/20 p-3"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="font-medium">{label}</div>
                <Badge variant={operation.state === "ready" ? "secondary" : "outline"}>
                  {evalsReadinessStateLabel(operation)}
                </Badge>
              </div>
              <p className="mt-2 text-xs text-muted-foreground">
                {evalsReadinessReasonText(operation)}
              </p>
            </div>
          )
        })}
      </div>
    </DataCard>
  )
}

function ScenarioCatalog({
  targetKey,
  pendingAction,
  runAction,
  mutateEnabled,
}: {
  targetKey: string
  pendingAction: string | null
  runAction: (
    name: string,
    action: (signal: AbortSignal) => Promise<string | undefined>,
  ) => Promise<void>
  mutateEnabled: boolean
}) {
  const [cursor, setCursor] = useState<string | undefined>()
  const [selectedRevision, setSelectedRevision] = useState<string | null>(null)
  const scenarios = useQuery({
    queryKey: ["evals", "scenarios", targetKey, cursor],
    queryFn: ({ signal }) =>
      fetchEvalScenarios({ target_key: targetKey, limit: 10, cursor }, signal),
  })
  useEffect(() => {
    if (
      selectedRevision === null ||
      !scenarios.data?.items.some((item) => item.revision === selectedRevision)
    ) {
      setSelectedRevision(scenarios.data?.items[0]?.revision ?? null)
    }
  }, [scenarios.data?.items, selectedRevision])
  const selected = useQuery({
    queryKey: ["evals", "scenario", selectedRevision],
    queryFn: ({ signal }) => fetchEvalScenario(selectedRevision ?? "", signal),
    enabled: selectedRevision !== null,
  })

  const download = (entry: EvalScenarioEntry) => {
    if (pendingAction !== null) return
    void runAction(`download-scenario:${entry.revision}`, async (signal) => {
      const file = await downloadEvalScenario(entry.revision, signal)
      if (signal.aborted) return
      downloadBlob(file.blob, file.filename)
      return `Downloaded scenario ${shortEvalIdentity(entry.revision)}.`
    })
  }

  return (
    <DataCard
      testId="scenario-catalog"
      title="Production scenarios"
      description="Immutable, editable multi-stage stimuli captured from production sessions."
    >
      <div className="grid min-w-0 gap-4 xl:grid-cols-[minmax(22rem,0.8fr)_minmax(0,1.4fr)]">
        <div className="min-w-0 border-r-0 border-border xl:border-r">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Scenario</TableHead>
                <TableHead>Stages</TableHead>
                <TableHead>Created</TableHead>
                <TableHead />
              </TableRow>
            </TableHeader>
            <TableBody>
              {scenarios.data?.items.map((entry) => (
                <TableRow
                  key={entry.revision}
                  data-state={selectedRevision === entry.revision ? "selected" : undefined}
                >
                  <TableCell>
                    <button
                      type="button"
                      className="max-w-56 truncate text-left font-medium text-primary hover:underline"
                      title={entry.name}
                      onClick={() => setSelectedRevision(entry.revision)}
                    >
                      {entry.name}
                    </button>
                    <div className="mt-1 font-mono text-xs text-muted-foreground">
                      {shortEvalIdentity(entry.revision)}
                    </div>
                  </TableCell>
                  <TableCell>
                    {formatCount(entry.event_count)} events
                    <div className="text-xs text-muted-foreground">
                      {formatCount(entry.artifact_requirement_count)} files ·{" "}
                      {formatCount(entry.secret_requirement_count)} secrets
                    </div>
                  </TableCell>
                  <TableCell>
                    {formatDateTime(entry.created_at)}
                    <div className="text-xs text-muted-foreground">
                      {formatBytes(entry.document_bytes)}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Button
                      type="button"
                      size="icon-xs"
                      variant="ghost"
                      aria-label={`Download scenario ${entry.name}`}
                      disabled={pendingAction !== null}
                      onClick={() => download(entry)}
                    >
                      {pendingAction === `download-scenario:${entry.revision}` ? (
                        <LoaderCircle className="animate-spin" />
                      ) : (
                        <Download />
                      )}
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {scenarios.isLoading ? (
            <LoadingState label="Loading scenarios..." />
          ) : scenarios.isError ? (
            <QueryError
              message="Could not load the scenario catalog."
              retry={() => void scenarios.refetch()}
            />
          ) : scenarios.data?.items.length === 0 ? (
            <StateMessage>
              No scenarios yet. Open a completed session, choose Evaluate, and save its production
              scenario.
            </StateMessage>
          ) : null}
          <PageControls
            scope="scenario catalog"
            cursor={cursor}
            nextCursor={scenarios.data?.next_cursor}
            fetching={scenarios.isFetching}
            first={() => {
              setCursor(undefined)
              setSelectedRevision(null)
            }}
            next={(nextCursor) => {
              setCursor(nextCursor)
              setSelectedRevision(null)
            }}
          />
        </div>
        <div className="min-w-0 p-3">
          {selected.isLoading ? (
            <LoadingState label="Loading scenario..." />
          ) : selected.isError ? (
            <QueryError
              message="Could not load the selected scenario."
              retry={() => void selected.refetch()}
            />
          ) : selected.data ? (
            <ScenarioAuthoring captured={selected.data} disabled={!mutateEnabled} saved />
          ) : (
            <StateMessage>Select a scenario revision to inspect and edit it.</StateMessage>
          )}
        </div>
      </div>
    </DataCard>
  )
}

function CatalogView({
  search,
  target,
  updateSearch,
  pendingAction,
  runAction,
  mutateEnabled,
}: {
  search: EvalsSearch
  target: EvalTarget | undefined
  updateSearch: UpdateEvalsSearch
  pendingAction: string | null
  runAction: (
    name: string,
    action: (signal: AbortSignal) => Promise<string | undefined>,
  ) => Promise<void>
  mutateEnabled: boolean
}) {
  const targetKey = target?.target_key ?? ""
  const queryClient = useQueryClient()
  const [maxConcurrency, setMaxConcurrency] = useState("1")
  const launchRegistryRef = useRef<EvalLaunchIdempotencyRegistry | null>(null)
  const corpora = useQuery({
    queryKey: ["evals", "corpora", targetKey, search.corpora_cursor],
    queryFn: ({ signal }) =>
      fetchEvalCorpora(
        { target_key: targetKey, limit: PAGE_LIMIT, cursor: search.corpora_cursor },
        signal,
      ),
    enabled: target !== undefined,
  })
  const suites = useQuery({
    queryKey: ["evals", "suites", search.corpus, search.suites_cursor],
    queryFn: ({ signal }) =>
      fetchEvalSuites(
        search.corpus ?? "",
        { limit: PAGE_LIMIT, cursor: search.suites_cursor },
        signal,
      ),
    enabled: search.corpus !== undefined,
  })
  const cases = useQuery({
    queryKey: ["evals", "cases", search.corpus, search.suite, search.cases_cursor],
    queryFn: ({ signal }) =>
      fetchEvalCases(
        search.corpus ?? "",
        search.suite ?? "",
        { limit: PAGE_LIMIT, cursor: search.cases_cursor },
        signal,
      ),
    enabled: search.corpus !== undefined && search.suite !== undefined,
  })

  if (target === undefined) {
    return <StateMessage>The selected eval target is unavailable.</StateMessage>
  }

  const selectCorpus = (corpus: EvalCorpusEntry) => {
    updateSearch((current) => ({
      ...evalsSearchWithout(current, "suite", "suites_cursor", "cases_cursor"),
      tab: "catalog",
      corpus: corpus.revision,
    }))
  }
  const selectedCorpus = corpora.data?.items.find((item) => item.revision === search.corpus)
  const parsedConcurrency = Number(maxConcurrency)
  const concurrencyIsValid =
    Number.isInteger(parsedConcurrency) &&
    parsedConcurrency >= 1 &&
    parsedConcurrency <= target.max_concurrency

  const downloadCorpus = () => {
    if (!search.corpus || pendingAction !== null) return
    void runAction("download-corpus", async (signal) => {
      const file = await downloadEvalCorpus(search.corpus ?? "", signal)
      if (signal.aborted) return
      downloadBlob(file.blob, file.filename)
      return `Downloaded corpus ${shortEvalIdentity(search.corpus ?? "")}.`
    })
  }

  const launchSuite = (suiteId: string) => {
    if (
      !search.corpus ||
      pendingAction !== null ||
      !mutateEnabled ||
      !concurrencyIsValid ||
      !target.execution_profile_ready ||
      target.execution_profile == null
    ) {
      return
    }
    const executionProfile = target.execution_profile
    const requestIdentity = evalLaunchRequestIdentity(
      search.corpus,
      suiteId,
      parsedConcurrency,
      executionProfile.revision,
    )
    void runAction(`launch:${suiteId}`, async (signal) => {
      const registry =
        launchRegistryRef.current ??
        new EvalLaunchIdempotencyRegistry(window.sessionStorage, dashboardConfig.apiBaseUrl)
      launchRegistryRef.current = registry
      const idempotencyKey = registry.keyFor(requestIdentity)
      let run: EvalRun
      try {
        run = await createEvalRun(
          {
            corpus_revision: search.corpus ?? "",
            suite_id: suiteId,
            expected_execution_profile_revision: executionProfile.revision,
            max_concurrency: parsedConcurrency,
          },
          idempotencyKey,
          signal,
        )
      } catch (error) {
        if (evalTargetCatalogMayBeStale(error)) {
          await queryClient.invalidateQueries({ queryKey: EVAL_TARGET_QUERY_KEY })
        }
        if (error instanceof ApiClientError && evalLaunchFailureIsDefinitive(error.status)) {
          registry.resolve(requestIdentity)
        }
        throw error
      }
      if (signal.aborted) return
      queryClient.setQueryData(["evals", "run", run.spec.run_id], run)
      await queryClient.invalidateQueries({ queryKey: ["evals", "runs"] })
      if (signal.aborted) return
      await updateSearch((current) => ({
        ...evalsSearchWithout(
          current,
          "suite",
          "suites_cursor",
          "cases_cursor",
          "runs_cursor",
          "status",
          "baseline",
        ),
        tab: "runs",
        corpus: run.spec.corpus_revision,
        run: run.spec.run_id,
      }))
      if (signal.aborted) return
      registry.resolve(requestIdentity)
      return evalLaunchNotice(run)
    })
  }

  return (
    <div className="grid min-w-0 gap-6 xl:grid-cols-[minmax(22rem,0.9fr)_minmax(0,1.4fr)]">
      <DataCard
        title="Corpora"
        description={
          corpora.isLoading
            ? "Loading a bounded catalog page..."
            : `${formatCount(corpora.data?.items.length)} revisions on this page${search.corpora_cursor ? " · later page" : " · first page"}`
        }
      >
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Revision</TableHead>
              <TableHead>Content</TableHead>
              <TableHead>Created</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {corpora.data?.items.map((corpus) => (
              <TableRow
                key={corpus.revision}
                data-state={search.corpus === corpus.revision ? "selected" : undefined}
              >
                <TableCell>
                  <button
                    type="button"
                    className="max-w-44 truncate text-left font-mono text-xs text-primary hover:underline"
                    title={corpus.revision}
                    onClick={() => selectCorpus(corpus)}
                  >
                    {shortEvalIdentity(corpus.revision)}
                  </button>
                  <div className="mt-1 text-xs text-muted-foreground">{corpus.target_key}</div>
                </TableCell>
                <TableCell>
                  <div>{formatCount(corpus.suite_count)} suites</div>
                  <div className="text-xs text-muted-foreground">
                    {formatCount(corpus.case_count)} cases · {formatCount(corpus.assertion_count)}{" "}
                    assertions
                  </div>
                </TableCell>
                <TableCell>
                  <div>{formatDateTime(corpus.created_at)}</div>
                  <div className="text-xs text-muted-foreground">
                    {formatBytes(corpus.document_bytes)}
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {corpora.isLoading ? (
          <LoadingState label="Loading corpora..." />
        ) : corpora.isError ? (
          <QueryError
            message="Could not load the eval catalog."
            retry={() => void corpora.refetch()}
          />
        ) : corpora.data?.items.length === 0 ? (
          <StateMessage>No eval corpora have been saved or imported yet.</StateMessage>
        ) : null}
        <PageControls
          scope="corpus catalog"
          cursor={search.corpora_cursor}
          nextCursor={corpora.data?.next_cursor}
          fetching={corpora.isFetching}
          first={() =>
            updateSearch((current) => ({
              ...evalsSearchWithout(
                current,
                "corpora_cursor",
                "corpus",
                "suite",
                "suites_cursor",
                "cases_cursor",
              ),
              tab: "catalog",
            }))
          }
          next={(cursor) =>
            updateSearch((current) => ({
              ...evalsSearchWithout(current, "corpus", "suite", "suites_cursor", "cases_cursor"),
              tab: "catalog",
              corpora_cursor: cursor,
            }))
          }
        />
      </DataCard>

      <div className="min-w-0 space-y-6">
        {!search.corpus ? (
          <StateMessage className="rounded-lg border border-border bg-muted/30 py-16">
            Select a corpus revision to browse its suites and cases.
          </StateMessage>
        ) : (
          <>
            <DataCard
              title={
                <span className="flex items-center gap-2">
                  Corpus {shortEvalIdentity(search.corpus)}
                  {selectedCorpus && <Badge variant="outline">{selectedCorpus.target_key}</Badge>}
                </span>
              }
              description={
                selectedCorpus
                  ? `${formatCount(selectedCorpus.expanded_assertion_result_count)} expanded assertion results · ${formatBytes(selectedCorpus.document_bytes)}`
                  : "Selected catalog revision"
              }
              actions={
                <div className="flex items-center gap-2">
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    Concurrency
                    <input
                      type="number"
                      min={1}
                      max={target.max_concurrency}
                      step={1}
                      value={maxConcurrency}
                      className="h-8 w-16 rounded-md border border-input bg-background px-2 text-sm"
                      aria-invalid={!concurrencyIsValid}
                      onChange={(event) => setMaxConcurrency(event.target.value)}
                    />
                  </label>
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={pendingAction !== null}
                    onClick={downloadCorpus}
                  >
                    {pendingAction === "download-corpus" ? (
                      <LoaderCircle className="animate-spin" />
                    ) : (
                      <Download />
                    )}
                    Download JSON
                  </Button>
                </div>
              }
            >
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Suite</TableHead>
                    <TableHead>Trials</TableHead>
                    <TableHead>Coverage</TableHead>
                    <TableHead>Run</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {suites.data?.items.map((suite) => (
                    <TableRow
                      key={suite.id}
                      data-state={search.suite === suite.id ? "selected" : undefined}
                    >
                      <TableCell>
                        <button
                          type="button"
                          className="max-w-72 truncate text-left font-medium text-primary hover:underline"
                          title={suite.name}
                          onClick={() =>
                            updateSearch((current) => ({
                              ...evalsSearchWithout(current, "cases_cursor"),
                              tab: "catalog",
                              corpus: search.corpus,
                              suite: suite.id,
                            }))
                          }
                        >
                          {suite.name}
                        </button>
                        <div className="mt-1 max-w-72 truncate text-xs text-muted-foreground">
                          {suite.description ?? suite.id}
                        </div>
                      </TableCell>
                      <TableCell>
                        {formatCount(suite.trials)} × {formatCount(suite.timeout_seconds)}s
                      </TableCell>
                      <TableCell>
                        {formatCount(suite.case_count)} cases · {formatCount(suite.assertion_count)}{" "}
                        assertions
                      </TableCell>
                      <TableCell>
                        <Button
                          type="button"
                          size="sm"
                          aria-label={`Run suite ${suite.name} (${suite.id}) on current app`}
                          disabled={
                            !mutateEnabled ||
                            !concurrencyIsValid ||
                            !target.execution_profile_ready ||
                            pendingAction !== null
                          }
                          title={
                            target.execution_profile_ready
                              ? undefined
                              : (target.execution_profile_diagnostics?.[0]?.message ??
                                "The current execution profile is unavailable.")
                          }
                          onClick={() => launchSuite(suite.id)}
                        >
                          {pendingAction === `launch:${suite.id}` ? (
                            <LoaderCircle className="animate-spin" />
                          ) : (
                            <Play />
                          )}
                          Run on current app
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              {suites.isLoading ? (
                <LoadingState label="Loading suites..." />
              ) : suites.isError ? (
                <QueryError
                  message="Could not load corpus suites."
                  retry={() => void suites.refetch()}
                />
              ) : suites.data?.items.length === 0 ? (
                <StateMessage>This corpus contains no suites.</StateMessage>
              ) : null}
              <PageControls
                scope="corpus suites"
                cursor={search.suites_cursor}
                nextCursor={suites.data?.next_cursor}
                fetching={suites.isFetching}
                first={() =>
                  updateSearch((current) =>
                    evalsSearchWithout(current, "suites_cursor", "suite", "cases_cursor"),
                  )
                }
                next={(cursor) =>
                  updateSearch((current) => ({
                    ...evalsSearchWithout(current, "suite", "cases_cursor"),
                    suites_cursor: cursor,
                  }))
                }
              />
            </DataCard>

            {search.suite && (
              <DataCard
                title="Cases"
                description={`${formatCount(cases.data?.items.length)} cases on this bounded page`}
              >
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Case</TableHead>
                      <TableHead>Input</TableHead>
                      <TableHead>Assertions</TableHead>
                      <TableHead>Revision</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {cases.data?.items.map((evalCase) => (
                      <TableRow key={evalCase.id}>
                        <TableCell>
                          <div className="max-w-64 truncate font-medium" title={evalCase.name}>
                            {evalCase.name}
                          </div>
                          <div className="mt-1 max-w-64 truncate text-xs text-muted-foreground">
                            {evalCase.description ?? evalCase.id}
                          </div>
                        </TableCell>
                        <TableCell>{formatCount(evalCase.message_count)} messages</TableCell>
                        <TableCell>{formatCount(evalCase.assertion_count)}</TableCell>
                        <TableCell className="font-mono text-xs" title={evalCase.revision}>
                          {shortEvalIdentity(evalCase.revision)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                {cases.isLoading ? (
                  <LoadingState label="Loading cases..." />
                ) : cases.isError ? (
                  <QueryError
                    message="Could not load suite cases."
                    retry={() => void cases.refetch()}
                  />
                ) : cases.data?.items.length === 0 ? (
                  <StateMessage>This suite contains no cases.</StateMessage>
                ) : null}
                <PageControls
                  scope="suite cases"
                  cursor={search.cases_cursor}
                  nextCursor={cases.data?.next_cursor}
                  fetching={cases.isFetching}
                  first={() =>
                    updateSearch((current) => evalsSearchWithout(current, "cases_cursor"))
                  }
                  next={(cursor) =>
                    updateSearch((current) => ({ ...current, cases_cursor: cursor }))
                  }
                />
              </DataCard>
            )}
          </>
        )}
      </div>
    </div>
  )
}

function ResultsView({
  search,
  targetKey,
  updateSearch,
  pendingAction,
  runAction,
  mutateEnabled,
}: {
  search: EvalsSearch
  targetKey: string
  updateSearch: UpdateEvalsSearch
  pendingAction: string | null
  runAction: (
    name: string,
    action: (signal: AbortSignal) => Promise<string | undefined>,
  ) => Promise<void>
  mutateEnabled: boolean
}) {
  const queryClient = useQueryClient()
  const results = useQuery({
    queryKey: ["evals", "results", targetKey, search.results_cursor],
    queryFn: ({ signal }) =>
      fetchEvalResults(
        { target_key: targetKey, limit: PAGE_LIMIT, cursor: search.results_cursor },
        signal,
      ),
  })
  const detail = useQuery({
    queryKey: ["evals", "result-detail", targetKey, search.result],
    queryFn: ({ signal }) => fetchEvalResultDetail(search.result ?? "", signal),
    enabled: search.result !== undefined,
  })
  const currentResultRevision = detail.data?.record.revision
  const approvedBaselineRevision = detail.data?.baseline?.result_revision
  const baselineResultRevision = search.baseline ?? approvedBaselineRevision
  const comparison = useQuery({
    queryKey: ["evals", "result-comparison", baselineResultRevision, currentResultRevision],
    queryFn: ({ signal }) =>
      compareEvalResults(baselineResultRevision ?? "", currentResultRevision ?? "", 0, signal),
    enabled:
      baselineResultRevision !== undefined &&
      currentResultRevision !== undefined &&
      baselineResultRevision !== currentResultRevision,
    retry: retryEvalQuery,
    staleTime: Number.POSITIVE_INFINITY,
  })

  const approveBaseline = (selected: EvalResultDetail) => {
    if (pendingAction !== null || !mutateEnabled) return
    const revision = selected.record.revision
    void runAction(`baseline:${revision}`, async (signal) => {
      await selectEvalBaseline(
        revision,
        {
          result_revision: revision,
          expected_generation: selected.baseline?.generation ?? 0,
          operation_id: randomEvalOperationId(),
        },
        signal,
      )
      if (signal.aborted) return
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: ["evals", "result-detail", targetKey, revision],
        }),
        queryClient.invalidateQueries({ queryKey: ["evals", "results"] }),
      ])
      return `Approved result ${shortEvalIdentity(revision)} as the suite baseline.`
    })
  }

  const downloadResult = (revision: string, format: "json" | "html") => {
    if (pendingAction !== null) return
    void runAction(`download-catalog-result-${format}`, async (signal) => {
      const file =
        format === "json"
          ? await downloadCatalogEvalResultJson(revision, signal)
          : await downloadCatalogEvalResultHtml(revision, signal)
      if (signal.aborted) return
      downloadBlob(file.blob, file.filename)
      return `Downloaded the ${format.toUpperCase()} report for ${shortEvalIdentity(revision)}.`
    })
  }

  return (
    <div className="grid min-w-0 gap-6 xl:grid-cols-[minmax(24rem,0.95fr)_minmax(0,1.3fr)]">
      <DataCard
        title="Evaluation results"
        description={`${formatCount(results.data?.items.length)} immutable results on this page${search.results_cursor ? " · later page" : " · first page"}`}
        actions={
          <Button
            type="button"
            size="sm"
            variant="ghost"
            disabled={results.isFetching}
            onClick={() => void results.refetch()}
          >
            <RotateCcw className={results.isFetching ? "animate-spin" : undefined} /> Refresh
          </Button>
        }
      >
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Result</TableHead>
              <TableHead>Origin</TableHead>
              <TableHead>Outcome</TableHead>
              <TableHead>Created</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {results.data?.items.map((result) => (
              <TableRow
                key={result.revision}
                data-state={search.result === result.revision ? "selected" : undefined}
              >
                <TableCell>
                  <button
                    type="button"
                    className="max-w-44 truncate text-left font-mono text-xs text-primary hover:underline"
                    title={result.revision}
                    onClick={() =>
                      updateSearch((current) => ({
                        ...evalsSearchWithout(current, "baseline"),
                        tab: "results",
                        result: result.revision,
                        corpus: result.corpus_revision,
                      }))
                    }
                  >
                    {shortEvalIdentity(result.revision)}
                  </button>
                  <div className="mt-1 max-w-44 truncate text-xs text-muted-foreground">
                    {result.suite_id}
                  </div>
                </TableCell>
                <TableCell>
                  <Badge variant="outline">
                    {result.origin === "captured_session" ? "Captured" : "Current app"}
                  </Badge>
                  <div className="mt-1 max-w-40 truncate text-xs text-muted-foreground">
                    {result.target.application_release_id}
                  </div>
                </TableCell>
                <TableCell>
                  <OutcomeBadge outcome={result.status} />
                  <div className="mt-1 text-xs text-muted-foreground">
                    {formatScore(result.score)}
                  </div>
                </TableCell>
                <TableCell>{formatDateTime(result.created_at)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {results.isLoading ? (
          <LoadingState label="Loading evaluation results..." />
        ) : results.isError ? (
          <QueryError
            message="Could not load the evaluation result catalog."
            retry={() => void results.refetch()}
          />
        ) : results.data?.items.length === 0 ? (
          <StateMessage>
            No results yet. Open a completed session and choose Evaluate to capture one.
          </StateMessage>
        ) : null}
        <PageControls
          scope="evaluation results"
          cursor={search.results_cursor}
          nextCursor={results.data?.next_cursor}
          fetching={results.isFetching}
          first={() =>
            updateSearch((current) =>
              evalsSearchWithout(current, "results_cursor", "result", "baseline"),
            )
          }
          next={(cursor) =>
            updateSearch((current) => ({
              ...evalsSearchWithout(current, "result", "baseline"),
              tab: "results",
              results_cursor: cursor,
            }))
          }
        />
      </DataCard>

      {!search.result ? (
        <StateMessage className="rounded-lg border border-border bg-muted/30 py-16">
          Select an immutable result to inspect its score and baseline status.
        </StateMessage>
      ) : detail.isLoading ? (
        <LoadingState label="Loading evaluation result..." />
      ) : detail.isError ? (
        <DataCard title="Result unavailable">
          <QueryError
            message="Could not load the selected evaluation result."
            retry={() => void detail.refetch()}
          />
        </DataCard>
      ) : detail.data && detail.data.record.target.target_key !== targetKey ? (
        <StateMessage className="rounded-lg border border-border bg-muted/30 py-16">
          The selected result does not belong to this eval target.
        </StateMessage>
      ) : detail.data ? (
        <div className="min-w-0 space-y-6">
          <CapturedResultInspector
            detail={detail.data}
            approving={pendingAction === `baseline:${detail.data.record.revision}`}
            downloading={pendingAction?.startsWith("download-catalog-result-") === true}
            canMutate={mutateEnabled}
            approve={() => approveBaseline(detail.data)}
            download={(format) => downloadResult(detail.data.record.revision, format)}
            openCorpus={() =>
              updateSearch((current) => ({
                ...evalsSearchWithout(current, "result", "results_cursor", "baseline"),
                tab: "catalog",
                target: detail.data.record.target.target_key,
                corpus: detail.data.record.corpus_revision,
              }))
            }
          />
          <ComparisonPanel
            key={`${detail.data.record.revision}:${baselineResultRevision ?? ""}`}
            currentResultRevision={detail.data.record.revision}
            baselineResultRevision={baselineResultRevision}
            approvedBaselineRevision={approvedBaselineRevision}
            candidates={results.data?.items ?? []}
            comparison={comparison.data}
            loading={comparison.isLoading && comparison.fetchStatus === "fetching"}
            error={comparison.error}
            retry={() => void comparison.refetch()}
            selectBaseline={(baseline) =>
              updateSearch((current) => ({
                ...evalsSearchWithout(current, "baseline"),
                ...(baseline ? { baseline } : {}),
              }))
            }
          />
        </div>
      ) : null}
    </div>
  )
}

function CapturedResultInspector({
  detail,
  approving,
  downloading,
  canMutate,
  approve,
  download,
  openCorpus,
}: {
  detail: EvalResultDetail
  approving: boolean
  downloading: boolean
  canMutate: boolean
  approve: () => void
  download: (format: "json" | "html") => void
  openCorpus: () => void
}) {
  const selectedAsBaseline = detail.baseline?.result_revision === detail.record.revision
  return (
    <div className="min-w-0 space-y-6">
      <DataCard
        title={
          <span className="flex items-center gap-2">
            Result {shortEvalIdentity(detail.record.revision)}
            <OutcomeBadge outcome={detail.record.status} />
            {selectedAsBaseline && <Badge variant="secondary">Baseline</Badge>}
          </span>
        }
        description={`${detail.record.origin === "captured_session" ? "Captured session" : "Current-app execution"} · release ${detail.record.target.application_release_id}`}
        actions={
          <>
            <Button
              type="button"
              size="sm"
              variant="outline"
              disabled={downloading}
              onClick={() => download("json")}
            >
              <FileJson /> JSON
            </Button>
            <Button
              type="button"
              size="sm"
              variant="outline"
              disabled={downloading}
              onClick={() => download("html")}
            >
              <Download /> HTML
            </Button>
            <Button type="button" size="sm" variant="outline" onClick={openCorpus}>
              <Database /> Open corpus
            </Button>
            {!selectedAsBaseline && (
              <Button type="button" size="sm" disabled={!canMutate || approving} onClick={approve}>
                {approving ? <LoaderCircle className="animate-spin" /> : <CheckCircle2 />}
                {approving ? "Approving..." : "Approve baseline"}
              </Button>
            )}
          </>
        }
      >
        <div className="grid gap-4 p-4 text-sm sm:grid-cols-2 lg:grid-cols-4">
          <RunFact label="Score" value={formatScore(detail.record.score)} />
          <RunFact label="Suite" value={detail.record.suite_id} />
          <RunFact label="Corpus" value={shortEvalIdentity(detail.record.corpus_revision)} />
          <RunFact label="Created" value={formatDateTime(detail.record.created_at)} />
        </div>
        {detail.baseline && (
          <div className="border-t border-border px-4 py-3 text-xs text-muted-foreground">
            Baseline generation {formatCount(detail.baseline.generation)} · selected by{" "}
            {detail.baseline.updated_by} · {formatDateTime(detail.baseline.updated_at)}
          </div>
        )}
      </DataCard>
      <DataCard
        title="Explainable result"
        description="Canonical public-safe outcome, evaluator, criterion, evidence, usage, and cost facts."
        contentClassName="p-4"
      >
        <ResultPresentationInspector presentation={detail.presentation} />
        <details className="mt-4 border-t border-border pt-4 text-xs">
          <summary className="cursor-pointer text-primary">Inspect immutable JSON evidence</summary>
          <PayloadViewer value={detail.result} className="mt-3" maxHeight="max-h-[36rem]" />
        </details>
      </DataCard>
    </div>
  )
}

function RunsView({
  search,
  targetKey,
  updateSearch,
  pendingAction,
  runAction,
  mutateEnabled,
}: {
  search: EvalsSearch
  targetKey: string
  updateSearch: UpdateEvalsSearch
  pendingAction: string | null
  runAction: (
    name: string,
    action: (signal: AbortSignal) => Promise<string | undefined>,
  ) => Promise<void>
  mutateEnabled: boolean
}) {
  const queryClient = useQueryClient()
  const runs = useQuery({
    queryKey: ["evals", "runs", targetKey, search.status, search.corpus, search.runs_cursor],
    queryFn: ({ signal }) =>
      fetchEvalRuns(
        {
          limit: PAGE_LIMIT,
          target_key: targetKey,
          cursor: search.runs_cursor,
          status: search.status,
          corpus_revision: search.corpus,
        },
        signal,
      ),
    refetchInterval: (query) =>
      query.state.data?.items.some((run) => evalRunIsActive(run)) ? 3_000 : false,
    refetchIntervalInBackground: false,
  })
  const selectedRun = useQuery({
    queryKey: ["evals", "run", search.run],
    queryFn: ({ signal }) => fetchEvalRun(search.run ?? "", signal),
    enabled: search.run !== undefined,
    refetchInterval: (query) => (evalRunIsActive(query.state.data) ? 1_500 : false),
    refetchIntervalInBackground: false,
  })
  const result = useQuery({
    queryKey: ["evals", "result", search.run],
    queryFn: ({ signal }) => fetchEvalResult(search.run ?? "", signal),
    enabled: evalRunHasResult(selectedRun.data),
    ...EVAL_RESULT_QUERY_RETENTION,
  })
  const currentResultRevision = result.data?.result.revision
  const comparisonResults = useQuery({
    queryKey: ["evals", "results", targetKey, { scope: "comparison-candidates" }],
    queryFn: ({ signal }) => fetchEvalResults({ target_key: targetKey, limit: PAGE_LIMIT }, signal),
    enabled: currentResultRevision !== undefined,
  })
  const approvedBaselineRevision = result.data?.baseline?.result_revision
  const baselineResultRevision = search.baseline ?? approvedBaselineRevision
  const comparison = useQuery({
    queryKey: ["evals", "result-comparison", baselineResultRevision, currentResultRevision],
    queryFn: ({ signal }) =>
      compareEvalResults(baselineResultRevision ?? "", currentResultRevision ?? "", 0, signal),
    enabled:
      result.data !== undefined &&
      baselineResultRevision !== undefined &&
      currentResultRevision !== undefined &&
      baselineResultRevision !== currentResultRevision,
    retry: retryEvalQuery,
    staleTime: Number.POSITIVE_INFINITY,
  })
  const comparisonCandidates =
    comparisonResults.data?.items.filter((item) => item.revision !== currentResultRevision) ?? []
  const selectedRunUpdatedAt = selectedRun.data?.updated_at

  useEffect(() => {
    if (!selectedRunUpdatedAt) return
    void queryClient.invalidateQueries({ queryKey: ["evals", "runs"] })
  }, [queryClient, selectedRunUpdatedAt])

  const cancelRun = () => {
    const run = selectedRun.data
    if (!run || !evalRunCanCancel(run) || pendingAction !== null || !mutateEnabled) return
    void runAction("cancel-run", async (signal) => {
      const cancelled = await cancelEvalRun(run.spec.run_id, signal)
      if (signal.aborted) return
      queryClient.setQueryData(["evals", "run", run.spec.run_id], cancelled)
      await queryClient.invalidateQueries({ queryKey: ["evals", "runs"] })
      if (signal.aborted) return
      return evalCancellationNotice(cancelled)
    })
  }

  const decideScenarioApproval = (
    trial: EvalScenarioTrialProgress,
    decision: "approve" | "deny",
  ) => {
    const run = selectedRun.data
    const progress = run?.scenario_progress
    if (
      !run ||
      !progress ||
      trial.phase !== "awaiting_approval" ||
      !trial.pending_event_id ||
      trial.approval ||
      pendingAction !== null ||
      !mutateEnabled
    ) {
      return
    }
    const actionName = `scenario-approval:${trial.trial_number}:${decision}`
    void runAction(actionName, async (signal) => {
      const updated = await submitEvalScenarioApproval(
        run.spec.run_id,
        {
          expected_progress_revision: progress.revision,
          trial_number: trial.trial_number,
          event_id: trial.pending_event_id ?? "",
          decision,
        },
        signal,
      )
      if (signal.aborted) return
      queryClient.setQueryData(["evals", "run", run.spec.run_id], updated)
      await queryClient.invalidateQueries({ queryKey: ["evals", "runs"] })
      return `${decision === "approve" ? "Approved" : "Denied"} trial ${trial.trial_number}'s fresh tool request.`
    })
  }

  const downloadResult = (format: "json" | "html") => {
    const runId = selectedRun.data?.spec.run_id
    if (!runId || !result.data || pendingAction !== null) return
    void runAction(`download-result-${format}`, async (signal) => {
      const file =
        format === "json"
          ? await downloadEvalResultJson(runId, signal)
          : await downloadEvalResultHtml(runId, signal)
      if (signal.aborted) return
      downloadBlob(file.blob, file.filename)
      return `Downloaded the ${format.toUpperCase()} report for ${shortEvalIdentity(runId)}.`
    })
  }

  return (
    <div className="grid min-w-0 gap-6 xl:grid-cols-[minmax(24rem,0.95fr)_minmax(0,1.3fr)]">
      <DataCard
        title="Durable runs"
        description={`${formatCount(runs.data?.items.length)} runs on this page${search.runs_cursor ? " · later page" : " · first page"}`}
      >
        <div className="flex flex-wrap items-center gap-2 border-b border-border p-3">
          <select
            value={search.status ?? "all"}
            className="h-8 rounded-md border border-input bg-background px-2 text-sm"
            aria-label="Filter eval runs by status"
            onChange={(event) =>
              updateSearch((current) => ({
                ...evalsSearchWithout(current, "status", "runs_cursor", "run", "baseline"),
                tab: "runs",
                ...(event.target.value === "all"
                  ? {}
                  : { status: event.target.value as EvalStatus }),
              }))
            }
          >
            <option value="all">All statuses</option>
            {(["queued", "running", "cancelling", "completed", "failed", "cancelled"] as const).map(
              (status) => (
                <option key={status} value={status}>
                  {status}
                </option>
              ),
            )}
          </select>
          {search.corpus && (
            <Button
              type="button"
              size="sm"
              variant="outline"
              title={search.corpus}
              onClick={() =>
                updateSearch((current) =>
                  evalsSearchWithout(current, "corpus", "runs_cursor", "run", "baseline"),
                )
              }
            >
              Corpus {shortEvalIdentity(search.corpus)} ×
            </Button>
          )}
          <Button
            type="button"
            size="sm"
            variant="ghost"
            disabled={runs.isFetching}
            onClick={() => void runs.refetch()}
          >
            <RotateCcw className={runs.isFetching ? "animate-spin" : undefined} /> Refresh
          </Button>
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Run</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Suite</TableHead>
              <TableHead>Updated</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {runs.data?.items.map((run) => (
              <TableRow
                key={run.spec.run_id}
                data-state={search.run === run.spec.run_id ? "selected" : undefined}
              >
                <TableCell>
                  <button
                    type="button"
                    className="max-w-44 truncate text-left font-mono text-xs text-primary hover:underline"
                    title={run.spec.run_id}
                    onClick={() =>
                      updateSearch((current) => ({
                        ...evalsSearchWithout(current, "baseline"),
                        tab: "runs",
                        run: run.spec.run_id,
                      }))
                    }
                  >
                    {shortEvalIdentity(run.spec.run_id)}
                  </button>
                </TableCell>
                <TableCell>
                  <EvalStatusBadge run={run} />
                </TableCell>
                <TableCell>
                  <div className="max-w-40 truncate" title={run.spec.suite_id}>
                    {run.spec.suite_id}
                  </div>
                  {run.result && (
                    <div className="mt-1 text-xs text-muted-foreground">
                      {run.result.status} · {formatScore(run.result.score)}
                    </div>
                  )}
                </TableCell>
                <TableCell>{formatDateTime(run.updated_at)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {runs.isLoading ? (
          <LoadingState label="Loading eval runs..." />
        ) : runs.isError ? (
          <QueryError
            message="Could not load durable eval runs."
            retry={() => void runs.refetch()}
          />
        ) : runs.data?.items.length === 0 ? (
          <StateMessage>No eval runs match these filters.</StateMessage>
        ) : null}
        <PageControls
          scope="eval runs"
          cursor={search.runs_cursor}
          nextCursor={runs.data?.next_cursor}
          fetching={runs.isFetching}
          first={() =>
            updateSearch((current) => evalsSearchWithout(current, "runs_cursor", "run", "baseline"))
          }
          next={(cursor) =>
            updateSearch((current) => ({
              ...evalsSearchWithout(current, "run", "baseline"),
              runs_cursor: cursor,
            }))
          }
        />
      </DataCard>

      {!search.run ? (
        <StateMessage className="rounded-lg border border-border bg-muted/30 py-16">
          Select a durable run to follow its status and inspect its result.
        </StateMessage>
      ) : selectedRun.isLoading ? (
        <LoadingState label="Loading eval run..." />
      ) : selectedRun.isError ? (
        <DataCard title="Run unavailable">
          <QueryError
            message="Could not load the selected eval run."
            retry={() => void selectedRun.refetch()}
          />
        </DataCard>
      ) : selectedRun.data ? (
        <div className="min-w-0 space-y-6">
          <RunLifecycleCard
            run={selectedRun.data}
            cancelling={pendingAction === "cancel-run"}
            pendingAction={pendingAction}
            canMutate={mutateEnabled}
            cancel={cancelRun}
            decideScenarioApproval={decideScenarioApproval}
          />
          {evalRunHasResult(selectedRun.data) &&
            (result.isLoading ? (
              <LoadingState label="Loading the published eval result..." />
            ) : result.isError ? (
              <DataCard title="Result unavailable">
                <QueryError
                  message="Could not load the published eval result."
                  retry={() => void result.refetch()}
                />
              </DataCard>
            ) : result.data ? (
              <>
                <ResultInspector
                  result={result.data}
                  pendingAction={pendingAction}
                  download={downloadResult}
                />
                <ComparisonPanel
                  key={`${currentResultRevision}:${baselineResultRevision ?? ""}`}
                  currentResultRevision={currentResultRevision ?? ""}
                  baselineResultRevision={baselineResultRevision}
                  approvedBaselineRevision={approvedBaselineRevision}
                  candidates={comparisonCandidates}
                  comparison={comparison.data}
                  loading={comparison.isLoading && comparison.fetchStatus === "fetching"}
                  error={comparison.error}
                  retry={() => void comparison.refetch()}
                  selectBaseline={(baseline) =>
                    updateSearch((current) => ({
                      ...evalsSearchWithout(current, "baseline"),
                      ...(baseline ? { baseline } : {}),
                    }))
                  }
                />
              </>
            ) : null)}
        </div>
      ) : null}
    </div>
  )
}

function RunLifecycleCard({
  run,
  cancelling,
  pendingAction,
  canMutate,
  cancel,
  decideScenarioApproval,
}: {
  run: EvalRun
  cancelling: boolean
  pendingAction: string | null
  canMutate: boolean
  cancel: () => void
  decideScenarioApproval: (trial: EvalScenarioTrialProgress, decision: "approve" | "deny") => void
}) {
  const scenarioInvocation = run.spec.invocation?.scenario
  return (
    <DataCard
      title={
        <span className="flex items-center gap-2">
          Run {shortEvalIdentity(run.spec.run_id)} <EvalStatusBadge run={run} />
        </span>
      }
      description={run.spec.run_id}
      actions={
        evalRunCanCancel(run) ? (
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={!canMutate || cancelling}
            onClick={cancel}
          >
            {cancelling ? <LoaderCircle className="animate-spin" /> : <Ban />}
            {cancelling ? "Cancelling..." : "Cancel run"}
          </Button>
        ) : undefined
      }
    >
      <div className="grid gap-4 p-4 text-sm sm:grid-cols-2 lg:grid-cols-3">
        <RunFact label="Suite" value={run.spec.suite_id} />
        <RunFact label="Corpus" value={shortEvalIdentity(run.spec.corpus_revision)} />
        <RunFact label="Concurrency" value={formatCount(run.spec.max_concurrency)} />
        <RunFact label="Created" value={formatDateTime(run.created_at)} />
        <RunFact label="Started" value={formatDateTime(run.started_at)} />
        <RunFact label="Finished" value={formatDateTime(run.finished_at)} />
        {scenarioInvocation && (
          <>
            <RunFact
              label="Scenario"
              value={shortEvalIdentity(scenarioInvocation.scenario_revision)}
            />
            <RunFact label="Trials" value={formatCount(scenarioInvocation.trials)} />
          </>
        )}
      </div>
      {run.scenario_progress && (
        <div className="space-y-2 border-t border-border p-4" data-testid="scenario-run-progress">
          <div className="text-sm font-medium">Scenario trial progress</div>
          {run.scenario_progress.trials.map((trial) => {
            const approvalPending =
              pendingAction === `scenario-approval:${trial.trial_number}:approve` ||
              pendingAction === `scenario-approval:${trial.trial_number}:deny`
            return (
              <div
                key={trial.trial_number}
                className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-border bg-muted/20 p-3 text-sm"
              >
                <div>
                  <span className="font-medium">Trial {trial.trial_number}</span>
                  <span className="ml-2 text-muted-foreground">
                    {trial.phase.replaceAll("_", " ")} · event {trial.next_event_sequence}
                  </span>
                  {trial.pending_tool_name && (
                    <div className="mt-1 text-xs text-muted-foreground">
                      Fresh approval required for {trial.pending_tool_name}.
                    </div>
                  )}
                  {trial.failure_code && (
                    <div className="mt-1 text-xs text-destructive">
                      {trial.failure_code.replaceAll("_", " ")}
                    </div>
                  )}
                </div>
                {trial.phase === "awaiting_approval" && !trial.approval && (
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      disabled={!canMutate || pendingAction !== null}
                      onClick={() => decideScenarioApproval(trial, "deny")}
                    >
                      {approvalPending ? <LoaderCircle className="animate-spin" /> : <Ban />}
                      Deny
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      disabled={!canMutate || pendingAction !== null}
                      onClick={() => decideScenarioApproval(trial, "approve")}
                    >
                      {approvalPending ? (
                        <LoaderCircle className="animate-spin" />
                      ) : (
                        <CheckCircle2 />
                      )}
                      Approve
                    </Button>
                  </div>
                )}
                {trial.approval && (
                  <Badge variant="secondary">{trial.approval.decision} submitted</Badge>
                )}
              </div>
            )
          })}
        </div>
      )}
      {run.cancel_requested_at && (
        <div className="border-t border-border px-4 py-3 text-sm text-amber-700 dark:text-amber-300">
          Cancellation requested {formatDateTime(run.cancel_requested_at)}.
        </div>
      )}
      {run.failure_code && (
        <div className="border-t border-border px-4 py-3 text-sm text-destructive" role="alert">
          Run failed safely: {run.failure_code.replaceAll("_", " ")}.
        </div>
      )}
      {run.result && (
        <div className="grid gap-4 border-t border-border p-4 text-sm sm:grid-cols-3">
          <RunFact label="Result" value={run.result.status} />
          <RunFact label="Score" value={formatScore(run.result.score)} />
          <RunFact label="Duration" value={formatDuration(run.result.duration_ms)} />
        </div>
      )}
      <span
        className="sr-only"
        role="status"
        aria-live="polite"
        aria-atomic="true"
        data-testid="eval-run-status-announcement"
      >
        Eval run status: {run.status}.
      </span>
      {evalRunIsActive(run) && (
        <div className="border-t border-border px-4 py-3 text-xs text-muted-foreground">
          <LoaderCircle className="mr-1.5 inline h-3.5 w-3.5 animate-spin" aria-hidden="true" />
          Following durable status.
        </div>
      )}
    </DataCard>
  )
}

function ResultInspector({
  result,
  pendingAction,
  download,
}: {
  result: EvalResult
  pendingAction: string | null
  download: (format: "json" | "html") => void
}) {
  const published = result.result.run
  const [selection, setSelection] = useState({
    revision: published.revision,
    caseIndex: 0,
    trialIndex: 0,
  })
  const currentSelection =
    selection.revision === published.revision
      ? selection
      : { revision: published.revision, caseIndex: 0, trialIndex: 0 }
  const caseIndex = Math.min(currentSelection.caseIndex, published.cases.length - 1)
  const selectedCase = published.cases[caseIndex]
  const trialIndex = selectedCase
    ? Math.min(currentSelection.trialIndex, selectedCase.trials.length - 1)
    : 0
  const selectedTrial = selectedCase?.trials[trialIndex]
  const presentedCase = result.presentation.cases[caseIndex]
  const presentedTrial = presentedCase?.trials[trialIndex]
  const selectedTrialCost = selectedTrial ? evalTrialCostSummary(selectedTrial.assertions) : null

  return (
    <DataCard
      title={
        <span className="flex items-center gap-2">
          Published result <OutcomeBadge outcome={published.status} />
        </span>
      }
      description={`Release ${result.result.target.application_release_id} · result ${shortEvalIdentity(published.revision)}`}
      actions={
        <>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={pendingAction !== null}
            onClick={() => download("json")}
          >
            {pendingAction === "download-result-json" ? (
              <LoaderCircle className="animate-spin" />
            ) : (
              <FileJson />
            )}
            JSON
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={pendingAction !== null}
            onClick={() => download("html")}
          >
            {pendingAction === "download-result-html" ? (
              <LoaderCircle className="animate-spin" />
            ) : (
              <Download />
            )}
            HTML
          </Button>
        </>
      }
    >
      <div className="grid gap-4 p-4 text-sm sm:grid-cols-2 lg:grid-cols-4">
        <RunFact label="Score" value={formatScore(published.score)} />
        <RunFact label="Duration" value={formatDuration(published.duration_ms)} />
        <RunFact label="Cases" value={formatCount(published.cases.length)} />
        <RunFact
          label="App manifest"
          value={shortEvalIdentity(result.result.target.app_manifest.fingerprint)}
        />
      </div>
      <div className="border-t border-border p-4">
        <div className="mb-4 grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
          <RunFact
            label="Corpus"
            value={shortEvalIdentity(result.presentation.corpus_revision)}
            title={result.presentation.corpus_revision}
          />
          <RunFact
            label="Suite revision"
            value={shortEvalIdentity(result.presentation.suite_revision)}
            title={result.presentation.suite_revision}
          />
          <RunFact
            label="Evidence policy"
            value={shortEvalIdentity(result.presentation.evidence_policy_revision)}
            title={result.presentation.evidence_policy_revision}
          />
          <RunFact
            label="Pricing profile"
            value={
              result.presentation.pricing_profile_fingerprint
                ? shortEvalIdentity(result.presentation.pricing_profile_fingerprint)
                : "not used"
            }
            title={result.presentation.pricing_profile_fingerprint ?? undefined}
          />
        </div>
        <OutcomeDimensionsGrid dimensions={result.presentation.dimensions} />
      </div>

      {selectedCase && selectedTrial && presentedCase && presentedTrial && (
        <div className="space-y-4 border-t border-border p-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="min-w-0 text-xs font-medium text-muted-foreground">
              Case
              <select
                className="mt-1 h-9 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground"
                value={caseIndex}
                onChange={(event) =>
                  setSelection({
                    revision: published.revision,
                    caseIndex: Number(event.target.value),
                    trialIndex: 0,
                  })
                }
              >
                {published.cases.map((evalCase, index) => (
                  <option key={evalCase.case_id} value={index}>
                    {evalCase.case_id} — {evalCase.status}
                  </option>
                ))}
              </select>
            </label>
            <label className="min-w-0 text-xs font-medium text-muted-foreground">
              Trial
              <select
                className="mt-1 h-9 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground"
                value={trialIndex}
                onChange={(event) =>
                  setSelection({
                    revision: published.revision,
                    caseIndex,
                    trialIndex: Number(event.target.value),
                  })
                }
              >
                {selectedCase.trials.map((trial, index) => (
                  <option key={trial.trial_number} value={index}>
                    Trial {trial.trial_number} — {trial.status}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="grid gap-4 rounded-lg border border-border bg-muted/20 p-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
            <RunFact label="Case outcome" value={selectedCase.status} />
            <RunFact label="Case score" value={formatScore(selectedCase.score)} />
            <RunFact label="Trial outcome" value={selectedTrial.status} />
            <RunFact label="Trial score" value={formatScore(selectedTrial.score)} />
            <RunFact label="Trial duration" value={formatDuration(selectedTrial.duration_ms)} />
            <RunFact
              label="Evidence"
              value={selectedTrial.evidence_complete ? "complete" : "incomplete"}
            />
            <RunFact
              label="Memory evidence"
              value={`${selectedTrial.memory_attribution.completeness} · ${formatCount(selectedTrial.memory_attribution.retained_source_count)}/${formatCount(selectedTrial.memory_attribution.total_source_count)} sources`}
            />
            <RunFact label="Diagnostic" value={selectedTrial.code.replaceAll("_", " ")} />
            <RunFact
              label="Usage"
              value={
                selectedTrial.usage
                  ? `${formatCount(selectedTrial.usage.total_tokens)} tokens · ${formatCount(selectedTrial.usage.model_steps)} model steps · ${formatCount(selectedTrial.usage.tool_calls)} tools`
                  : "unavailable"
              }
            />
            <RunFact
              label="Estimated cost"
              value={selectedTrialCost?.display ?? "not evaluated"}
              title={selectedTrialCost?.exact}
            />
          </div>
          <OutcomeDimensionsGrid dimensions={presentedTrial.dimensions} />

          <div className="text-sm">
            <div className="text-xs font-medium text-muted-foreground">Trial diagnostic</div>
            <div className="mt-1">{selectedTrial.message}</div>
          </div>

          <div>
            <div className="mb-2 flex items-center justify-between gap-3">
              <div className="text-sm font-medium">Output preview</div>
              <Badge variant="outline">{selectedTrial.output.evidence_state}</Badge>
            </div>
            {selectedTrial.output.evidence_state !== "unavailable" &&
            selectedTrial.output.text !== undefined ? (
              <PayloadViewer value={selectedTrial.output.text} maxHeight="max-h-72" />
            ) : (
              <StateMessage className="rounded-md border border-border py-6">
                No output preview is available for this trial.
              </StateMessage>
            )}
            {selectedTrial.output.preview_truncated && (
              <div className="mt-2 text-xs text-amber-700 dark:text-amber-300">
                The preview is truncated. Its retained SHA-256 fingerprint is{" "}
                {selectedTrial.output.retained_sha256 ?? "unavailable"}.
              </div>
            )}
          </div>

          <div>
            <div className="mb-2 text-sm font-medium">
              Assertions ({formatCount(selectedTrial.assertions.length)})
            </div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Assertion</TableHead>
                  <TableHead>Outcome</TableHead>
                  <TableHead>Observation</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {selectedTrial.assertions.map((assertion, assertionIndex) => {
                  const presentedAssertion = presentedTrial.assertions[assertionIndex]
                  return (
                    <TableRow key={assertion.assertion_id}>
                      <TableCell>
                        <div
                          className="max-w-52 truncate font-medium"
                          title={assertion.assertion_id}
                        >
                          {assertion.assertion_id}
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          {assertion.detail.kind?.replaceAll("_", " ") ?? "assertion"}
                        </div>
                      </TableCell>
                      <TableCell>
                        <OutcomeBadge outcome={assertion.outcome} />
                        <div className="mt-1 text-xs text-muted-foreground">
                          {formatScore(assertion.score)}
                        </div>
                      </TableCell>
                      <TableCell className="min-w-64 whitespace-normal">
                        <div>{assertion.message}</div>
                        {presentedAssertion?.structured_judge ? (
                          <StructuredJudgeDetails
                            className="mt-3"
                            judgment={presentedAssertion.structured_judge}
                          />
                        ) : presentedAssertion?.tool_json ? (
                          <ToolJsonDetails className="mt-3" detail={presentedAssertion.tool_json} />
                        ) : presentedAssertion?.process ? (
                          <ProcessAssertionDetails
                            className="mt-3"
                            detail={presentedAssertion.process}
                          />
                        ) : (
                          <details className="mt-2 text-xs">
                            <summary className="cursor-pointer text-primary">
                              Inspect evidence
                            </summary>
                            <PayloadViewer
                              value={assertion.detail}
                              className="mt-2"
                              maxHeight="max-h-40"
                            />
                          </details>
                        )}
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </div>
        </div>
      )}
    </DataCard>
  )
}

function ResultPresentationInspector({ presentation }: { presentation: EvalResultPresentationV1 }) {
  const [selection, setSelection] = useState({
    revision: presentation.result_revision,
    caseIndex: 0,
    trialIndex: 0,
  })
  const current =
    selection.revision === presentation.result_revision
      ? selection
      : { revision: presentation.result_revision, caseIndex: 0, trialIndex: 0 }
  const caseIndex = Math.min(current.caseIndex, presentation.cases.length - 1)
  const selectedCase = presentation.cases[caseIndex]
  const trialIndex = selectedCase ? Math.min(current.trialIndex, selectedCase.trials.length - 1) : 0
  const selectedTrial = selectedCase?.trials[trialIndex]

  if (!selectedCase || !selectedTrial) {
    return <StateMessage>No retained case and trial presentation is available.</StateMessage>
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
        <RunFact label="Release" value={presentation.application_release_id} />
        <RunFact
          label={presentation.origin === "fresh_execution" ? "Published run" : "Captured score"}
          value={shortEvalIdentity(presentation.evaluation_revision)}
          title={presentation.evaluation_revision}
        />
        <RunFact
          label="App manifest"
          value={shortEvalIdentity(presentation.app_manifest_fingerprint)}
          title={presentation.app_manifest_fingerprint}
        />
        <RunFact
          label="Evidence policy"
          value={shortEvalIdentity(presentation.evidence_policy_revision)}
          title={presentation.evidence_policy_revision}
        />
        <RunFact
          label="Pricing profile"
          value={
            presentation.pricing_profile_fingerprint
              ? shortEvalIdentity(presentation.pricing_profile_fingerprint)
              : "not used"
          }
          title={presentation.pricing_profile_fingerprint ?? undefined}
        />
      </div>
      <OutcomeDimensionsGrid dimensions={presentation.dimensions} />
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="min-w-0 text-xs font-medium text-muted-foreground">
          Case
          <select
            className="mt-1 h-9 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground"
            value={caseIndex}
            onChange={(event) =>
              setSelection({
                revision: presentation.result_revision,
                caseIndex: Number(event.target.value),
                trialIndex: 0,
              })
            }
          >
            {presentation.cases.map((evalCase, index) => (
              <option key={evalCase.case_id} value={index}>
                {evalCase.case_id} — {evalCase.status}
              </option>
            ))}
          </select>
        </label>
        <label className="min-w-0 text-xs font-medium text-muted-foreground">
          Trial
          <select
            className="mt-1 h-9 w-full rounded-md border border-input bg-background px-2 text-sm text-foreground"
            value={trialIndex}
            onChange={(event) =>
              setSelection({
                revision: presentation.result_revision,
                caseIndex,
                trialIndex: Number(event.target.value),
              })
            }
          >
            {selectedCase.trials.map((trial, index) => (
              <option key={trial.trial_number ?? `captured-${index}`} value={index}>
                {trial.trial_number ? `Trial ${trial.trial_number}` : "Captured evidence"} —{" "}
                {trial.status}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="grid gap-3 rounded-lg border border-border bg-muted/20 p-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
        <RunFact label="Case outcome" value={selectedCase.status} />
        <RunFact label="Case score" value={formatScore(selectedCase.score)} />
        <RunFact label="Trial outcome" value={selectedTrial.status} />
        <RunFact label="Trial score" value={formatScore(selectedTrial.score)} />
      </div>
      <OutcomeDimensionsGrid dimensions={selectedTrial.dimensions} />
      <PresentedAssertions assertions={selectedTrial.assertions} />
    </div>
  )
}

function OutcomeDimensionsGrid({ dimensions }: { dimensions: EvalResultOutcomeDimensionsV1 }) {
  const items = [
    ["Candidate", dimensions.candidate],
    ["Deterministic assertions", dimensions.deterministic_assertions],
    ["Semantic quality", dimensions.semantic_quality],
    ["Evaluator health", dimensions.evaluator_health],
    ["Runtime", dimensions.runtime],
    ["Evidence", dimensions.evidence],
  ] as const
  return (
    <div className="grid gap-3 rounded-lg border border-border bg-muted/20 p-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
      {items.map(([label, value]) => (
        <RunFact key={label} label={label} value={value.replaceAll("_", " ")} />
      ))}
    </div>
  )
}

function PresentedAssertions({ assertions }: { assertions: Array<EvalAssertionPresentationV1> }) {
  return (
    <div className="space-y-3">
      <div className="text-sm font-medium">Assertions ({formatCount(assertions.length)})</div>
      {assertions.map((assertion) => (
        <div key={assertion.assertion_id} className="rounded-lg border border-border p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="font-medium">{assertion.assertion_id}</div>
              <div className="text-xs text-muted-foreground">
                {assertion.kind.replaceAll("_", " ")} · {assertion.category}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <OutcomeBadge outcome={assertion.outcome} />
              <span className="text-xs text-muted-foreground">{formatScore(assertion.score)}</span>
            </div>
          </div>
          {assertion.structured_judge && (
            <StructuredJudgeDetails className="mt-3" judgment={assertion.structured_judge} />
          )}
          {assertion.tool_json && <ToolJsonDetails className="mt-3" detail={assertion.tool_json} />}
          {assertion.process && (
            <ProcessAssertionDetails className="mt-3" detail={assertion.process} />
          )}
        </div>
      ))}
    </div>
  )
}

function ToolJsonDetails({
  detail,
  className,
}: {
  detail: NonNullable<EvalAssertionPresentationV1["tool_json"]>
  className?: string
}) {
  return (
    <div
      className={`space-y-3 rounded-lg border border-border bg-muted/20 p-3 ${className ?? ""}`}
      data-testid="eval-tool-json-detail"
    >
      <div className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
        <RunFact label="Tool" value={detail.tool_name} />
        <RunFact label="Occurrence" value={String(detail.occurrence)} />
        <RunFact label="Evidence" value={detail.observation_state.replaceAll("_", " ")} />
        <RunFact
          label="Match"
          value={detail.matched == null ? "unavailable" : detail.matched ? "matched" : "mismatch"}
        />
        <RunFact
          label="Invocation"
          value={
            detail.invocation_index == null || detail.invocation_revision == null
              ? "unavailable"
              : `#${detail.invocation_index} · ${shortEvalIdentity(detail.invocation_revision)}`
          }
          title={detail.invocation_revision ?? undefined}
        />
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">Expected subset</div>
          <PayloadViewer value={detail.expected_subset} maxHeight="max-h-40" />
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">Observed safe value</div>
          {detail.actual == null ? (
            <StateMessage className="rounded-md border border-border py-4">
              No safe observed value is available.
            </StateMessage>
          ) : (
            <PayloadViewer value={detail.actual} maxHeight="max-h-40" />
          )}
        </div>
      </div>
    </div>
  )
}

const PROCESS_EVENT_LABELS = new Map(PROCESS_EVENT_OPTIONS)

function ProcessAssertionDetails({
  detail,
  className,
}: {
  detail: NonNullable<EvalAssertionPresentationV1["process"]>
  className?: string
}) {
  let facts: Array<[string, string]>
  if (detail.kind === "child_status") {
    facts = [
      ["Child status", detail.expected],
      ["Observed", detail.matching_count == null ? "unavailable" : String(detail.matching_count)],
      ["Required", formatCountRange(detail.min_count, detail.max_count)],
    ]
  } else if (detail.kind === "process_event") {
    facts = [
      ["Process event", PROCESS_EVENT_LABELS.get(detail.event) ?? detail.event],
      ["Observed", detail.matching_count == null ? "unavailable" : String(detail.matching_count)],
      ["Required", formatCountRange(detail.min_count, detail.max_count)],
    ]
  } else {
    const order = detail as PublishedProcessEventsInOrderDetail
    facts = [
      ["Expected selected events", String(order.expected.length)],
      [
        "Observed selected events",
        order.actual_count == null ? "unavailable" : String(order.actual_count),
      ],
      [
        "Exact order",
        order.matched == null ? "unavailable" : order.matched ? "matched" : "mismatch",
      ],
    ]
  }
  return (
    <div
      className={`rounded-lg border border-border bg-muted/20 p-3 ${className ?? ""}`}
      data-testid="eval-process-detail"
    >
      <div className="grid gap-3 text-xs sm:grid-cols-3">
        {facts.map(([label, value]) => (
          <RunFact key={label} label={label} value={value} />
        ))}
      </div>
      {detail.kind === "process_events_in_order" && (
        <div className="mt-3 space-y-2 text-xs text-muted-foreground">
          <p className="break-words font-mono" data-testid="eval-process-expected-order">
            {detail.expected.map((event) => PROCESS_EVENT_LABELS.get(event) ?? event).join(" → ")}
          </p>
          <p>
            Only the event kinds selected by the assertion participate; their complete filtered
            sequence and multiplicity must match exactly.
          </p>
        </div>
      )}
    </div>
  )
}

function formatCountRange(minimum: number, maximum: number | null | undefined): string {
  if (maximum === minimum) return String(minimum)
  if (maximum == null) return `at least ${minimum}`
  return `${minimum} to ${maximum}`
}

function StructuredJudgeDetails({
  judgment,
  className,
}: {
  judgment: EvalStructuredJudgePresentationV1
  className?: string
}) {
  const detail = judgment.detail
  const profile = detail.judge_profile
  const reference = detail.reference
  const criteria = judgment.criteria ?? []
  const thresholdState =
    judgment.threshold_passed === null
      ? "unavailable"
      : judgment.threshold_passed
        ? "passed"
        : "failed"
  const usage = structuredJudgeUsageText(judgment)
  const cost = structuredJudgeCostText(judgment)
  return (
    <div className={`space-y-3 rounded-lg border border-border bg-muted/20 p-3 ${className ?? ""}`}>
      <div className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-3">
        <RunFact
          label="Judge profile"
          value={`${profile.label} · ${shortEvalIdentity(profile.revision)}`}
          title={`${profile.key}@${profile.revision}`}
        />
        <RunFact label="Provider / model" value={`${profile.provider_name} / ${profile.model}`} />
        <RunFact
          label="Candidate route"
          value={
            detail.candidate_route_relation === "same_model"
              ? "same model · explicitly allowed"
              : "independent model"
          }
        />
        <RunFact
          label="Rubric"
          value={`${detail.rubric_id} · ${shortEvalIdentity(detail.rubric_revision)}`}
          title={detail.rubric_revision}
        />
        <RunFact label="Evaluator" value={detail.diagnostic.replaceAll("_", " ")} />
        <RunFact
          label="Aggregate / threshold"
          value={`${detail.aggregate_score ?? "unavailable"} / ${detail.threshold} · ${thresholdState}`}
        />
        <RunFact label="Observed usage" value={usage} />
        <RunFact label="Observed cost" value={cost} />
        <RunFact
          label="Evidence"
          value={
            [
              detail.evidence.include_final_output ? "final output" : null,
              detail.evidence.include_transcript ? "transcript" : null,
            ]
              .filter(Boolean)
              .join(" + ") || "none"
          }
        />
        <RunFact
          label="Reference"
          value={reference ? `${reference.kind.replaceAll("_", " ")} · available` : "none"}
          title={reference ? `${reference.key}@${reference.revision}` : undefined}
        />
        <RunFact
          label="Evaluator implementation"
          value={shortEvalIdentity(profile.implementation_revision)}
          title={profile.implementation_revision}
        />
        <RunFact
          label="Evidence privacy policy"
          value={`${profile.privacy_policy_key} · ${shortEvalIdentity(profile.privacy_policy_revision)}`}
          title={profile.privacy_policy_revision}
        />
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Criterion</TableHead>
            <TableHead>Weight</TableHead>
            <TableHead>Score</TableHead>
            <TableHead>Contribution</TableHead>
            <TableHead>Explanation</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {criteria.length === 0 ? (
            <TableRow>
              <TableCell colSpan={5}>No criterion scores were recorded.</TableCell>
            </TableRow>
          ) : (
            criteria.map((criterion) => (
              <TableRow key={criterion.criterion_id}>
                <TableCell className="font-medium">{criterion.criterion_id}</TableCell>
                <TableCell>{criterion.weight}</TableCell>
                <TableCell>{criterion.score}</TableCell>
                <TableCell>{criterion.weighted_contribution}</TableCell>
                <TableCell className="min-w-64 whitespace-normal">
                  <div>{criterion.explanation ?? "Unavailable"}</div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    {criterion.explanation_state}
                  </div>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  )
}

function structuredJudgeUsageText(judgment: EvalStructuredJudgePresentationV1): string {
  const usage = judgment.detail.usage
  return usage
    ? `${formatCount(usage.total_tokens)} tokens · ${formatCount(usage.model_steps)} model steps`
    : "unavailable"
}

function structuredJudgeCostText(judgment: EvalStructuredJudgePresentationV1): string {
  const cost = judgment.detail.cost
  if (!cost) return "unavailable · not observed"
  return cost.availability === "priced"
    ? `${cost.estimated_cost} ${cost.currency}`
    : "unavailable · unpriced"
}

function ComparisonPanel({
  currentResultRevision,
  baselineResultRevision,
  approvedBaselineRevision,
  candidates,
  comparison,
  loading,
  error,
  retry,
  selectBaseline,
}: {
  currentResultRevision: string
  baselineResultRevision?: string
  approvedBaselineRevision?: string
  candidates: Array<EvalResultSummary>
  comparison?: EvalResultComparison
  loading: boolean
  error: unknown
  retry: () => void
  selectBaseline: (resultRevision?: string) => void
}) {
  const [draft, setDraft] = useState(baselineResultRevision ?? "")
  const normalizedDraft = draft.trim()
  const baselineIsSelf = baselineResultRevision === currentResultRevision
  const draftIsValid =
    evalResultRevisionIsValid(normalizedDraft) && normalizedDraft !== currentResultRevision
  const usingApprovedBaseline =
    baselineResultRevision !== undefined && baselineResultRevision === approvedBaselineRevision

  return (
    <DataCard
      title="Compare results"
      description="The approved suite baseline is used by default; the server decides whether the immutable results share a compatible eval contract."
    >
      <form
        className="flex min-w-0 flex-wrap items-end gap-2 border-b border-border p-4"
        onSubmit={(event) => {
          event.preventDefault()
          if (draftIsValid) selectBaseline(normalizedDraft)
        }}
      >
        <label className="min-w-64 flex-1 text-xs font-medium text-muted-foreground">
          Baseline result revision
          <input
            type="text"
            list="eval-baseline-candidates"
            value={draft}
            maxLength={71}
            className="mt-1 h-9 w-full rounded-md border border-input bg-background px-3 text-sm text-foreground"
            placeholder="Select or paste a sha256 result revision"
            aria-invalid={normalizedDraft.length > 0 && !draftIsValid}
            onChange={(event) => setDraft(event.target.value)}
          />
          <datalist id="eval-baseline-candidates">
            {candidates.map((result) => (
              <option key={result.revision} value={result.revision}>
                {result.suite_id} ·{" "}
                {result.origin === "captured_session" ? "captured" : "current app"} ·{" "}
                {result.status}
              </option>
            ))}
          </datalist>
        </label>
        <Button type="submit" size="sm" disabled={!draftIsValid || loading}>
          {loading ? <LoaderCircle className="animate-spin" /> : <FlaskConical />}
          Compare
        </Button>
        {baselineResultRevision && !usingApprovedBaseline && (
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => selectBaseline(undefined)}
          >
            {approvedBaselineRevision ? "Use approved baseline" : "Clear"}
          </Button>
        )}
      </form>

      {!baselineResultRevision ? (
        <StateMessage>
          Approve a suite baseline or choose another immutable result. Different application
          releases are allowed when the corpus, suite, cases, assertions, evidence policy, and
          applicable pricing contract match.
        </StateMessage>
      ) : baselineIsSelf && usingApprovedBaseline ? (
        <StateMessage>
          This result is the approved suite baseline. Select a different current result to compare
          against it.
        </StateMessage>
      ) : baselineIsSelf ? (
        <StateMessage tone="danger" role="alert">
          Choose a different immutable result as the comparison baseline.
        </StateMessage>
      ) : loading ? (
        <LoadingState label="Checking comparison compatibility..." />
      ) : error ? (
        <QueryError message="Could not compare the selected eval results." retry={retry} />
      ) : comparison ? (
        <ComparisonSummary comparison={comparison} />
      ) : null}
    </DataCard>
  )
}

function ComparisonSummary({ comparison }: { comparison: EvalResultComparison }) {
  const { comparison: resultComparison, baseline, current } = comparison
  const { compatibility } = resultComparison
  const regressions = resultComparison.regressions ?? []
  const structuredJudgments = resultComparison.structured_judgments ?? []
  const structuredRegressions = structuredJudgments.filter((item) => item.regressed)
  const toolJsonAssertions = resultComparison.tool_json_assertions ?? []
  const toolJsonRegressions = toolJsonAssertions.filter((item) => item.regressed)
  const regressionCount =
    regressions.length + structuredRegressions.length + toolJsonRegressions.length
  const reasons = compatibility.reasons ?? []
  return (
    <div className="space-y-4 p-4">
      <div
        className={
          compatibility.comparable
            ? "rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 text-sm text-emerald-700 dark:text-emerald-300"
            : "rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-amber-700 dark:text-amber-300"
        }
        role="status"
      >
        <div className="font-medium">
          {compatibility.comparable
            ? "These results are comparable."
            : "These results are not comparable as one regression contract."}
        </div>
        {!compatibility.comparable && reasons.length > 0 && (
          <ul className="mt-2 list-disc space-y-1 pl-5">
            {reasons.map((reason) => (
              <li key={reason}>{evalComparisonReasonText(reason)}</li>
            ))}
          </ul>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <ComparisonResultSummary
          label="Baseline"
          record={baseline}
          result={resultComparison.baseline}
        />
        <ComparisonResultSummary
          label="Current"
          record={current}
          result={resultComparison.current}
        />
      </div>
      {compatibility.comparable && (
        <div
          className={
            regressionCount === 0
              ? "rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 text-sm"
              : "rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm"
          }
          data-testid="eval-comparison-regressions"
        >
          <div className="font-medium">
            {regressionCount === 0
              ? "No compatible-result regressions."
              : `${regressionCount} compatible-result regression${regressionCount === 1 ? "" : "s"}.`}
          </div>
          {regressions.length > 0 && (
            <ul className="mt-2 list-disc space-y-1 pl-5">
              {regressions.map((regression) => (
                <li key={`${regression.scope}:${regression.case_id ?? "run"}:${regression.kind}`}>
                  {regression.scope === "case" ? `Case ${regression.case_id}: ` : "Run: "}
                  {regression.kind === "status"
                    ? `status ${regression.baseline_status} → ${regression.current_status}`
                    : `score ${formatScore(regression.baseline_score)} → ${formatScore(regression.current_score)}`}
                </li>
              ))}
            </ul>
          )}
          {structuredRegressions.length > 0 && (
            <ul className="mt-2 list-disc space-y-1 pl-5">
              {structuredRegressions.map((item) => (
                <li key={`${item.case_id}:${item.trial_number ?? "captured"}:${item.assertion_id}`}>
                  Case {item.case_id},{" "}
                  {item.trial_number ? `trial ${item.trial_number}` : "captured evidence"},
                  assertion {item.assertion_id}: evaluator {item.evaluator_change}, aggregate{" "}
                  {item.aggregate_change}
                  {item.aggregate_delta === null ? "" : ` (${item.aggregate_delta})`}.
                </li>
              ))}
            </ul>
          )}
          {toolJsonRegressions.length > 0 && (
            <ul className="mt-2 list-disc space-y-1 pl-5">
              {toolJsonRegressions.map((item) => (
                <li
                  key={[item.case_id, item.trial_number ?? "captured", item.assertion_id].join(":")}
                >
                  Case {item.case_id},{" "}
                  {item.trial_number ? `trial ${item.trial_number}` : "captured evidence"},
                  assertion {item.assertion_id}: outcome {item.baseline_outcome} →{" "}
                  {item.current_outcome}; evidence {item.baseline.observation_state} →{" "}
                  {item.current.observation_state}; observed value {item.observed_value_change}.
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      <StructuredJudgeComparison comparison={resultComparison} />
      <ToolJsonComparison comparison={resultComparison} />
      <div className="grid gap-4 rounded-lg border border-border bg-muted/20 p-3 text-xs sm:grid-cols-2">
        <RunFact
          label="Baseline result revision"
          value={shortEvalIdentity(compatibility.baseline_result_revision)}
        />
        <RunFact
          label="Current result revision"
          value={shortEvalIdentity(compatibility.current_result_revision)}
        />
      </div>
    </div>
  )
}

function ToolJsonComparison({ comparison }: { comparison: EvalResultComparison["comparison"] }) {
  const state = comparison.tool_json_comparison_state
  const stateText = {
    compared: "Exact bounded tool-JSON observations were compared.",
    contract_incompatible:
      "Tool-JSON observations were not diffed because the immutable evaluation contracts differ.",
    no_tool_json_assertions: "Neither result contains tool-JSON assertions.",
    observation_identity_mismatch:
      "Tool-JSON observations were not paired because retained observation identities differ.",
    source_detail_unavailable:
      "One compact result projection does not retain tool-JSON observation detail.",
  }[state]
  const mismatches = comparison.tool_json_observation_mismatches ?? []
  const assertions = comparison.tool_json_assertions ?? []
  return (
    <div className="space-y-3" data-testid="eval-tool-json-comparison">
      <div>
        <div className="font-medium">Tool JSON assertion comparison</div>
        <div className="mt-1 text-xs text-muted-foreground">
          {stateText} State: {state.replaceAll("_", " ")}.
        </div>
      </div>
      {mismatches.length > 0 && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm">
          <div className="font-medium">Unmatched retained observations</div>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            {mismatches.map((item) => (
              <li
                key={[
                  item.availability,
                  item.case_id,
                  item.trial_number ?? "captured",
                  item.assertion_id,
                ].join(":")}
              >
                {item.case_id} ·{" "}
                {item.trial_number ? `trial ${item.trial_number}` : "captured evidence"} ·{" "}
                {item.assertion_id}: {item.availability.replaceAll("_", " ")}
              </li>
            ))}
          </ul>
        </div>
      )}
      {assertions.map((item) => (
        <ToolJsonComparisonDetails
          key={[item.case_id, item.trial_number ?? "captured", item.assertion_id].join(":")}
          item={item}
        />
      ))}
    </div>
  )
}

function ToolJsonComparisonDetails({ item }: { item: EvalToolJsonAssertionComparisonV1 }) {
  return (
    <div
      className={
        item.regressed
          ? "rounded-lg border border-destructive/30 bg-destructive/5 p-3"
          : "rounded-lg border border-border p-3"
      }
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="font-medium">
            {item.case_id} ·{" "}
            {item.trial_number ? `trial ${item.trial_number}` : "captured evidence"} ·{" "}
            {item.assertion_id}
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            {(item.baseline.kind ?? "tool JSON").replaceAll("_", " ")} · {item.baseline.tool_name} ·
            occurrence {item.baseline.occurrence}
          </div>
        </div>
        <Badge variant={item.regressed ? "destructive" : "outline"}>
          {item.regressed ? "regressed" : item.observed_value_change}
        </Badge>
      </div>
      <div className="mt-3 grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
        <RunFact label="Outcome" value={`${item.baseline_outcome} → ${item.current_outcome}`} />
        <RunFact
          label="Evidence"
          value={`${item.baseline.observation_state} → ${item.current.observation_state}`}
        />
        <RunFact label="Observed value" value={item.observed_value_change} />
        <RunFact
          label="Evidence state changed"
          value={item.evidence_state_changed ? "yes" : "no"}
        />
      </div>
      <div className="mt-3 grid gap-3 lg:grid-cols-3">
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">Expected subset</div>
          <PayloadViewer value={item.baseline.expected_subset} maxHeight="max-h-40" />
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">Baseline safe value</div>
          <PayloadViewer value={item.baseline.actual ?? "unavailable"} maxHeight="max-h-40" />
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-muted-foreground">Current safe value</div>
          <PayloadViewer value={item.current.actual ?? "unavailable"} maxHeight="max-h-40" />
        </div>
      </div>
    </div>
  )
}

function StructuredJudgeComparison({
  comparison,
}: {
  comparison: EvalResultComparison["comparison"]
}) {
  const state = comparison.structured_judge_comparison_state
  const stateText = {
    compared: "Exact retained case, trial, and assertion identities were paired.",
    contract_incompatible:
      "Judged outcomes were not diffed because the immutable evaluation contracts differ.",
    no_structured_judges: "Neither result contains structured AI-judge observations.",
    observation_identity_mismatch:
      "Judged outcomes were not paired because retained observation identities differ.",
    source_detail_unavailable:
      "One compact result projection does not retain structured judgment detail.",
  }[state]
  const mismatches = comparison.structured_judge_observation_mismatches ?? []
  const judgments = comparison.structured_judgments ?? []
  return (
    <div className="space-y-3">
      <div>
        <div className="font-medium">Structured judge comparison</div>
        <div className="mt-1 text-xs text-muted-foreground">
          {stateText} State: {state.replaceAll("_", " ")}.
        </div>
      </div>
      {mismatches.length > 0 && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm">
          <div className="font-medium">Unmatched retained observations</div>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            {mismatches.map((item) => (
              <li
                key={`${item.availability}:${item.case_id}:${item.trial_number ?? "captured"}:${item.assertion_id}`}
              >
                {item.case_id} ·{" "}
                {item.trial_number ? `trial ${item.trial_number}` : "captured evidence"} ·{" "}
                {item.assertion_id}: {item.availability.replaceAll("_", " ")}
              </li>
            ))}
          </ul>
        </div>
      )}
      {judgments.map((item) => (
        <StructuredJudgeComparisonDetails
          key={`${item.case_id}:${item.trial_number ?? "captured"}:${item.assertion_id}`}
          item={item}
        />
      ))}
    </div>
  )
}

function StructuredJudgeComparisonDetails({ item }: { item: EvalStructuredJudgeComparisonV1 }) {
  const criteria = item.criteria ?? []
  const baselineCriteria = new Map(
    (item.baseline.criteria ?? []).map((criterion) => [criterion.criterion_id, criterion]),
  )
  const currentCriteria = new Map(
    (item.current.criteria ?? []).map((criterion) => [criterion.criterion_id, criterion]),
  )
  const profile = item.baseline.detail.judge_profile
  return (
    <div
      className={
        item.regressed
          ? "rounded-lg border border-destructive/30 bg-destructive/5 p-3"
          : "rounded-lg border border-border p-3"
      }
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="font-medium">
            {item.case_id} ·{" "}
            {item.trial_number ? `trial ${item.trial_number}` : "captured evidence"} ·{" "}
            {item.assertion_id}
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            {profile.label} · {profile.provider_name}/{profile.model} ·{" "}
            {item.baseline.detail.candidate_route_relation.replaceAll("_", " ")}
          </div>
        </div>
        <Badge variant={item.regressed ? "destructive" : "outline"}>
          {item.regressed ? "regressed" : item.aggregate_change}
        </Badge>
      </div>
      <div className="mt-3 grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
        <RunFact label="Outcome" value={`${item.baseline_outcome} → ${item.current_outcome}`} />
        <RunFact label="Evaluator" value={item.evaluator_change} />
        <RunFact
          label="Aggregate"
          value={`${item.baseline.detail.aggregate_score ?? "unavailable"} → ${item.current.detail.aggregate_score ?? "unavailable"}`}
        />
        <RunFact
          label="Aggregate delta"
          value={`${item.aggregate_delta ?? "unavailable"} · ${item.aggregate_change}`}
        />
        <RunFact
          label="Rubric"
          value={`${item.baseline.detail.rubric_id} · ${shortEvalIdentity(item.baseline.detail.rubric_revision)}`}
          title={item.baseline.detail.rubric_revision}
        />
        <RunFact
          label="Judge profile"
          value={`${profile.key} · ${shortEvalIdentity(profile.revision)}`}
          title={profile.revision}
        />
        <RunFact
          label="Evaluator diagnostic"
          value={`${item.baseline.detail.diagnostic.replaceAll("_", " ")} → ${item.current.detail.diagnostic.replaceAll("_", " ")}`}
        />
        <RunFact
          label="Reference"
          value={
            item.baseline.detail.reference
              ? `${item.baseline.detail.reference.kind.replaceAll("_", " ")} · available`
              : "none"
          }
        />
        <RunFact
          label="Observed usage"
          value={`${structuredJudgeUsageText(item.baseline)} → ${structuredJudgeUsageText(item.current)}`}
        />
        <RunFact
          label="Observed cost"
          value={`${structuredJudgeCostText(item.baseline)} → ${structuredJudgeCostText(item.current)}`}
        />
      </div>
      <Table className="mt-3">
        <TableHeader>
          <TableRow>
            <TableHead>Criterion</TableHead>
            <TableHead>Weight</TableHead>
            <TableHead>Baseline</TableHead>
            <TableHead>Current</TableHead>
            <TableHead>Delta</TableHead>
            <TableHead>Explanations</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {criteria.length === 0 ? (
            <TableRow>
              <TableCell colSpan={6}>
                Criterion deltas are unavailable because one or both judgments were not recorded.
              </TableCell>
            </TableRow>
          ) : (
            criteria.map((criterion) => {
              const baseline = baselineCriteria.get(criterion.criterion_id)
              const current = currentCriteria.get(criterion.criterion_id)
              return (
                <TableRow key={criterion.criterion_id}>
                  <TableCell className="font-medium">{criterion.criterion_id}</TableCell>
                  <TableCell>{criterion.weight}</TableCell>
                  <TableCell>{criterion.baseline_score}</TableCell>
                  <TableCell>{criterion.current_score}</TableCell>
                  <TableCell>{criterion.score_delta}</TableCell>
                  <TableCell className="min-w-72 whitespace-normal">
                    <div>
                      <span className="font-medium">Baseline:</span>{" "}
                      {baseline?.explanation ?? "Unavailable"}
                    </div>
                    <div className="mt-1">
                      <span className="font-medium">Current:</span>{" "}
                      {current?.explanation ?? "Unavailable"}
                    </div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      {criterion.baseline_explanation_state} → {criterion.current_explanation_state}
                    </div>
                  </TableCell>
                </TableRow>
              )
            })
          )}
        </TableBody>
      </Table>
    </div>
  )
}

function ComparisonResultSummary({
  label,
  record,
  result,
}: {
  label: string
  record: EvalResultSummary
  result: EvalResultComparison["comparison"]["baseline"]
}) {
  return (
    <div className="rounded-lg border border-border p-3">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="font-medium">{label}</div>
        <OutcomeBadge outcome={result.status} />
      </div>
      <div className="grid gap-3 text-sm sm:grid-cols-2">
        <RunFact label="Result" value={shortEvalIdentity(record.revision)} />
        <RunFact
          label="Origin"
          value={
            record.origin === "captured_session" ? "Captured session" : "Current-app execution"
          }
        />
        <RunFact label="Suite" value={record.suite_id} />
        <RunFact label="Target" value={result.target_key} />
        <RunFact label="Release" value={result.application_release_id} />
        <RunFact label="App manifest" value={shortEvalIdentity(result.app_manifest_fingerprint)} />
        <RunFact label="Score" value={formatScore(result.score)} />
        <RunFact
          label="Corpus"
          value={shortEvalIdentity(result.corpus_revision)}
          title={result.corpus_revision}
        />
        <RunFact
          label="Suite revision"
          value={shortEvalIdentity(result.suite_revision)}
          title={result.suite_revision}
        />
        <RunFact
          label="Evidence policy"
          value={shortEvalIdentity(result.evidence_policy_revision)}
          title={result.evidence_policy_revision}
        />
        <RunFact
          label="Pricing profile"
          value={
            result.pricing_profile_fingerprint
              ? shortEvalIdentity(result.pricing_profile_fingerprint)
              : "not used"
          }
          title={result.pricing_profile_fingerprint ?? undefined}
        />
        <RunFact label="Created" value={formatDateTime(record.created_at)} />
      </div>
    </div>
  )
}

function OutcomeBadge({ outcome }: { outcome: string }) {
  const variant =
    outcome === "failed" || outcome === "error"
      ? "destructive"
      : outcome === "passed" || outcome === "completed"
        ? "secondary"
        : "outline"
  return <Badge variant={variant}>{outcome}</Badge>
}

function EvalStatusBadge({ run }: { run: EvalRun }) {
  const status = run.status
  const variant =
    status === "failed" ? "destructive" : status === "completed" ? "secondary" : "outline"
  return <Badge variant={variant}>{status}</Badge>
}

function RunFact({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="min-w-0">
      <div className="text-xs font-medium text-muted-foreground">{label}</div>
      <div className="mt-1 truncate" title={title ?? value}>
        {value}
      </div>
    </div>
  )
}

function formatScore(score: number | null | undefined): string {
  return score == null ? "No score" : `${(score * 100).toFixed(1)}%`
}

function formatDuration(milliseconds: number): string {
  if (milliseconds < 1_000) return `${formatCount(milliseconds)} ms`
  return `${(milliseconds / 1_000).toFixed(milliseconds < 10_000 ? 1 : 0)} s`
}

function LoadingState({ label }: { label: string }) {
  return (
    <StateMessage role="status" aria-live="polite">
      <LoaderCircle className="mr-2 inline h-4 w-4 animate-spin" />
      {label}
    </StateMessage>
  )
}

function QueryError({ message, retry }: { message: string; retry: () => void }) {
  return (
    <StateMessage tone="danger" role="alert">
      <div>{message}</div>
      <Button type="button" variant="outline" size="sm" className="mt-3" onClick={retry}>
        <RotateCcw /> Retry
      </Button>
    </StateMessage>
  )
}

function PageControls({
  scope,
  cursor,
  nextCursor,
  fetching,
  first,
  next,
}: {
  scope: string
  cursor?: string
  nextCursor?: string | null
  fetching: boolean
  first: () => void
  next: (cursor: string) => void
}) {
  if (!cursor && !nextCursor) return null
  return (
    <nav
      aria-label={`${scope} pagination`}
      className="flex items-center justify-end gap-2 border-t border-border p-3"
    >
      <Button
        type="button"
        size="sm"
        variant="outline"
        aria-label={`First ${scope} page`}
        disabled={!cursor || fetching}
        onClick={first}
      >
        First page
      </Button>
      <Button
        type="button"
        size="sm"
        variant="outline"
        aria-label={`Next ${scope} page`}
        disabled={!nextCursor || fetching}
        onClick={() => nextCursor && next(nextCursor)}
      >
        Next page
      </Button>
    </nav>
  )
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = filename
  anchor.hidden = true
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
}

function randomEvalOperationId(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32))
  const digest = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")
  return `sha256:${digest}`
}
