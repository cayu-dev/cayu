import { useQuery, useQueryClient } from "@tanstack/react-query"
import { useNavigate, useSearch } from "@tanstack/react-router"
import {
  Ban,
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
import {
  DataCard,
  Page,
  PageHeader,
  PayloadViewer,
  StateMessage,
} from "@/components/dashboard/layout"
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
  compareEvalRuns,
  createEvalRun,
  downloadEvalCorpus,
  downloadEvalResultHtml,
  downloadEvalResultJson,
  type EvalComparison,
  type EvalCorpusEntry,
  type EvalResult,
  type EvalRun,
  type EvalStatus,
  fetchEvalCases,
  fetchEvalCorpora,
  fetchEvalResult,
  fetchEvalRun,
  fetchEvalRuns,
  fetchEvalSuites,
  importEvalCorpus,
} from "@/lib/api"
import { dashboardConfig } from "@/lib/config"
import { dashboardCapabilityUnavailableText } from "@/lib/dashboard-capabilities"
import {
  EVAL_RESULT_QUERY_RETENTION,
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
  evalTrialCostSummary,
  preflightEvalCorpusFile,
  shortEvalIdentity,
} from "@/lib/evals-dashboard"
import {
  EVALS_READINESS_OPERATIONS,
  evalsReadinessReasonText,
  evalsReadinessStateLabel,
} from "@/lib/evals-readiness"
import { type EvalsSearch, evalRunIdIsValid, evalsSearchWithout } from "@/lib/evals-search"
import { formatBytes, formatCount, formatDateTime } from "@/lib/format"
import type { EvalsReadiness } from "@/lib/generated/server-api"

const PAGE_LIMIT = 25
type UpdateEvalsSearch = (next: (current: EvalsSearch) => EvalsSearch) => Promise<void>

export function EvalsPage() {
  const search = useSearch({ from: "/evals" })
  const navigate = useNavigate({ from: "/evals" })
  const queryClient = useQueryClient()
  const readiness = useServerContract().capabilities.evals_readiness
  const catalogReady = readiness.catalog_read.state === "ready"
  const mutateCapability = useDashboardCapability({
    kind: "surface",
    surface: "evals",
    operation: "mutate",
  })
  const mutationUnavailable = dashboardCapabilityUnavailableText(mutateCapability)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const catalogTabRef = useRef<HTMLButtonElement>(null)
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
      }))
      return `Imported corpus ${shortEvalIdentity(imported.revision)}.`
    })
  }

  const activeTab = search.tab ?? "catalog"
  const showCatalog = () =>
    updateSearch((current) => ({
      ...evalsSearchWithout(current, "run", "baseline", "runs_cursor", "status"),
      tab: "catalog",
    }))
  const showRuns = () =>
    updateSearch((current) => ({
      ...evalsSearchWithout(current, "suite", "suites_cursor", "cases_cursor", "corpora_cursor"),
      tab: "runs",
    }))
  const moveTabFocus = (event: KeyboardEvent<HTMLDivElement>) => {
    const focusedTab = document.activeElement === runsTabRef.current ? "runs" : "catalog"
    const nextTab =
      event.key === "Home"
        ? "catalog"
        : event.key === "End"
          ? "runs"
          : event.key === "ArrowLeft" || event.key === "ArrowRight"
            ? focusedTab === "catalog"
              ? "runs"
              : "catalog"
            : null
    if (nextTab === null) return
    event.preventDefault()
    if (nextTab === "catalog") {
      catalogTabRef.current?.focus()
      void showCatalog()
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
        description="Manage portable regression corpora and durable fresh evaluation runs."
        actions={
          <>
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

      <div
        id="evals-panel-catalog"
        role="tabpanel"
        aria-labelledby="evals-tab-catalog"
        hidden={activeTab !== "catalog"}
      >
        {activeTab === "catalog" && (
          <CatalogView
            search={search}
            updateSearch={updateSearch}
            pendingAction={pendingAction}
            runAction={runAction}
            mutateEnabled={mutateCapability.enabled}
          />
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
            updateSearch={updateSearch}
            pendingAction={pendingAction}
            runAction={runAction}
            mutateEnabled={mutateCapability.enabled}
          />
        )}
      </div>
    </Page>
  )
}

function EvalsReadinessOverview({ readiness }: { readiness: EvalsReadiness }) {
  return (
    <DataCard
      title="Readiness"
      description="Server-published operation availability. Underlying routes still enforce authentication and runtime policy."
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

function CatalogView({
  search,
  updateSearch,
  pendingAction,
  runAction,
  mutateEnabled,
}: {
  search: EvalsSearch
  updateSearch: UpdateEvalsSearch
  pendingAction: string | null
  runAction: (
    name: string,
    action: (signal: AbortSignal) => Promise<string | undefined>,
  ) => Promise<void>
  mutateEnabled: boolean
}) {
  const queryClient = useQueryClient()
  const [maxConcurrency, setMaxConcurrency] = useState("1")
  const launchRegistryRef = useRef<EvalLaunchIdempotencyRegistry | null>(null)
  const corpora = useQuery({
    queryKey: ["evals", "corpora", search.corpora_cursor],
    queryFn: ({ signal }) =>
      fetchEvalCorpora({ limit: PAGE_LIMIT, cursor: search.corpora_cursor }, signal),
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
    Number.isInteger(parsedConcurrency) && parsedConcurrency >= 1 && parsedConcurrency <= 32

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
    if (!search.corpus || pendingAction !== null || !mutateEnabled || !concurrencyIsValid) {
      return
    }
    const requestIdentity = evalLaunchRequestIdentity(search.corpus, suiteId, parsedConcurrency)
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
            max_concurrency: parsedConcurrency,
          },
          idempotencyKey,
          signal,
        )
      } catch (error) {
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
                      max={32}
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
                          aria-label={`Run suite ${suite.name} (${suite.id})`}
                          disabled={!mutateEnabled || !concurrencyIsValid || pendingAction !== null}
                          onClick={() => launchSuite(suite.id)}
                        >
                          {pendingAction === `launch:${suite.id}` ? (
                            <LoaderCircle className="animate-spin" />
                          ) : (
                            <Play />
                          )}
                          Run suite
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

function RunsView({
  search,
  updateSearch,
  pendingAction,
  runAction,
  mutateEnabled,
}: {
  search: EvalsSearch
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
    queryKey: ["evals", "runs", search.status, search.corpus, search.runs_cursor],
    queryFn: ({ signal }) =>
      fetchEvalRuns(
        {
          limit: PAGE_LIMIT,
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
  const comparison = useQuery({
    queryKey: ["evals", "comparison", search.baseline, search.run],
    queryFn: ({ signal }) => compareEvalRuns(search.baseline ?? "", search.run ?? "", signal),
    enabled:
      result.data !== undefined &&
      search.baseline !== undefined &&
      search.run !== undefined &&
      search.baseline !== search.run,
    staleTime: Number.POSITIVE_INFINITY,
  })
  const comparisonCandidates =
    runs.data?.items.filter(
      (run) =>
        run.spec.run_id !== search.run &&
        run.status === "completed" &&
        run.result !== null &&
        run.result !== undefined,
    ) ?? []
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
            canMutate={mutateEnabled}
            cancel={cancelRun}
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
                  key={`${search.run}:${search.baseline ?? ""}`}
                  currentRunId={search.run ?? ""}
                  baselineRunId={search.baseline}
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
  canMutate,
  cancel,
}: {
  run: EvalRun
  cancelling: boolean
  canMutate: boolean
  cancel: () => void
}) {
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
      </div>
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

      {selectedCase && selectedTrial && (
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
              value={evalTrialCostSummary(selectedTrial.assertions)}
            />
          </div>

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
                {selectedTrial.assertions.map((assertion) => (
                  <TableRow key={assertion.assertion_id}>
                    <TableCell>
                      <div className="max-w-52 truncate font-medium" title={assertion.assertion_id}>
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
                      <details className="mt-2 text-xs">
                        <summary className="cursor-pointer text-primary">Inspect evidence</summary>
                        <PayloadViewer
                          value={assertion.detail}
                          className="mt-2"
                          maxHeight="max-h-40"
                        />
                      </details>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      )}
    </DataCard>
  )
}

function ComparisonPanel({
  currentRunId,
  baselineRunId,
  candidates,
  comparison,
  loading,
  error,
  retry,
  selectBaseline,
}: {
  currentRunId: string
  baselineRunId?: string
  candidates: Array<EvalRun>
  comparison?: EvalComparison
  loading: boolean
  error: unknown
  retry: () => void
  selectBaseline: (runId?: string) => void
}) {
  const [draft, setDraft] = useState(baselineRunId ?? "")
  const normalizedDraft = draft.trim()
  const baselineIsSelf = baselineRunId === currentRunId
  const draftIsValid = evalRunIdIsValid(normalizedDraft) && normalizedDraft !== currentRunId

  return (
    <DataCard
      title="Compare runs"
      description="The server decides whether two immutable results share a compatible eval contract."
    >
      <form
        className="flex min-w-0 flex-wrap items-end gap-2 border-b border-border p-4"
        onSubmit={(event) => {
          event.preventDefault()
          if (draftIsValid) selectBaseline(normalizedDraft)
        }}
      >
        <label className="min-w-64 flex-1 text-xs font-medium text-muted-foreground">
          Baseline run ID
          <input
            type="text"
            list="eval-baseline-candidates"
            value={draft}
            maxLength={128}
            className="mt-1 h-9 w-full rounded-md border border-input bg-background px-3 text-sm text-foreground"
            placeholder="Select or paste a completed run ID"
            aria-invalid={normalizedDraft.length > 0 && !draftIsValid}
            onChange={(event) => setDraft(event.target.value)}
          />
          <datalist id="eval-baseline-candidates">
            {candidates.map((run) => (
              <option key={run.spec.run_id} value={run.spec.run_id}>
                {run.spec.suite_id} · {run.result?.status}
              </option>
            ))}
          </datalist>
        </label>
        <Button type="submit" size="sm" disabled={!draftIsValid || loading}>
          {loading ? <LoaderCircle className="animate-spin" /> : <FlaskConical />}
          Compare
        </Button>
        {baselineRunId && (
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => selectBaseline(undefined)}
          >
            Clear
          </Button>
        )}
      </form>

      {!baselineRunId ? (
        <StateMessage>
          Choose another completed result. Different application releases are allowed when the
          corpus, suite, cases, assertions, evidence policy, and applicable pricing contract match.
        </StateMessage>
      ) : baselineIsSelf ? (
        <StateMessage tone="danger" role="alert">
          Choose a different completed run as the comparison baseline.
        </StateMessage>
      ) : loading ? (
        <LoadingState label="Checking comparison compatibility..." />
      ) : error ? (
        <QueryError message="Could not compare the selected eval runs." retry={retry} />
      ) : comparison ? (
        <ComparisonSummary comparison={comparison} />
      ) : null}
    </DataCard>
  )
}

function ComparisonSummary({ comparison }: { comparison: EvalComparison }) {
  const { comparison: resultComparison, baseline, current } = comparison
  const { compatibility } = resultComparison
  const regressions = resultComparison.regressions ?? []
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
            ? "These runs are comparable."
            : "These runs are not comparable as one regression contract."}
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
        <ComparisonRunSummary label="Baseline" run={baseline} result={resultComparison.baseline} />
        <ComparisonRunSummary label="Current" run={current} result={resultComparison.current} />
      </div>
      {compatibility.comparable && (
        <div
          className={
            regressions.length === 0
              ? "rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3 text-sm"
              : "rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm"
          }
          data-testid="eval-comparison-regressions"
        >
          <div className="font-medium">
            {regressions.length === 0
              ? "No compatible-result regressions."
              : `${regressions.length} compatible-result regression${regressions.length === 1 ? "" : "s"}.`}
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
        </div>
      )}
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

function ComparisonRunSummary({
  label,
  run,
  result,
}: {
  label: string
  run: EvalRun
  result: EvalComparison["comparison"]["baseline"]
}) {
  return (
    <div className="rounded-lg border border-border p-3">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="font-medium">{label}</div>
        <OutcomeBadge outcome={result.status} />
      </div>
      <div className="grid gap-3 text-sm sm:grid-cols-2">
        <RunFact label="Run" value={shortEvalIdentity(run.spec.run_id)} />
        <RunFact label="Suite" value={run.spec.suite_id} />
        <RunFact label="Release" value={result.application_release_id} />
        <RunFact label="App manifest" value={shortEvalIdentity(result.app_manifest_fingerprint)} />
        <RunFact label="Score" value={formatScore(result.score)} />
        <RunFact
          label="Duration"
          value={run.result ? formatDuration(run.result.duration_ms) : "unavailable"}
        />
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

function RunFact({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <div className="text-xs font-medium text-muted-foreground">{label}</div>
      <div className="mt-1 truncate" title={value}>
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
