import { useQuery, useQueryClient } from "@tanstack/react-query"
import { useNavigate, useSearch } from "@tanstack/react-router"
import { Database, Download, FlaskConical, LoaderCircle, RotateCcw, Upload } from "lucide-react"
import { type ChangeEvent, useCallback, useEffect, useRef, useState } from "react"
import { DataCard, Page, PageHeader, StateMessage } from "@/components/dashboard/layout"
import { useDashboardCapability } from "@/components/dashboard/server-contract"
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
  downloadEvalCorpus,
  type EvalCorpusEntry,
  fetchEvalCases,
  fetchEvalCorpora,
  fetchEvalSuites,
  importEvalCorpus,
} from "@/lib/api"
import { dashboardCapabilityUnavailableText } from "@/lib/dashboard-capabilities"
import { evalErrorMessage, parseEvalCorpusFile, shortEvalIdentity } from "@/lib/evals-dashboard"
import { type EvalsSearch, evalsSearchWithout } from "@/lib/evals-search"
import { formatBytes, formatCount, formatDateTime } from "@/lib/format"

const PAGE_LIMIT = 25

export function EvalsPage() {
  const search = useSearch({ from: "/evals" })
  const navigate = useNavigate({ from: "/evals" })
  const queryClient = useQueryClient()
  const mutateCapability = useDashboardCapability({
    kind: "surface",
    surface: "evals",
    operation: "mutate",
  })
  const mutationUnavailable = dashboardCapabilityUnavailableText(mutateCapability)
  const fileInputRef = useRef<HTMLInputElement>(null)
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
    (next: (current: EvalsSearch) => EvalsSearch) => {
      void navigate({ search: next, resetScroll: false })
    },
    [navigate],
  )

  const importCorpus = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (!file || pendingAction !== null) return
    void runAction("import", async (signal) => {
      const corpus = await parseEvalCorpusFile(file)
      const imported = await importEvalCorpus(corpus, signal)
      await queryClient.invalidateQueries({ queryKey: ["evals", "corpora"] })
      updateSearch((current) => ({
        ...evalsSearchWithout(current, "suite", "suites_cursor", "cases_cursor", "corpora_cursor"),
        tab: "catalog",
        corpus: imported.revision,
      }))
      return `Imported corpus ${shortEvalIdentity(imported.revision)}.`
    })
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

      <div className="flex gap-2 border-b border-border" role="tablist" aria-label="Evals views">
        <Button
          role="tab"
          aria-selected={(search.tab ?? "catalog") === "catalog"}
          variant="ghost"
          className="rounded-b-none"
          onClick={() =>
            updateSearch((current) => ({
              ...evalsSearchWithout(current, "run", "baseline", "runs_cursor", "status"),
              tab: "catalog",
            }))
          }
        >
          <Database /> Catalog
        </Button>
        <Button
          role="tab"
          aria-selected={search.tab === "runs"}
          variant="ghost"
          className="rounded-b-none"
          onClick={() =>
            updateSearch((current) => ({
              ...evalsSearchWithout(
                current,
                "suite",
                "suites_cursor",
                "cases_cursor",
                "corpora_cursor",
              ),
              tab: "runs",
            }))
          }
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

      {(search.tab ?? "catalog") === "catalog" ? (
        <CatalogView
          search={search}
          updateSearch={updateSearch}
          pendingAction={pendingAction}
          runAction={runAction}
        />
      ) : (
        <StateMessage className="rounded-lg border border-border bg-muted/30 py-12">
          Durable eval runs are loading.
        </StateMessage>
      )}
    </Page>
  )
}

function CatalogView({
  search,
  updateSearch,
  pendingAction,
  runAction,
}: {
  search: EvalsSearch
  updateSearch: (next: (current: EvalsSearch) => EvalsSearch) => void
  pendingAction: string | null
  runAction: (
    name: string,
    action: (signal: AbortSignal) => Promise<string | undefined>,
  ) => Promise<void>
}) {
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

  const downloadCorpus = () => {
    if (!search.corpus || pendingAction !== null) return
    void runAction("download-corpus", async (signal) => {
      const file = await downloadEvalCorpus(search.corpus ?? "", signal)
      downloadBlob(file.blob, file.filename)
      return `Downloaded corpus ${shortEvalIdentity(search.corpus ?? "")}.`
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
              }
            >
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Suite</TableHead>
                    <TableHead>Trials</TableHead>
                    <TableHead>Coverage</TableHead>
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
  cursor,
  nextCursor,
  fetching,
  first,
  next,
}: {
  cursor?: string
  nextCursor?: string | null
  fetching: boolean
  first: () => void
  next: (cursor: string) => void
}) {
  if (!cursor && !nextCursor) return null
  return (
    <div className="flex items-center justify-end gap-2 border-t border-border p-3">
      <Button
        type="button"
        size="sm"
        variant="outline"
        disabled={!cursor || fetching}
        onClick={first}
      >
        First page
      </Button>
      <Button
        type="button"
        size="sm"
        variant="outline"
        disabled={!nextCursor || fetching}
        onClick={() => nextCursor && next(nextCursor)}
      >
        Next page
      </Button>
    </div>
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
