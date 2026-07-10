import { keepPreviousData, useQuery } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  ExternalLink,
  FileArchive,
  FileText,
  ImageIcon,
  Search,
} from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"
import {
  DataCard,
  Page,
  PageHeader,
  PayloadViewer,
  StateMessage,
} from "../components/dashboard/layout"
import { Badge } from "../components/ui/badge"
import { Button, buttonVariants } from "../components/ui/button"
import { Input } from "../components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table"
import {
  type ArtifactRead,
  type ArtifactSummary,
  type ArtifactsQuery,
  artifactContentUrl,
  fetchArtifact,
  fetchArtifacts,
} from "../lib/api"
import { formatBytes, formatCount, formatDateTime } from "../lib/format"
import { currentQueryParam, dashboardPath, replaceDashboardLocation } from "../lib/links"
import { cn } from "../lib/utils"

type ArtifactScopeFilter = "all" | "session" | "environment"
type ArtifactCopyFeedback = {
  artifactRef: string
  status: "copied" | "failed"
}

const PAGE_LIMIT = 100
const selectClassName =
  "h-8 min-w-32 rounded-lg border border-input bg-background px-2.5 py-1 text-sm outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
const safeInlineContentTypes = new Set([
  "application/json",
  "application/pdf",
  "image/gif",
  "image/jpeg",
  "image/png",
  "image/webp",
  "text/csv",
  "text/markdown",
  "text/plain",
])
const safePreviewImageContentTypes = new Set(["image/gif", "image/jpeg", "image/png", "image/webp"])

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const handle = window.setTimeout(() => setDebounced(value), delayMs)
    return () => window.clearTimeout(handle)
  }, [delayMs, value])
  return debounced
}

function artifactMatchesSearch(artifact: ArtifactSummary, query: string) {
  const needle = query.trim().toLowerCase()
  if (!needle) return true
  return [
    artifact.id,
    artifact.filename,
    artifact.content_type,
    artifact.scope,
    artifact.session_id,
    artifact.agent_name,
    artifact.environment_name,
    artifact.artifact_store_id,
  ].some((value) => value?.toLowerCase().includes(needle))
}

function artifactIcon(artifact: ArtifactSummary) {
  const contentType = normalizedContentType(artifact.content_type)
  if (contentType.startsWith("image/")) return ImageIcon
  if (contentType.startsWith("text/") || contentType.includes("json")) {
    return FileText
  }
  return FileArchive
}

function artifactReference(artifact: ArtifactSummary) {
  return JSON.stringify({
    artifact_store_id: artifact.artifact_store_id,
    artifact_id: artifact.id,
  })
}

function artifactKey(artifact: ArtifactSummary) {
  return artifactReference(artifact)
}

function normalizedContentType(contentType: string) {
  return contentType.split(";", 1)[0]?.trim().toLowerCase() ?? contentType
}

function artifactDataUrl(read: ArtifactRead) {
  const contentType = normalizedContentType(read.artifact.content_type)
  if (!safePreviewImageContentTypes.has(contentType) || read.truncated) {
    return null
  }
  return `data:${contentType};base64,${read.preview_base64}`
}

function artifactCanOpenInline(artifact: ArtifactSummary) {
  return safeInlineContentTypes.has(normalizedContentType(artifact.content_type))
}

function artifactInventoryDescription(
  data: Awaited<ReturnType<typeof fetchArtifacts>> | undefined,
) {
  const loaded = data?.artifacts.length ?? 0
  const total = data?.total_count
  const offset = data?.offset ?? 0
  if (total === null || total === undefined) {
    if (data?.truncated && data.next_offset === null) {
      return `${formatCount(loaded)} loaded from offset ${formatCount(offset)}; narrow filters to continue`
    }
    return `${formatCount(loaded)} loaded from offset ${formatCount(offset)}`
  }
  if (total === 0) return "0 artifacts"
  const first = loaded === 0 ? 0 : offset + 1
  const last = offset + loaded
  if (data?.truncated && data.next_offset === null) {
    return `${formatCount(first)}-${formatCount(last)} of ${formatCount(total)} artifacts; narrow filters to continue`
  }
  return `${formatCount(first)}-${formatCount(last)} of ${formatCount(total)} artifacts`
}

function pageCount(data: Awaited<ReturnType<typeof fetchArtifacts>> | undefined) {
  const total = data?.total_count
  if (total === null || total === undefined || total === 0 || !data) return null
  if (data.truncated && data.next_offset === null) {
    return Math.floor(data.offset / data.limit) + 1
  }
  return Math.max(1, Math.ceil(total / data.limit))
}

function currentPage(data: Awaited<ReturnType<typeof fetchArtifacts>> | undefined) {
  const total = data?.total_count
  if (total === null || total === undefined || total === 0 || !data) return null
  return Math.floor(data.offset / data.limit) + 1
}

function clampOffset(offset: number) {
  return Math.max(0, offset)
}

function PaginationControls({
  data,
  onPrevious,
  onNext,
}: {
  data: Awaited<ReturnType<typeof fetchArtifacts>> | undefined
  onPrevious: () => void
  onNext: () => void
}) {
  const page = currentPage(data)
  const pages = pageCount(data)
  const canGoPrevious = (data?.offset ?? 0) > 0
  const canGoNext = data?.next_offset !== null && data?.next_offset !== undefined

  return (
    <div className="flex items-center gap-2">
      {page !== null && pages !== null && (
        <span className="hidden text-xs text-muted-foreground sm:inline">
          Page {formatCount(page)} / {formatCount(pages)}
        </span>
      )}
      <Button
        type="button"
        variant="outline"
        size="icon"
        aria-label="Previous artifacts page"
        disabled={!canGoPrevious}
        onClick={onPrevious}
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>
      <Button
        type="button"
        variant="outline"
        size="icon"
        aria-label="Next artifacts page"
        disabled={!canGoNext}
        onClick={onNext}
      >
        <ChevronRight className="h-4 w-4" />
      </Button>
    </div>
  )
}

function scopeFilter(value: string): ArtifactScopeFilter {
  if (value === "session" || value === "environment") {
    return value
  }
  return "all"
}

function optionalFilter(value: string) {
  const trimmed = value.trim()
  return trimmed === "" ? undefined : trimmed
}

async function copyText(value: string): Promise<void> {
  if (!navigator.clipboard?.writeText) {
    throw new Error("The Clipboard API is unavailable.")
  }
  await navigator.clipboard.writeText(value)
}

function ArtifactPreview({ read }: { read: ArtifactRead | null }) {
  if (read === null) {
    return (
      <StateMessage className="py-16">Select an artifact to preview bounded content.</StateMessage>
    )
  }
  const imageUrl = artifactDataUrl(read)
  if (imageUrl) {
    return (
      <div className="rounded-md border border-border bg-muted/20 p-3">
        <img
          src={imageUrl}
          alt={read.artifact.filename}
          className="max-h-96 max-w-full rounded-sm object-contain"
        />
      </div>
    )
  }
  if (read.text_preview !== null) {
    return (
      <div className="overflow-auto rounded-md border border-border bg-muted/20 p-3">
        <pre className="max-h-96 whitespace-pre-wrap break-words font-mono text-sm leading-relaxed">
          {read.text_preview}
        </pre>
      </div>
    )
  }
  return (
    <StateMessage className="rounded-md border border-border py-12">
      Binary preview unavailable for {read.artifact.content_type}.
    </StateMessage>
  )
}

function ArtifactDetail({
  artifact,
  read,
  isLoading,
  error,
}: {
  artifact: ArtifactSummary | null
  read: ArtifactRead | null
  isLoading: boolean
  error: string | null
}) {
  const artifactRef = artifact === null ? null : artifactReference(artifact)
  const [copyFeedback, setCopyFeedback] = useState<ArtifactCopyFeedback | null>(null)
  const copyResetTimer = useRef<number | null>(null)
  const activeArtifactRef = useRef(artifactRef)
  const copyAttempt = useRef(0)

  useEffect(() => {
    activeArtifactRef.current = artifactRef
    return () => {
      activeArtifactRef.current = null
      copyAttempt.current += 1
      if (copyResetTimer.current !== null) {
        window.clearTimeout(copyResetTimer.current)
        copyResetTimer.current = null
      }
    }
  }, [artifactRef])

  if (artifact === null) {
    return (
      <StateMessage className="py-16">
        Select an artifact to inspect metadata and content.
      </StateMessage>
    )
  }
  const contentQuery = {
    artifact_store_id: artifact.artifact_store_id,
  }
  const artifactId = artifact.id
  const downloadUrl = artifactContentUrl(artifactId, {
    ...contentQuery,
    disposition: "attachment",
  })
  const inlineUrl = artifactContentUrl(artifactId, {
    ...contentQuery,
    disposition: "inline",
  })
  const selectedArtifactRef = artifactReference(artifact)
  const selectedCopyFeedback =
    copyFeedback?.artifactRef === selectedArtifactRef ? copyFeedback.status : null

  function showCopyFeedback(status: ArtifactCopyFeedback["status"], attempt: number) {
    if (activeArtifactRef.current !== selectedArtifactRef || copyAttempt.current !== attempt) return
    if (copyResetTimer.current !== null) {
      window.clearTimeout(copyResetTimer.current)
    }
    setCopyFeedback({ artifactRef: selectedArtifactRef, status })
    copyResetTimer.current = window.setTimeout(() => {
      setCopyFeedback(null)
      copyResetTimer.current = null
    }, 1200)
  }

  async function copyArtifactRef() {
    const attempt = copyAttempt.current + 1
    copyAttempt.current = attempt
    try {
      await copyText(selectedArtifactRef)
      showCopyFeedback("copied", attempt)
    } catch {
      showCopyFeedback("failed", attempt)
    }
  }

  return (
    <div className="space-y-5 p-4 sm:p-5">
      <div>
        <div className="flex min-w-0 flex-wrap items-center justify-between gap-3">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <Badge variant="secondary">{artifact.scope}</Badge>
            <Badge variant="outline">{artifact.content_type}</Badge>
            {read?.truncated && <Badge variant="outline">preview truncated</Badge>}
          </div>
          <div className="flex shrink-0 flex-wrap gap-2">
            <Button variant="outline" size="sm" onClick={copyArtifactRef}>
              {selectedCopyFeedback === "copied" ? (
                <Check className="h-4 w-4" />
              ) : (
                <Copy className="h-4 w-4" />
              )}
              {selectedCopyFeedback === "copied"
                ? "Copied"
                : selectedCopyFeedback === "failed"
                  ? "Copy failed"
                  : "Copy reference"}
            </Button>
            {artifactCanOpenInline(artifact) && (
              <a
                href={inlineUrl}
                target="_blank"
                rel="noreferrer"
                className={buttonVariants({ variant: "outline", size: "sm" })}
              >
                <ExternalLink className="h-4 w-4" />
                Open raw
              </a>
            )}
            <a
              href={downloadUrl}
              target="_blank"
              rel="noreferrer"
              className={buttonVariants({ variant: "default", size: "sm" })}
            >
              <Download className="h-4 w-4" />
              Download
            </a>
          </div>
        </div>
        <h2 className="mt-3 break-words text-xl font-semibold">{artifact.filename}</h2>
        <div className="mt-1 font-mono text-xs text-muted-foreground">{artifact.id}</div>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-md border border-border p-3">
          <div className="text-sm font-medium text-muted-foreground">Size</div>
          <div className="mt-1 text-2xl font-semibold">{formatBytes(artifact.size_bytes)}</div>
          <div className="mt-1 text-xs text-muted-foreground">
            {read
              ? read.truncated
                ? "bounded preview loaded"
                : "complete preview loaded"
              : "metadata only"}
          </div>
        </div>
        <div className="rounded-md border border-border p-3">
          <div className="text-sm font-medium text-muted-foreground">Store</div>
          <div className="mt-1 truncate font-mono text-sm font-medium">
            {artifact.artifact_store_id}
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            {formatDateTime(artifact.created_at)}
          </div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(16rem,0.7fr)]">
        <div>
          <div className="mb-2 flex items-center justify-between gap-3">
            <div className="text-sm font-medium">Preview</div>
            {read?.truncated && (
              <span className="text-xs text-muted-foreground">
                Preview is truncated; raw open/download is bounded by the server content limit.
              </span>
            )}
          </div>
          {isLoading ? (
            <StateMessage className="rounded-md border border-border py-12">
              Loading preview...
            </StateMessage>
          ) : error ? (
            <StateMessage tone="danger" className="rounded-md border border-border py-12">
              {error}
            </StateMessage>
          ) : (
            <ArtifactPreview read={read} />
          )}
        </div>
        <div>
          <div className="mb-2 text-sm font-medium">Metadata</div>
          <PayloadViewer
            value={{
              session_id: artifact.session_id,
              agent_name: artifact.agent_name,
              environment_name: artifact.environment_name,
              metadata: artifact.metadata,
            }}
            maxHeight="max-h-96"
          />
          <div className="mt-3 flex flex-wrap gap-2">
            {artifact.session_id && (
              <Button
                variant="outline"
                size="sm"
                render={
                  <Link to="/sessions/$sessionId" params={{ sessionId: artifact.session_id }} />
                }
              >
                Session
              </Button>
            )}
            {artifact.agent_name && (
              <a
                href={dashboardPath("/agents", { q: artifact.agent_name })}
                className={buttonVariants({ variant: "outline", size: "sm" })}
              >
                Agent
              </a>
            )}
            {artifact.environment_name && (
              <a
                href={dashboardPath("/environments", { q: artifact.environment_name })}
                className={buttonVariants({ variant: "outline", size: "sm" })}
              >
                Environment
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export function ArtifactsPage() {
  const [search, setSearch] = useState(() => currentQueryParam("q"))
  const [scope, setScope] = useState<ArtifactScopeFilter>(() =>
    scopeFilter(currentQueryParam("scope")),
  )
  const [artifactStoreId, setArtifactStoreId] = useState(() =>
    currentQueryParam("artifact_store_id"),
  )
  const [sessionFilter, setSessionFilter] = useState(() => currentQueryParam("session_id"))
  const [agentFilter, setAgentFilter] = useState(() => currentQueryParam("agent_name"))
  const [environmentFilter, setEnvironmentFilter] = useState(() =>
    currentQueryParam("environment_name"),
  )
  const [offset, setOffset] = useState(0)
  const [selectedArtifactKey, setSelectedArtifactKey] = useState<string | null>(null)
  const debouncedSearch = useDebouncedValue(search, 300)

  const query = useMemo<ArtifactsQuery>(
    () => ({
      limit: PAGE_LIMIT,
      offset,
      agent_name: optionalFilter(agentFilter),
      artifact_store_id: optionalFilter(artifactStoreId),
      environment_name: optionalFilter(environmentFilter),
      session_id: optionalFilter(sessionFilter),
      scope: scope === "all" ? undefined : scope,
    }),
    [agentFilter, artifactStoreId, environmentFilter, offset, scope, sessionFilter],
  )

  const artifacts = useQuery({
    queryKey: ["artifacts", query],
    queryFn: () => fetchArtifacts(query),
    placeholderData: keepPreviousData,
    refetchInterval: 10_000,
  })

  const artifactList = useMemo(() => {
    const items = artifacts.data?.artifacts ?? []
    return items.filter((artifact) => artifactMatchesSearch(artifact, debouncedSearch))
  }, [artifacts.data?.artifacts, debouncedSearch])

  useEffect(() => {
    if (
      selectedArtifactKey &&
      artifactList.some((artifact) => artifactKey(artifact) === selectedArtifactKey)
    ) {
      return
    }
    setSelectedArtifactKey(artifactList[0] ? artifactKey(artifactList[0]) : null)
  }, [artifactList, selectedArtifactKey])

  const selectedArtifact =
    artifactList.find((artifact) => artifactKey(artifact) === selectedArtifactKey) ??
    artifactList[0] ??
    null

  const read = useQuery({
    queryKey: ["artifact", selectedArtifact?.artifact_store_id, selectedArtifact?.id],
    queryFn: () =>
      fetchArtifact(selectedArtifact?.id ?? "", {
        artifact_store_id: selectedArtifact?.artifact_store_id,
        max_bytes: 64_000,
      }),
    enabled: selectedArtifact !== null,
    staleTime: 30_000,
  })

  const listError = artifacts.error instanceof Error ? artifacts.error.message : null
  const readError = read.error instanceof Error ? read.error.message : null
  const SelectedIcon = selectedArtifact ? artifactIcon(selectedArtifact) : FileArchive
  const hasServerFilters =
    artifactStoreId.trim() !== "" ||
    sessionFilter.trim() !== "" ||
    agentFilter.trim() !== "" ||
    environmentFilter.trim() !== ""

  function clearServerFilters() {
    setArtifactStoreId("")
    setSessionFilter("")
    setAgentFilter("")
    setEnvironmentFilter("")
    setOffset(0)
    replaceDashboardLocation("/artifacts", {
      q: optionalFilter(search),
      scope: scope === "all" ? undefined : scope,
    })
  }

  return (
    <Page>
      <PageHeader
        title="Artifacts"
        description="Browse stored files, generated artifacts, attachment references, and bounded previews."
      />

      <div className="grid min-h-[calc(100vh-10rem)] gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(30rem,1.25fr)]">
        <DataCard
          title="Artifact Inventory"
          description={artifactInventoryDescription(artifacts.data)}
          actions={
            <>
              <select
                value={scope}
                onChange={(event) => {
                  setScope(scopeFilter(event.target.value))
                  setOffset(0)
                }}
                className={selectClassName}
              >
                <option value="all">All scopes</option>
                <option value="session">Session</option>
                <option value="environment">Environment</option>
              </select>
              {hasServerFilters && (
                <Button variant="outline" size="sm" onClick={clearServerFilters}>
                  Clear links
                </Button>
              )}
              <PaginationControls
                data={artifacts.data}
                onPrevious={() => setOffset((value) => clampOffset(value - PAGE_LIMIT))}
                onNext={() => {
                  if (
                    artifacts.data?.next_offset !== null &&
                    artifacts.data?.next_offset !== undefined
                  ) {
                    setOffset(artifacts.data.next_offset)
                  }
                }}
              />
            </>
          }
        >
          <div className="border-b border-border p-4">
            <div className="relative">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Filter loaded page by file, session, agent, environment..."
                className="pl-8"
              />
            </div>
            {hasServerFilters && (
              <div className="mt-3 flex flex-wrap gap-2 text-xs">
                {artifactStoreId && <Badge variant="outline">store: {artifactStoreId}</Badge>}
                {sessionFilter && <Badge variant="outline">session: {sessionFilter}</Badge>}
                {agentFilter && <Badge variant="outline">agent: {agentFilter}</Badge>}
                {environmentFilter && (
                  <Badge variant="outline">environment: {environmentFilter}</Badge>
                )}
              </div>
            )}
          </div>
          {artifacts.isLoading ? (
            <StateMessage>Loading artifacts...</StateMessage>
          ) : listError ? (
            <StateMessage tone="danger">{listError}</StateMessage>
          ) : artifactList.length === 0 ? (
            <StateMessage>No artifacts match the current filters.</StateMessage>
          ) : (
            <div className="overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Artifact</TableHead>
                    <TableHead>Owner</TableHead>
                    <TableHead className="text-right">Size</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {artifactList.map((artifact) => {
                    const Icon = artifactIcon(artifact)
                    return (
                      <TableRow
                        key={artifactKey(artifact)}
                        className={cn(
                          "cursor-pointer",
                          selectedArtifactKey === artifactKey(artifact) && "bg-muted/60",
                        )}
                        onClick={() => setSelectedArtifactKey(artifactKey(artifact))}
                      >
                        <TableCell>
                          <div className="flex min-w-0 items-center gap-2">
                            <Icon className="h-4 w-4 shrink-0 text-muted-foreground" />
                            <div className="min-w-0">
                              <div className="truncate font-medium">{artifact.filename}</div>
                              <div className="truncate text-xs text-muted-foreground">
                                {artifact.content_type}
                              </div>
                            </div>
                          </div>
                        </TableCell>
                        <TableCell className="min-w-40">
                          <div className="truncate text-sm">
                            {artifact.session_id ?? artifact.environment_name ?? "-"}
                          </div>
                          <div className="truncate text-xs text-muted-foreground">
                            {artifact.agent_name ?? artifact.scope}
                          </div>
                        </TableCell>
                        <TableCell className="text-right">
                          {formatBytes(artifact.size_bytes)}
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </DataCard>

        <DataCard
          title="Artifact Detail"
          description="Content previews are bounded; full binary bytes stay in the artifact store."
          actions={
            selectedArtifact ? (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <SelectedIcon className="h-4 w-4" />
                {selectedArtifact.scope}
              </div>
            ) : null
          }
        >
          <ArtifactDetail
            artifact={selectedArtifact}
            read={read.data ?? null}
            isLoading={read.isLoading}
            error={readError}
          />
        </DataCard>
      </div>
    </Page>
  )
}
