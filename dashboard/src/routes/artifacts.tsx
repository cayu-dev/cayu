import { keepPreviousData, useQuery } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import { ChevronLeft, ChevronRight, FileArchive, FileText, ImageIcon, Search } from "lucide-react"
import { useEffect, useMemo, useState } from "react"
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
  fetchArtifact,
  fetchArtifacts,
} from "../lib/api"
import { formatBytes, formatCount, formatDateTime } from "../lib/format"
import { currentQueryParam, dashboardPath, replaceDashboardLocation } from "../lib/links"
import { cn } from "../lib/utils"

type ArtifactScopeFilter = "all" | "session" | "environment"

const PAGE_LIMIT = 100
const selectClassName =
  "h-8 min-w-32 rounded-lg border border-input bg-background px-2.5 py-1 text-sm outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"

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
  if (artifact.content_type.startsWith("image/")) return ImageIcon
  if (artifact.content_type.startsWith("text/") || artifact.content_type.includes("json")) {
    return FileText
  }
  return FileArchive
}

function artifactKey(artifact: ArtifactSummary) {
  return `${artifact.artifact_store_id}:${artifact.id}`
}

function artifactDataUrl(read: ArtifactRead) {
  if (!read.artifact.content_type.startsWith("image/") || read.truncated) return null
  return `data:${read.artifact.content_type};base64,${read.preview_base64}`
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
    return <PayloadViewer value={read.text_preview} maxHeight="max-h-96" />
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
  if (artifact === null) {
    return (
      <StateMessage className="py-16">
        Select an artifact to inspect metadata and content.
      </StateMessage>
    )
  }

  return (
    <div className="space-y-5 p-4 sm:p-5">
      <div>
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <Badge variant="secondary">{artifact.scope}</Badge>
          <Badge variant="outline">{artifact.content_type}</Badge>
          {read?.truncated && <Badge variant="outline">preview truncated</Badge>}
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
          <div className="mb-2 text-sm font-medium">Preview</div>
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
                        key={`${artifact.artifact_store_id}:${artifact.id}`}
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
