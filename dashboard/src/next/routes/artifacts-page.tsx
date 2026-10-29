/**
 * Artifacts: the files the agent's sessions saved. Filters live in the URL (specialist,
 * environment, session, scope, page offset); a file opens in a drawer (`?artifact=<id>`, with
 * `preview_store_id`, since IDs are only unique within a store). The separate
 * `artifact_store_id` filters the list and remains supported in direct preview bookmarks.
 * The drawer offers a preview, Download and Open raw. The API has no text search,
 * so the search box filters the loaded page only.
 */
import { keepPreviousData, useQuery } from "@tanstack/react-query"
import { useSearch } from "@tanstack/react-router"
import { ChevronLeft, ChevronRight, Search, X } from "lucide-react"
import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { type ArtifactSummary, fetchArtifacts } from "@/lib/api"
import type { ArtifactSearch } from "@/lib/artifact-search"
import { formatBytes } from "@/lib/format"
import { cn } from "@/lib/utils"
import type { SessionOverview } from "../api/types"
import { useEnvironments, useSpecialists } from "../app/queries"
import { useApi, useNow, useServices } from "../app/services"
import { relativeTime } from "../app/session-model"
import { ArtifactDrawer } from "./artifacts/artifact-drawer"
import { kindOf, sessionTitle } from "./artifacts/artifact-model"
import {
  EmptyState,
  ErrorState,
  NavLink,
  PageBody,
  ROW_CLASS,
  ROW_LINK_CLASS,
  sessionRoute,
  TopBar,
  useGo,
  useRowClick,
} from "./common"

const PAGE_SIZE = 25
const SCOPES = [
  { id: "session", label: "Saved by a session" },
  { id: "environment", label: "Shared in an environment" },
] as const

const scopeOf = (value: string | undefined): "session" | "environment" | undefined =>
  value === "session" || value === "environment" ? value : undefined

const offsetOf = (value: string | undefined) => {
  const n = Number(value)
  return Number.isSafeInteger(n) && n > 0 ? n : 0
}

const field =
  "h-8 max-w-full rounded-lg border border-input bg-background px-2 text-sm outline-none focus:border-ring"

export function ArtifactsPage() {
  const search = useSearch({ from: "/artifacts" })
  const { overviewQueriesEnabled = true } = useServices()
  const go = useGo()
  const rowClick = useRowClick()
  const now = useNow()
  const api = useApi()
  const [draft, setDraft] = useState(search.q ?? "")
  useEffect(() => setDraft(search.q ?? ""), [search.q])
  const set = (patch: Partial<ArtifactSearch>) =>
    go({ to: "/artifacts", search: { ...search, ...patch }, replace: true, resetScroll: false })
  /** A filter change goes back to the first page. */
  const filter = (patch: Partial<ArtifactSearch>) => set({ ...patch, offset: undefined })

  const offset = offsetOf(search.offset)
  const scope = scopeOf(search.scope)
  const query = {
    session_id: search.session_id,
    agent_name: search.agent_name,
    environment_name: search.environment_name,
    scope,
    artifact_store_id: search.artifact_store_id,
    limit: PAGE_SIZE,
    offset,
  }
  const artifacts = useQuery({
    queryKey: ["artifacts", query],
    queryFn: () => fetchArtifacts(query),
    placeholderData: keepPreviousData,
  })
  const specialists = useSpecialists()
  const environments = useEnvironments()

  const loaded = artifacts.data?.artifacts ?? []
  const sessionIds = [
    ...new Set([
      ...loaded.flatMap((a) => (a.session_id ? [a.session_id] : [])),
      ...(search.session_id ? [search.session_id] : []),
    ]),
  ].sort()
  const overviews = useQuery({
    queryKey: ["artifact-session-overviews", sessionIds],
    enabled: overviewQueriesEnabled && sessionIds.length > 0,
    placeholderData: keepPreviousData,
    queryFn: () => api.overviews(sessionIds),
  })
  const overviewOf = (id: string | null) =>
    id && overviewQueriesEnabled ? overviews.data?.find((o) => o.session_id === id) : undefined

  const needle = (search.q ?? "").trim().toLocaleLowerCase()
  const rows = needle
    ? loaded.filter((a) =>
        [
          a.filename,
          kindOf(a).label,
          a.content_type,
          a.agent_name,
          a.environment_name,
          a.session_id ? sessionTitle(overviewOf(a.session_id)) : null,
          a.id,
        ].some((value) => value?.toLocaleLowerCase().includes(needle)),
      )
    : loaded
  const total = artifacts.data?.total_count
  const filtered = !!(
    search.session_id ||
    search.agent_name ||
    search.environment_name ||
    search.artifact_store_id ||
    scope
  )
  const agents = [
    ...new Set([
      ...(specialists.data?.agents ?? []).map((a) => a.name),
      ...(search.agent_name ? [search.agent_name] : []),
    ]),
  ]
  const environmentNames = [
    ...new Set([
      ...(environments.data?.environments ?? []).map((e) => e.name),
      ...(search.environment_name ? [search.environment_name] : []),
    ]),
  ]
  const open = search.artifact
  const previewStoreId = search.preview_store_id ?? search.artifact_store_id
  const isOpen = (a: ArtifactSummary) =>
    a.id === open && !!previewStoreId && a.artifact_store_id === previewStoreId
  const openRoute = (a: ArtifactSummary) =>
    ({
      to: "/artifacts",
      search: { ...search, artifact: a.id, preview_store_id: a.artifact_store_id ?? undefined },
    }) as const

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar>Artifacts</TopBar>
      <PageBody width="max-w-6xl">
        <p className="text-sm text-muted-foreground">
          Files your agent's sessions saved: reports, data, images and archives.
        </p>

        <div className="mt-4 flex flex-wrap items-center gap-2">
          <label className="flex h-8 w-full min-w-0 flex-none items-center gap-2 rounded-lg border border-input bg-background px-2.5 text-sm focus-within:border-ring sm:max-w-sm sm:min-w-56 sm:flex-[1_1_14rem]">
            <Search className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
            <span className="sr-only">Search files</span>
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") set({ q: e.currentTarget.value || undefined })
              }}
              onBlur={(e) => set({ q: e.currentTarget.value || undefined })}
              placeholder="Search by file name, kind or session"
              className="grouped-field-input w-full min-w-0 bg-transparent outline-none placeholder:text-muted-foreground"
            />
          </label>
          <select
            aria-label="Specialist"
            className={field}
            value={search.agent_name ?? ""}
            onChange={(e) => filter({ agent_name: e.target.value || undefined })}
          >
            <option value="">All specialists</option>
            {agents.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
          <select
            aria-label="Environment"
            className={field}
            value={search.environment_name ?? ""}
            onChange={(e) => filter({ environment_name: e.target.value || undefined })}
          >
            <option value="">All environments</option>
            {environmentNames.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          <select
            aria-label="Saved by"
            className={field}
            value={scope ?? ""}
            onChange={(e) => filter({ scope: scopeOf(e.target.value) })}
          >
            <option value="">Any owner</option>
            {SCOPES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
          {(filtered || search.q) && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setDraft("")
                go({ to: "/artifacts", search: {}, replace: true, resetScroll: false })
              }}
            >
              Clear filters
            </Button>
          )}
        </div>

        {search.session_id && (
          <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
            <span className="inline-flex h-7 max-w-full items-center gap-1.5 rounded-full border border-border bg-muted/50 pr-1 pl-3">
              <span className="shrink-0 text-muted-foreground">Session</span>
              <span className="min-w-0 truncate">
                {overviews.isFetching ? "…" : sessionTitle(overviewOf(search.session_id))}
              </span>
              <button
                type="button"
                aria-label="Show files from every session"
                onClick={() => filter({ session_id: undefined })}
                className="flex size-5 shrink-0 items-center justify-center rounded-full hover:bg-muted"
              >
                <X className="size-3.5" aria-hidden="true" />
              </button>
            </span>
          </div>
        )}

        {needle && total !== null && total !== undefined && total > loaded.length && (
          <p className="mt-3 text-xs text-muted-foreground">
            Search looks at the {loaded.length} files on this page only.
          </p>
        )}

        {artifacts.error && (
          <div className="mt-4">
            <ErrorState error={artifacts.error} />
          </div>
        )}
        {artifacts.isPending && (
          <p role="status" className="mt-4 text-sm text-muted-foreground">
            Loading files…
          </p>
        )}

        {artifacts.isSuccess && rows.length === 0 ? (
          <div className="mt-4">
            {!filtered && !needle && offset === 0 ? (
              <EmptyState title="No files yet">
                Files show up here when an agent saves output, like a report or a chart, to its
                environment's ArtifactStore.
              </EmptyState>
            ) : (
              <EmptyState title="No files match">
                {needle && loaded.length > 0
                  ? `None of the files on this page match “${search.q}”.`
                  : "Try removing a filter."}
              </EmptyState>
            )}
          </div>
        ) : (
          rows.length > 0 && (
            <div
              className={cn(
                "mt-3 overflow-x-auto rounded-xl border border-border",
                artifacts.isPlaceholderData && "opacity-60",
              )}
            >
              <table className="w-full text-sm sm:min-w-[720px]">
                <thead>
                  <tr className="border-b border-border text-left text-xs text-muted-foreground">
                    <th className="px-3 py-2 font-medium">File</th>
                    <th className="hidden px-3 py-2 font-medium sm:table-cell">Kind</th>
                    <th className="hidden px-3 py-2 text-right font-medium sm:table-cell">Size</th>
                    <th className="hidden px-3 py-2 font-medium sm:table-cell">Made by</th>
                    <th className="px-3 py-2 text-right font-medium">Saved</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((a) => {
                    const kind = kindOf(a)
                    const Icon = kind.icon
                    return (
                      <tr
                        key={`${a.artifact_store_id}:${a.id}`}
                        onClick={rowClick(openRoute(a))}
                        className={cn(ROW_CLASS, isOpen(a) && "bg-muted/60")}
                      >
                        <td className="max-w-[22rem] px-3 py-2 max-sm:max-w-[17rem]">
                          <div className="flex min-w-0 items-center gap-2">
                            <Icon
                              className="size-4 shrink-0 text-muted-foreground"
                              aria-hidden="true"
                            />
                            <NavLink
                              {...openRoute(a)}
                              replace
                              resetScroll={false}
                              className={cn("truncate", ROW_LINK_CLASS)}
                            >
                              {a.filename}
                            </NavLink>
                          </div>
                          <div className="mt-0.5 truncate pl-6 text-xs text-muted-foreground sm:hidden">
                            {kind.label} · {formatBytes(a.size_bytes)} ·{" "}
                            {a.session_id ? sessionTitle(overviewOf(a.session_id)) : "Shared file"}
                          </div>
                        </td>
                        <td className="hidden px-3 py-2 whitespace-nowrap text-muted-foreground sm:table-cell">
                          {kind.label}
                        </td>
                        <td className="hidden px-3 py-2 text-right whitespace-nowrap tabular-nums text-muted-foreground sm:table-cell">
                          {formatBytes(a.size_bytes)}
                        </td>
                        <td className="hidden max-w-[20rem] px-3 py-2 sm:table-cell">
                          <MadeBy artifact={a} overview={overviewOf(a.session_id)} />
                        </td>
                        <td className="px-3 py-2 text-right whitespace-nowrap tabular-nums text-muted-foreground">
                          {relativeTime(Date.parse(a.created_at), now())}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )
        )}

        {artifacts.isSuccess && (loaded.length > 0 || offset > 0) && (
          <Pager
            offset={offset}
            count={loaded.length}
            total={total ?? null}
            next={artifacts.data.next_offset ?? null}
            onPage={(next) => set({ offset: next ? String(next) : undefined })}
          />
        )}
      </PageBody>

      {open && (
        <ArtifactDrawer
          key={`${previewStoreId ?? ""}:${open}`}
          artifactId={open}
          storeId={previewStoreId}
          known={loaded.find(isOpen)}
          onClose={() => set({ artifact: undefined, preview_store_id: undefined })}
        />
      )}
    </div>
  )
}

function MadeBy({
  artifact,
  overview,
}: {
  artifact: ArtifactSummary
  overview: SessionOverview | undefined
}) {
  if (!artifact.session_id)
    return (
      <div className="min-w-0">
        <div className="truncate text-muted-foreground">Shared file</div>
        <div className="truncate text-xs text-muted-foreground">
          {artifact.environment_name ? `In ${artifact.environment_name}` : "No session"}
        </div>
      </div>
    )
  return (
    <div className="min-w-0">
      <NavLink
        {...sessionRoute(artifact.session_id)}
        className="block truncate underline-offset-4 hover:underline"
      >
        {sessionTitle(overview)}
      </NavLink>
      <div className="truncate text-xs text-muted-foreground">{artifact.agent_name ?? "–"}</div>
    </div>
  )
}

function Pager({
  offset,
  count,
  total,
  next,
  onPage,
}: {
  offset: number
  count: number
  total: number | null
  next: number | null
  onPage: (offset: number) => void
}) {
  const first = count ? offset + 1 : 0
  const last = offset + count
  return (
    <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
      <span>
        {count === 0
          ? "No files on this page"
          : `${first}–${last}${total !== null ? ` of ${total}` : ""} ${total === 1 ? "file" : "files"}`}
      </span>
      {(offset > 0 || next !== null) && (
        <span className="flex gap-1">
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="Previous page"
            disabled={offset === 0}
            onClick={() => onPage(Math.max(0, offset - PAGE_SIZE))}
          >
            <ChevronLeft />
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="Next page"
            disabled={next === null}
            onClick={() => next !== null && onPage(next)}
          >
            <ChevronRight />
          </Button>
        </span>
      )}
    </div>
  )
}
