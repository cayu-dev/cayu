import { useQuery } from "@tanstack/react-query"
import { Link, useNavigate, useSearch } from "@tanstack/react-router"
import { Search, SlidersHorizontal } from "lucide-react"
import { useCallback, useEffect, useMemo, useState } from "react"
import { DataCard, Page, PageHeader, StateMessage } from "../components/dashboard/layout"
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
import { Textarea } from "../components/ui/textarea"
import { fetchSessionsSummary, type SessionsSummary } from "../lib/api"
import { formatDateTime } from "../lib/format"
import { dashboardPath } from "../lib/links"
import { summarizeSessionDebugState } from "../lib/session-debug"
import { sessionIndexErrorState } from "../lib/session-index-error"
import {
  DEFAULT_SESSION_ORDER,
  multilineSessionFilters,
  SESSION_INDEX_DEBUG_STATES,
  SESSION_INDEX_ORDERS,
  SESSION_INDEX_STATUSES,
  type SessionIndexFilterPatch,
  type SessionIndexSearch,
  sessionIndexAdvancedFilterKey,
  sessionIndexFilterCount,
  sessionIndexQuery,
  sessionIndexSearchWithCursor,
  sessionIndexSearchWithFilters,
  sessionIndexSearchWithoutFilters,
} from "../lib/session-index-search"

type SessionSummaryItem = SessionsSummary["sessions"][number]

type AdvancedFilterDraft = {
  agentName: string
  providerName: string
  model: string
  environmentName: string
  parentSessionId: string
  causalBudgetId: string
  labels: string
  labelSelectors: string
}

const selectClassName =
  "h-8 min-w-32 rounded-lg border border-input bg-background px-2.5 py-1 text-sm outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"

const advancedInputClassName = "mt-1.5"

const ORDER_LABELS: Record<(typeof SESSION_INDEX_ORDERS)[number], string> = {
  updated_at_desc: "Updated newest",
  updated_at_asc: "Updated oldest",
  created_at_desc: "Created newest",
  created_at_asc: "Created oldest",
}

const DEBUG_LABELS: Record<(typeof SESSION_INDEX_DEBUG_STATES)[number], string> = {
  needs_attention: "Needs attention",
  session_failure: "Session failures",
  tool_issue: "Tool issues",
  interruption: "Interruptions",
}

function StatusBadge({ status }: { status: string }) {
  const variant =
    status === "completed"
      ? "default"
      : status === "running"
        ? "secondary"
        : status === "failed"
          ? "destructive"
          : "outline"
  return <Badge variant={variant}>{status}</Badge>
}

function DebugBadge({ item }: { item: SessionSummaryItem }) {
  const summary = summarizeSessionDebugState({
    status: item.session.status,
    latestEvent: item.events.latest_event,
    terminalEvent: item.outcome.terminal_event,
    countsByType: item.events.counts_by_type,
  })
  if (!summary) {
    return <span className="text-sm text-muted-foreground">-</span>
  }
  return (
    <div className="min-w-0">
      <Badge
        variant={summary.kind === "interruption" ? "outline" : "destructive"}
        className="max-w-44 truncate"
      >
        {summary.label}
      </Badge>
      <div className="mt-1 max-w-56 truncate text-xs text-muted-foreground">{summary.detail}</div>
    </div>
  )
}

function optionalFilter(value: string): string | undefined {
  const trimmed = value.trim()
  return trimmed === "" ? undefined : trimmed
}

function advancedFilterDraft(search: SessionIndexSearch): AdvancedFilterDraft {
  return {
    agentName: search.agent_name ?? "",
    providerName: search.provider_name ?? "",
    model: search.model ?? "",
    environmentName: search.environment_name ?? "",
    parentSessionId: search.parent_session_id ?? "",
    causalBudgetId: search.causal_budget_id ?? "",
    labels: search.label?.join("\n") ?? "",
    labelSelectors: search.label_selector?.join("\n") ?? "",
  }
}

function advancedFilterDraftKey(draft: AdvancedFilterDraft): string {
  return JSON.stringify(Object.values(draft).map((value) => value.trim()))
}

function advancedFilterPatch(draft: AdvancedFilterDraft): SessionIndexFilterPatch {
  return {
    agent_name: optionalFilter(draft.agentName),
    provider_name: optionalFilter(draft.providerName),
    model: optionalFilter(draft.model),
    environment_name: optionalFilter(draft.environmentName),
    parent_session_id: optionalFilter(draft.parentSessionId),
    causal_budget_id: optionalFilter(draft.causalBudgetId),
    label: multilineSessionFilters(draft.labels),
    label_selector: multilineSessionFilters(draft.labelSelectors),
  }
}

function activeAdvancedFilterCount(search: SessionIndexSearch): number {
  return (
    [
      search.agent_name,
      search.provider_name,
      search.model,
      search.environment_name,
      search.parent_session_id,
      search.causal_budget_id,
    ].filter((value) => value !== undefined).length +
    (search.label?.length ?? 0) +
    (search.label_selector?.length ?? 0)
  )
}

function SessionSearchInput({
  value,
  onCommit,
}: {
  value: string
  onCommit: (value: string | undefined) => void
}) {
  const [draft, setDraft] = useState(value)

  useEffect(() => {
    const normalized = optionalFilter(draft)
    if (normalized === (optionalFilter(value) ?? undefined)) return
    const timer = window.setTimeout(() => onCommit(normalized), 300)
    return () => window.clearTimeout(timer)
  }, [draft, onCommit, value])

  return (
    <div className="relative min-w-56 flex-[2_1_18rem]">
      <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
      <Input
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        placeholder="Search sessions, agents, providers, models, labels..."
        className="pl-8"
        aria-label="Search sessions"
      />
    </div>
  )
}

function AdvancedFilters({
  search,
  onApply,
}: {
  search: SessionIndexSearch
  onApply: (patch: SessionIndexFilterPatch) => void
}) {
  const confirmedDraft = advancedFilterDraft(search)
  const [draft, setDraft] = useState(confirmedDraft)
  const [open, setOpen] = useState(activeAdvancedFilterCount(search) > 0)
  const confirmedKey = advancedFilterDraftKey(confirmedDraft)
  const dirty = advancedFilterDraftKey(draft) !== confirmedKey
  const activeCount = activeAdvancedFilterCount(search)

  function setField<Key extends keyof AdvancedFilterDraft>(
    field: Key,
    value: AdvancedFilterDraft[Key],
  ) {
    setDraft((current) => ({ ...current, [field]: value }))
  }

  return (
    <details
      className="min-w-full border-t border-border pt-3"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="flex cursor-pointer list-none items-center gap-2 text-sm font-medium text-foreground">
        <SlidersHorizontal className="h-4 w-4" />
        Advanced filters
        {activeCount > 0 && <Badge variant="secondary">{activeCount}</Badge>}
      </summary>
      <form
        className="mt-4 space-y-4"
        onSubmit={(event) => {
          event.preventDefault()
          onApply(advancedFilterPatch(draft))
        }}
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <label htmlFor="session-agent-filter" className="text-sm text-muted-foreground">
            Agent
            <Input
              id="session-agent-filter"
              value={draft.agentName}
              onChange={(event) => setField("agentName", event.target.value)}
              placeholder="agent name"
              className={advancedInputClassName}
            />
          </label>
          <label htmlFor="session-environment-filter" className="text-sm text-muted-foreground">
            Environment
            <Input
              id="session-environment-filter"
              value={draft.environmentName}
              onChange={(event) => setField("environmentName", event.target.value)}
              placeholder="environment name"
              className={advancedInputClassName}
            />
          </label>
          <label htmlFor="session-provider-filter" className="text-sm text-muted-foreground">
            Provider
            <Input
              id="session-provider-filter"
              value={draft.providerName}
              onChange={(event) => setField("providerName", event.target.value)}
              placeholder="provider name"
              className={advancedInputClassName}
            />
          </label>
          <label htmlFor="session-model-filter" className="text-sm text-muted-foreground">
            Model
            <Input
              id="session-model-filter"
              value={draft.model}
              onChange={(event) => setField("model", event.target.value)}
              placeholder="model name"
              className={advancedInputClassName}
            />
          </label>
          <label htmlFor="session-parent-filter" className="text-sm text-muted-foreground">
            Parent session
            <Input
              id="session-parent-filter"
              value={draft.parentSessionId}
              onChange={(event) => setField("parentSessionId", event.target.value)}
              placeholder="parent session ID"
              className={advancedInputClassName}
            />
          </label>
          <label htmlFor="session-budget-filter" className="text-sm text-muted-foreground">
            Causal budget
            <Input
              id="session-budget-filter"
              value={draft.causalBudgetId}
              onChange={(event) => setField("causalBudgetId", event.target.value)}
              placeholder="causal budget ID"
              className={advancedInputClassName}
            />
          </label>
        </div>
        <div className="grid gap-3 lg:grid-cols-2">
          <label htmlFor="session-label-filter" className="text-sm text-muted-foreground">
            Exact labels
            <Textarea
              id="session-label-filter"
              value={draft.labels}
              onChange={(event) => setField("labels", event.target.value)}
              placeholder={"tenant=acme\ntier=critical"}
              className={advancedInputClassName}
              rows={3}
            />
            <span className="mt-1 block text-xs">One key=value label per line.</span>
          </label>
          <label htmlFor="session-selector-filter" className="text-sm text-muted-foreground">
            Label selectors
            <Textarea
              id="session-selector-filter"
              value={draft.labelSelectors}
              onChange={(event) => setField("labelSelectors", event.target.value)}
              placeholder={"region in (us,eu)\n!archived"}
              className={advancedInputClassName}
              rows={3}
            />
            <span className="mt-1 block text-xs">One selector expression per line.</span>
          </label>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button type="submit" size="sm" disabled={!dirty}>
            Apply filters
          </Button>
          {dirty && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setDraft(confirmedDraft)}
            >
              Discard changes
            </Button>
          )}
        </div>
      </form>
    </details>
  )
}

export function SessionsPage() {
  const navigate = useNavigate({ from: "/sessions" })
  const search = useSearch({ from: "/sessions" })
  const [filterResetVersion, setFilterResetVersion] = useState(0)
  const requestQuery = useMemo(() => sessionIndexQuery(search), [search])
  const filterCount = sessionIndexFilterCount(search)
  const advancedKey = sessionIndexAdvancedFilterKey(search)
  const currentCursor = search.cursor
  const orderBy = search.order_by ?? DEFAULT_SESSION_ORDER

  const updateFilters = useCallback(
    (patch: SessionIndexFilterPatch, replace = false) => {
      void navigate({
        search: (current) => sessionIndexSearchWithFilters(current, patch),
        replace,
        resetScroll: false,
      })
    },
    [navigate],
  )

  const commitSearch = useCallback(
    (q: string | undefined) => updateFilters({ q }, true),
    [updateFilters],
  )

  const sessions = useQuery({
    queryKey: ["sessions-summary", "sessions-page", requestQuery],
    queryFn: ({ signal }) => fetchSessionsSummary(requestQuery, {}, signal),
    gcTime: 0,
    retry: (failureCount, error) => sessionIndexErrorState(error).retryable && failureCount < 3,
  })
  const errorState = sessions.isError ? sessionIndexErrorState(sessions.error) : null
  const list = sessions.data?.sessions ?? []
  const totalCount = sessions.data?.total_count
  const nextCursor = sessions.data?.next_cursor
  const confirmedPageDescription = `${list.length} on this page${totalCount != null ? ` · ${totalCount} matching` : ""}${currentCursor ? " · later page" : " · first page"}`
  const pageDescription = sessions.isLoading
    ? "Loading a bounded session page..."
    : sessions.isError
      ? "Session page unavailable"
      : confirmedPageDescription

  function clearFilters() {
    setFilterResetVersion((current) => current + 1)
    void navigate({
      search: (current) => sessionIndexSearchWithoutFilters(current),
      resetScroll: false,
    })
  }

  function showFirstPage() {
    void navigate({
      search: (current) => sessionIndexSearchWithCursor(current, undefined),
      resetScroll: false,
    })
  }

  function showNextPage() {
    if (!nextCursor || sessions.isFetching) return
    void navigate({
      search: (current) => sessionIndexSearchWithCursor(current, nextCursor),
      resetScroll: false,
    })
  }

  return (
    <Page>
      <PageHeader
        title="Sessions"
        actions={
          <Link to="/run" className={buttonVariants()}>
            New Run
          </Link>
        }
      />

      <DataCard title="Sessions" description={pageDescription}>
        <div className="flex min-w-0 flex-wrap items-center gap-2 border-b border-border p-4">
          <SessionSearchInput
            key={`${search.q ?? ""}:${filterResetVersion}`}
            value={search.q ?? ""}
            onCommit={commitSearch}
          />
          <select
            value={search.status ?? "all"}
            onChange={(event) =>
              updateFilters({
                status:
                  event.target.value === "all"
                    ? undefined
                    : (event.target.value as SessionIndexSearch["status"]),
              })
            }
            className={selectClassName}
            aria-label="Filter by status"
          >
            <option value="all">All statuses</option>
            {SESSION_INDEX_STATUSES.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
          <select
            value={search.debug_state ?? "all"}
            onChange={(event) =>
              updateFilters({
                debug_state:
                  event.target.value === "all"
                    ? undefined
                    : (event.target.value as SessionIndexSearch["debug_state"]),
              })
            }
            className={selectClassName}
            aria-label="Filter by debug state"
          >
            <option value="all">All debug states</option>
            {SESSION_INDEX_DEBUG_STATES.map((value) => (
              <option key={value} value={value}>
                {DEBUG_LABELS[value]}
              </option>
            ))}
          </select>
          <select
            value={orderBy}
            onChange={(event) =>
              updateFilters({ order_by: event.target.value as SessionIndexSearch["order_by"] })
            }
            className={selectClassName}
            aria-label="Sort sessions"
          >
            {SESSION_INDEX_ORDERS.map((value) => (
              <option key={value} value={value}>
                {ORDER_LABELS[value]}
              </option>
            ))}
          </select>
          {filterCount > 0 && (
            <Button variant="outline" size="sm" onClick={clearFilters}>
              Clear filters
            </Button>
          )}
          <AdvancedFilters
            key={`${advancedKey}:${filterResetVersion}`}
            search={search}
            onApply={(patch) => updateFilters(patch)}
          />
        </div>

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Session ID</TableHead>
              <TableHead>Agent</TableHead>
              <TableHead>Environment</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Attention</TableHead>
              <TableHead>Provider</TableHead>
              <TableHead>Model</TableHead>
              <TableHead>Updated</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sessions.isLoading ? (
              <TableRow>
                <TableCell colSpan={8}>
                  <StateMessage>Loading session page...</StateMessage>
                </TableCell>
              </TableRow>
            ) : sessions.isError ? (
              <TableRow>
                <TableCell colSpan={8}>
                  <StateMessage tone="danger">
                    <div
                      className="space-y-2"
                      data-session-index-error-kind={errorState?.kind ?? "unexpected"}
                      role="alert"
                    >
                      <p className="font-medium">
                        {errorState?.title ?? "The session page could not be loaded."}
                      </p>
                      <p>{errorState?.detail ?? "An unexpected response was received."}</p>
                      {errorState?.retryable && (
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => void sessions.refetch()}
                          disabled={sessions.isFetching}
                        >
                          {sessions.isFetching ? "Retrying..." : "Retry"}
                        </Button>
                      )}
                    </div>
                  </StateMessage>
                </TableCell>
              </TableRow>
            ) : !list.length ? (
              <TableRow>
                <TableCell colSpan={8}>
                  <StateMessage>
                    {filterCount > 0
                      ? "No sessions match the current filters."
                      : currentCursor
                        ? "This session page is empty. Return to the first page or go back."
                        : "No sessions yet"}
                  </StateMessage>
                </TableCell>
              </TableRow>
            ) : (
              list.map((item) => (
                <TableRow key={item.session.id} className="cursor-pointer">
                  <TableCell>
                    <Link
                      to="/sessions/$sessionId"
                      params={{ sessionId: item.session.id }}
                      className="font-mono text-sm text-primary hover:underline"
                    >
                      {item.session.id}
                    </Link>
                  </TableCell>
                  <TableCell className="text-sm">{item.session.agent_name}</TableCell>
                  <TableCell className="max-w-44 truncate text-sm text-muted-foreground">
                    {item.session.environment_name ? (
                      <a
                        href={dashboardPath("/environments", {
                          q: item.session.environment_name,
                        })}
                        className="text-primary hover:underline"
                      >
                        {item.session.environment_name}
                      </a>
                    ) : (
                      "-"
                    )}
                  </TableCell>
                  <TableCell>
                    <StatusBadge status={item.session.status} />
                  </TableCell>
                  <TableCell>
                    <DebugBadge item={item} />
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {item.session.provider_name}
                  </TableCell>
                  <TableCell className="max-w-56 truncate text-sm text-muted-foreground">
                    {item.session.model}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {formatDateTime(item.session.updated_at || item.session.created_at)}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>

        {(currentCursor || nextCursor) && (
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border p-4">
            <span className="text-sm text-muted-foreground" aria-live="polite">
              {sessions.isFetching ? "Loading session page..." : pageDescription}
            </span>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={showFirstPage}
                disabled={!currentCursor || sessions.isFetching}
              >
                First page
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={showNextPage}
                disabled={!nextCursor || sessions.isFetching}
              >
                Next page
              </Button>
            </div>
          </div>
        )}
      </DataCard>
    </Page>
  )
}
