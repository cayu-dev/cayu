import { useQuery } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import { AlertTriangle, ExternalLink, Search, ShieldCheck, UserRound } from "lucide-react"
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
  fetchPendingActions,
  isApiPayloadTooLarge,
  type PendingAction,
  type PendingActionIssue,
  type PendingActionsQuery,
} from "../lib/api"
import { formatDateTime } from "../lib/format"
import { cn } from "../lib/utils"

type PendingActionKind = "all" | "tool_approval" | "user_input" | "manual_recovery"

const PAGE_LIMIT_OPTIONS = [1, 5, 10, 25, 50, 100] as const
const DEFAULT_PAGE_LIMIT = 100
const selectClassName =
  "h-8 min-w-36 rounded-lg border border-input bg-background px-2.5 py-1 text-sm outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const handle = window.setTimeout(() => setDebounced(value), delayMs)
    return () => window.clearTimeout(handle)
  }, [delayMs, value])
  return debounced
}

function optionalFilter(value: string) {
  const trimmed = value.trim()
  return trimmed === "" ? undefined : trimmed
}

function actionIcon(kind: PendingAction["kind"]) {
  if (kind === "tool_approval") return ShieldCheck
  if (kind === "user_input") return UserRound
  return AlertTriangle
}

function actionTone(kind: PendingAction["kind"]) {
  if (kind === "tool_approval") return "text-chart-1 border-chart-1/30 bg-chart-1/10"
  if (kind === "user_input") return "text-chart-3 border-chart-3/30 bg-chart-3/10"
  return "text-destructive border-destructive/30 bg-destructive/10"
}

function actionLabel(kind: PendingAction["kind"]) {
  if (kind === "tool_approval") return "Approval"
  if (kind === "user_input") return "User input"
  return "Manual recovery"
}

function actionDetail(action: PendingAction) {
  if (action.question) return action.question
  if (action.detail) return action.detail
  if (action.tool_name) return action.tool_name
  return action.title
}

function ActionBadge({ action }: { action: PendingAction }) {
  const Icon = actionIcon(action.kind)
  return (
    <Badge variant="outline" className={cn("gap-1", actionTone(action.kind))}>
      <Icon className="h-3.5 w-3.5" />
      {actionLabel(action.kind)}
    </Badge>
  )
}

function idLine(action: PendingAction) {
  if (action.approval_id) return `approval_id: ${action.approval_id}`
  if (action.input_id) return `input_id: ${action.input_id}`
  if (action.round_id) return `round_id: ${action.round_id}`
  if (action.tool_call_id) return `tool_call_id: ${action.tool_call_id}`
  return action.event.id
}

function PendingActionRow({ action }: { action: PendingAction }) {
  return (
    <TableRow>
      <TableCell>
        <div className="min-w-0 space-y-2">
          <ActionBadge action={action} />
          <div className="font-medium">{action.title}</div>
          <div className="max-w-[28rem] truncate text-sm text-muted-foreground">
            {actionDetail(action)}
          </div>
          <div className="break-all font-mono text-xs text-muted-foreground">{idLine(action)}</div>
        </div>
      </TableCell>
      <TableCell>
        <Link
          to="/sessions/$sessionId"
          params={{ sessionId: action.session.id }}
          className="font-mono text-sm text-primary hover:underline"
        >
          {action.session.id}
        </Link>
        <div className="mt-1 text-xs text-muted-foreground">{action.session.status}</div>
      </TableCell>
      <TableCell className="text-sm">
        <div>{action.session.agent_name}</div>
        <div className="mt-1 max-w-52 truncate text-xs text-muted-foreground">
          {action.session.provider_name ?? "unknown provider"} /{" "}
          {action.session.model ?? "unknown model"}
        </div>
      </TableCell>
      <TableCell className="text-sm text-muted-foreground">
        <div>{action.tool_name ?? "-"}</div>
        {action.tool_call_id && (
          <div className="mt-1 max-w-44 truncate font-mono text-xs">{action.tool_call_id}</div>
        )}
      </TableCell>
      <TableCell className="text-sm text-muted-foreground">
        {formatDateTime(action.event.timestamp)}
      </TableCell>
      <TableCell>
        <Link
          to="/sessions/$sessionId"
          params={{ sessionId: action.session.id }}
          className={buttonVariants({ variant: "outline", size: "sm" })}
        >
          <ExternalLink className="mr-1.5 h-3.5 w-3.5" />
          {action.kind === "manual_recovery" ? "Recover" : "Open"}
        </Link>
      </TableCell>
    </TableRow>
  )
}

function PendingActionIssueRow({ issue }: { issue: PendingActionIssue }) {
  return (
    <TableRow>
      <TableCell>
        <div className="min-w-0 space-y-2">
          <Badge
            variant="outline"
            className="gap-1 border-destructive/30 bg-destructive/10 text-destructive"
          >
            <AlertTriangle className="h-3.5 w-3.5" />
            Inspection required
          </Badge>
          <div className="max-w-[28rem] text-sm text-muted-foreground">{issue.detail}</div>
        </div>
      </TableCell>
      <TableCell>
        <Link
          to="/sessions/$sessionId"
          params={{ sessionId: issue.session_id }}
          className="break-all font-mono text-sm text-primary hover:underline"
        >
          {issue.session_id}
        </Link>
        <div className="mt-1 text-xs text-muted-foreground">{issue.status}</div>
      </TableCell>
      <TableCell className="text-sm">{issue.agent_name}</TableCell>
      <TableCell className="text-sm text-muted-foreground">-</TableCell>
      <TableCell className="text-sm text-muted-foreground">
        {formatDateTime(issue.updated_at)}
      </TableCell>
      <TableCell>
        <Link
          to="/sessions/$sessionId"
          params={{ sessionId: issue.session_id }}
          className={buttonVariants({ variant: "outline", size: "sm" })}
        >
          <ExternalLink className="mr-1.5 h-3.5 w-3.5" />
          Inspect
        </Link>
      </TableCell>
    </TableRow>
  )
}

export function PendingActionsPage() {
  const [search, setSearch] = useState("")
  const [kind, setKind] = useState<PendingActionKind>("all")
  const [pageLimit, setPageLimit] = useState(DEFAULT_PAGE_LIMIT)
  const debouncedSearch = useDebouncedValue(search, 300)
  const query = useMemo<Omit<PendingActionsQuery, "cursor">>(
    () => ({
      limit: pageLimit,
      q: optionalFilter(debouncedSearch),
      kind: kind === "all" ? undefined : kind,
    }),
    [debouncedSearch, kind, pageLimit],
  )
  const [pageCursors, setPageCursors] = useState<(string | undefined)[]>([undefined])
  const pageCursor = pageCursors.at(-1)
  const actions = useQuery({
    queryKey: ["pending-actions", query, pageCursor],
    queryFn: () => fetchPendingActions({ ...query, cursor: pageCursor }),
    retry: (failureCount, error) => !isApiPayloadTooLarge(error) && failureCount < 3,
    refetchInterval: (activeQuery) =>
      isApiPayloadTooLarge(activeQuery.state.error) ? false : 5000,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: (activeQuery) => !isApiPayloadTooLarge(activeQuery.state.error),
  })
  const list = actions.data?.actions ?? []
  const issues = actions.data?.issues ?? []
  const totalCount = actions.data?.total_count
  const pageNumber = pageCursors.length
  const hasFilters = search.trim() !== "" || kind !== "all"
  const errorMessage =
    actions.error instanceof Error ? actions.error.message : "Failed to load pending actions."

  function clearFilters() {
    setSearch("")
    setKind("all")
    setPageCursors([undefined])
  }

  return (
    <Page>
      <PageHeader
        title="Pending Actions"
        description="Session-blocking approvals, user questions, and manual recovery items."
      />

      <DataCard
        title="Action Queue"
        description={`${list.length + issues.length} queue items shown on page ${pageNumber}${totalCount != null ? ` · ${totalCount} total` : ""}`}
      >
        <div className="flex min-w-0 flex-wrap items-center gap-2 border-b border-border p-4">
          <div className="relative min-w-56 flex-[2_1_18rem]">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(event) => {
                setSearch(event.target.value)
                setPageCursors([undefined])
              }}
              placeholder="Search pending actions..."
              className="pl-8"
            />
          </div>
          <select
            value={kind}
            onChange={(event) => {
              setKind(event.target.value as PendingActionKind)
              setPageCursors([undefined])
            }}
            className={selectClassName}
            aria-label="Filter by action type"
          >
            <option value="all">All action types</option>
            <option value="tool_approval">Approvals</option>
            <option value="user_input">User input</option>
            <option value="manual_recovery">Manual recovery</option>
          </select>
          <label className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>Page size</span>
            <select
              value={pageLimit}
              onChange={(event) => {
                setPageLimit(Number(event.target.value))
                setPageCursors([undefined])
              }}
              className={selectClassName}
              aria-label="Pending actions page size"
            >
              {PAGE_LIMIT_OPTIONS.map((limit) => (
                <option key={limit} value={limit}>
                  {limit}
                </option>
              ))}
            </select>
          </label>
          {hasFilters && (
            <Button variant="outline" size="sm" onClick={clearFilters}>
              Clear
            </Button>
          )}
        </div>

        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Action</TableHead>
              <TableHead>Session</TableHead>
              <TableHead>Agent</TableHead>
              <TableHead>Tool</TableHead>
              <TableHead>Created</TableHead>
              <TableHead>Open</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {actions.isLoading ? (
              <TableRow>
                <TableCell colSpan={6}>
                  <StateMessage>Loading pending actions...</StateMessage>
                </TableCell>
              </TableRow>
            ) : actions.isError ? (
              <TableRow>
                <TableCell colSpan={6}>
                  <StateMessage tone="danger">{errorMessage}</StateMessage>
                </TableCell>
              </TableRow>
            ) : !list.length && !issues.length ? (
              <TableRow>
                <TableCell colSpan={6}>
                  <StateMessage>
                    {hasFilters
                      ? "No pending actions match the current filters."
                      : "No pending actions."}
                  </StateMessage>
                </TableCell>
              </TableRow>
            ) : (
              <>
                {issues.map((issue) => (
                  <PendingActionIssueRow key={`${issue.code}:${issue.session_id}`} issue={issue} />
                ))}
                {list.map((action) => (
                  <PendingActionRow key={action.id} action={action} />
                ))}
              </>
            )}
          </TableBody>
        </Table>
        {(pageNumber > 1 || actions.data?.has_more) && (
          <div className="flex items-center justify-center gap-2 border-t border-border p-4">
            <Button
              variant="outline"
              disabled={pageNumber === 1 || actions.isFetching}
              onClick={() => setPageCursors((current) => current.slice(0, -1))}
            >
              Previous
            </Button>
            <span className="px-2 text-sm text-muted-foreground">Page {pageNumber}</span>
            <Button
              variant="outline"
              disabled={!actions.data?.has_more || !actions.data.next_cursor || actions.isFetching}
              onClick={() => {
                const nextCursor = actions.data?.next_cursor
                if (nextCursor) setPageCursors((current) => [...current, nextCursor])
              }}
            >
              Next
            </Button>
          </div>
        )}
      </DataCard>

      {list.some((action) => action.kind === "manual_recovery") && (
        <DataCard
          title="Manual Recovery Payloads"
          description="Manual recovery means a tool may have started before the runtime recorded its final result."
        >
          <div className="grid gap-4 p-4 xl:grid-cols-2">
            {list
              .filter((action) => action.kind === "manual_recovery")
              .map((action) => (
                <div
                  key={action.id}
                  className="min-w-0 space-y-2 rounded-md border border-border p-3"
                >
                  <div className="flex min-w-0 items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate font-mono text-sm">{action.session.id}</div>
                      <div className="text-xs text-muted-foreground">{idLine(action)}</div>
                    </div>
                    <Link
                      to="/sessions/$sessionId"
                      params={{ sessionId: action.session.id }}
                      className="text-sm text-primary hover:underline"
                    >
                      Open session
                    </Link>
                  </div>
                  <div className="space-y-2">
                    <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Tool arguments
                    </div>
                    <PayloadViewer value={action.arguments ?? {}} maxHeight="max-h-64" />
                  </div>
                </div>
              ))}
          </div>
        </DataCard>
      )}
    </Page>
  )
}
