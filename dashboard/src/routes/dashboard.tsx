import { useQuery } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import { Activity, AlertTriangle, CheckCircle, Database, ListTodo, XCircle } from "lucide-react"
import { DataCard, Page, PageHeader, StateMessage } from "../components/dashboard/layout"
import { Badge } from "../components/ui/badge"
import { Button, buttonVariants } from "../components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import {
  aggregateRequestFailureIsPermanent,
  aggregateSnapshotNeedsWarning,
  retryAggregateRequest,
} from "../lib/aggregate-query"
import {
  fetchOperationalSnapshot,
  fetchPendingActions,
  fetchSessionsSummary,
  fetchTasks,
  isApiPayloadTooLarge,
  type OperationalSnapshot,
  type PendingAction,
  type SessionsSummary,
  type Task,
} from "../lib/api"
import { formatCount, formatDateTime, sumCounts } from "../lib/format"
import { summarizeSessionDebugState } from "../lib/session-debug"

const OVERVIEW_SOURCE_LIMIT = 25
const OVERVIEW_VISIBLE_LIST_LIMIT = 8

function describeOverviewSessionScope(
  loadedCount: number,
  totalCount: number | null,
  hasMore: boolean,
) {
  const reliableTotal =
    totalCount !== null && Number.isSafeInteger(totalCount) && totalCount >= loadedCount
      ? totalCount
      : null
  if (hasMore) {
    if (reliableTotal !== null) {
      return `latest ${formatCount(loadedCount)} of ${formatCount(reliableTotal)} sessions by updated time (${OVERVIEW_SOURCE_LIMIT}-session limit)`
    }
    return `${formatCount(loadedCount)} most recently updated sessions loaded (${OVERVIEW_SOURCE_LIMIT}-session limit; more available; total unavailable)`
  }
  if (reliableTotal === loadedCount) {
    return `all ${formatCount(loadedCount)} sessions loaded (${OVERVIEW_SOURCE_LIMIT}-session limit)`
  }
  if (reliableTotal !== null) {
    return `${formatCount(loadedCount)} of ${formatCount(reliableTotal)} reported sessions loaded (${OVERVIEW_SOURCE_LIMIT}-session limit; no continuation reported)`
  }
  return `${formatCount(loadedCount)} sessions loaded (${OVERVIEW_SOURCE_LIMIT}-session limit; total unavailable)`
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

function needsAttentionTask(task: Task) {
  return ["blocked", "needs_attention", "failed"].includes(task.status)
}

type SessionSummaryItem = SessionsSummary["sessions"][number]

function sessionDebugSummary(item: SessionSummaryItem) {
  return summarizeSessionDebugState({
    status: item.session.status,
    latestEvent: item.events.latest_event,
    terminalEvent: item.outcome.terminal_event,
    countsByType: item.events.counts_by_type,
  })
}

function needsAttentionSessionItem(item: SessionSummaryItem) {
  return (
    ["failed", "interrupted"].includes(item.session.status) || sessionDebugSummary(item) !== null
  )
}

function latestEventLabel(value: unknown) {
  if (typeof value === "string") return value
  if (value && typeof value === "object" && "type" in value) {
    const type = (value as { type?: unknown }).type
    return typeof type === "string" ? type : "event"
  }
  return "no events"
}

function attentionSessionLabel(item: SessionSummaryItem) {
  const debug = sessionDebugSummary(item)
  if (debug) return debug.label
  if (item.session.status === "failed") return "Failed session"
  if (item.session.status === "interrupted") return "Interrupted session"
  return item.session.status
}

function pendingActionLabel(action: PendingAction) {
  if (action.kind === "tool_approval") return "Awaiting approval"
  if (action.kind === "user_input") return "Awaiting user input"
  return "Manual recovery required"
}

function snapshotAccuracyDetail(
  snapshot: OperationalSnapshot["sessions"] | NonNullable<OperationalSnapshot["tasks"]>,
): string {
  if (snapshot.accuracy.kind === "exact") return "Exact"
  const reason = snapshot.accuracy.reason ? `: ${snapshot.accuracy.reason}` : ""
  const limit = snapshot.accuracy.limit ? ` (limit ${formatCount(snapshot.accuracy.limit)})` : ""
  return `${snapshot.accuracy.kind}${reason}${limit}`
}

function taskSnapshotStatus(snapshot: OperationalSnapshot): string {
  if (snapshot.tasks) {
    return `${snapshotAccuracyDetail(snapshot.tasks)} task counts as of ${formatDateTime(snapshot.tasks.as_of)}`
  }
  if (snapshot.task_snapshot_status === "not_configured") {
    return "Task metrics unavailable: no task store is configured"
  }
  if (snapshot.task_snapshot_status === "unsupported") {
    return "Task metrics unavailable: the configured task store does not support snapshots"
  }
  return "Task metrics were not requested"
}

export function DashboardPage() {
  const operations = useQuery({
    queryKey: ["operational-snapshot", "dashboard"],
    queryFn: ({ signal }) => fetchOperationalSnapshot({}, signal),
    retry: retryAggregateRequest,
    refetchInterval: (query) =>
      aggregateRequestFailureIsPermanent(query.state.error) ? false : 5000,
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: (query) => !aggregateRequestFailureIsPermanent(query.state.error),
  })
  const summary = useQuery({
    queryKey: ["sessions-summary", "dashboard"],
    queryFn: () =>
      fetchSessionsSummary({ limit: OVERVIEW_SOURCE_LIMIT, order_by: "updated_at_desc" }),
    refetchInterval: 5000,
    refetchIntervalInBackground: false,
  })
  const tasks = useQuery({
    queryKey: ["tasks", "dashboard"],
    queryFn: () => fetchTasks({ limit: OVERVIEW_SOURCE_LIMIT }),
    refetchInterval: 5000,
    refetchIntervalInBackground: false,
  })
  const pendingActions = useQuery({
    queryKey: ["pending-actions", "dashboard"],
    queryFn: () => fetchPendingActions({ limit: OVERVIEW_SOURCE_LIMIT }),
    retry: (failureCount, error) => !isApiPayloadTooLarge(error) && failureCount < 3,
    refetchInterval: (query) => (isApiPayloadTooLarge(query.state.error) ? false : 5000),
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: (query) => !isApiPayloadTooLarge(query.state.error),
  })

  const sessionItems = summary.data?.sessions || []
  const list = sessionItems.map((item) => item.session)
  const taskList = tasks.data || []
  const pendingActionList = pendingActions.data?.actions || []
  const pendingActionIssues = pendingActions.data?.issues || []
  const sessionsError = summary.error instanceof Error ? summary.error.message : null
  const tasksError = tasks.error instanceof Error ? tasks.error.message : null
  const attentionTasks = taskList.filter(needsAttentionTask)
  const attentionSessionItems = sessionItems.filter(needsAttentionSessionItem)
  const attentionSessions = attentionSessionItems.map((item) => item.session)
  const sessionScope = summary.data
    ? describeOverviewSessionScope(
        list.length,
        summary.data.total_count,
        Boolean(summary.data.next_cursor),
      )
    : `up to ${OVERVIEW_SOURCE_LIMIT} most recently updated sessions`
  const sessionSampleLoading = summary.isLoading && summary.data === undefined
  const taskSampleLoading = tasks.isLoading && tasks.data === undefined
  const attentionSampleLoading =
    sessionSampleLoading ||
    taskSampleLoading ||
    (pendingActions.isLoading && pendingActions.data === undefined)
  const operationalSnapshot = operations.data
  const sessionCounts = operationalSnapshot?.sessions.counts_by_status
  const activeSessionCount = sessionCounts
    ? sumCounts(sessionCounts.pending, sessionCounts.running, sessionCounts.interrupting)
    : null
  const taskCounts = operationalSnapshot?.tasks?.counts_by_status
  const attentionTaskCount = taskCounts
    ? sumCounts(taskCounts.blocked, taskCounts.needs_attention, taskCounts.failed)
    : null
  const sessionMetricDetail = operationalSnapshot
    ? `${snapshotAccuracyDetail(operationalSnapshot.sessions)} as of ${formatDateTime(operationalSnapshot.sessions.as_of)}${operations.isError ? "; refresh failed" : ""}.`
    : operations.isError
      ? "Operational session snapshot unavailable."
      : "Loading the operational session snapshot."
  const taskMetricDetail = operationalSnapshot
    ? `${taskSnapshotStatus(operationalSnapshot)}${operations.isError ? "; refresh failed" : ""}.`
    : operations.isError
      ? "Operational task snapshot unavailable."
      : "Loading the operational task snapshot."
  const operationalSnapshotIsWarning = aggregateSnapshotNeedsWarning(
    operationalSnapshot,
    operations.isError,
  )

  return (
    <Page>
      <PageHeader
        title="Dashboard"
        actions={
          <Link to="/run" className={buttonVariants()}>
            New Run
          </Link>
        }
      />

      {operations.isError && !operationalSnapshot ? (
        <StateMessage tone="danger">
          <div data-testid="overview-operational-scope" role="alert">
            <div className="font-medium">Operational snapshot unavailable</div>
            <div className="mt-1">
              {operations.error instanceof Error
                ? operations.error.message
                : "Failed to load configured-store metrics."}{" "}
              Recent drill-down lists remain available below.
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="mt-3"
              disabled={operations.isFetching}
              onClick={() => void operations.refetch()}
            >
              {operations.isFetching ? "Retrying..." : "Retry metrics"}
            </Button>
          </div>
        </StateMessage>
      ) : operationalSnapshot ? (
        <div
          className={`flex items-start gap-3 rounded-md border px-4 py-3 text-sm ${
            operationalSnapshotIsWarning
              ? "border-chart-1/30 bg-chart-1/5 text-chart-1"
              : "border-border bg-muted/40"
          }`}
          data-testid="overview-operational-scope"
          role={operationalSnapshotIsWarning ? "alert" : "status"}
        >
          {operationalSnapshotIsWarning ? (
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          ) : (
            <Database className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
          )}
          <div className="space-y-1">
            <div className="font-medium">Configured-store operational snapshot</div>
            <div className="text-muted-foreground">
              {snapshotAccuracyDetail(operationalSnapshot.sessions)} session counts as of{" "}
              {formatDateTime(operationalSnapshot.sessions.as_of)}.{" "}
              {taskSnapshotStatus(operationalSnapshot)}. Scope is the server&apos;s configured
              stores; session and task snapshots are independent, not one cross-store atomic read.
              {operations.isError &&
                " The latest refresh failed; last confirmed values remain visible."}
            </div>
          </div>
        </div>
      ) : (
        <StateMessage>
          <div data-testid="overview-operational-scope">Loading configured-store metrics...</div>
        </StateMessage>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[
          {
            icon: Activity,
            label: "Active Sessions",
            value: activeSessionCount === null ? "—" : formatCount(activeSessionCount),
            detail: `${sessionMetricDetail} Pending, running, and interrupting sessions are active.`,
          },
          {
            icon: CheckCircle,
            label: "Completed Sessions",
            value: sessionCounts ? formatCount(sessionCounts.completed) : "—",
            detail: sessionMetricDetail,
          },
          {
            icon: XCircle,
            label: "Failed Sessions",
            value: sessionCounts ? formatCount(sessionCounts.failed) : "—",
            detail: sessionMetricDetail,
          },
          {
            icon: ListTodo,
            label: "Tasks Needing Attention",
            value: attentionTaskCount === null ? "—" : formatCount(attentionTaskCount),
            detail: `${taskMetricDetail} Blocked, needs attention, and failed tasks are included.`,
          },
        ].map(({ icon: Icon, label, value, detail }) => (
          <Card key={label}>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
              <Icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{value}</div>
              <div className="mt-1 text-sm text-muted-foreground">{detail}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div
        className="flex items-start gap-3 rounded-md border border-border bg-muted/40 px-4 py-3 text-sm"
        data-testid="overview-sample-scope"
      >
        <Database className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
        <div className="space-y-1">
          <div className="font-medium">Recent lists — bounded drill-down</div>
          <div className="text-muted-foreground">
            Recent sessions cover {sessionScope}. Attention and task lists independently load up to{" "}
            {OVERVIEW_SOURCE_LIMIT} tasks and {OVERVIEW_SOURCE_LIMIT} pending actions; categories
            can overlap and are not deployment totals.
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <DataCard
          title="Recent Sessions"
          description={`Showing up to ${OVERVIEW_VISIBLE_LIST_LIMIT} rows from ${sessionScope}.`}
          contentClassName="p-4"
        >
          {summary.isError ? (
            <StateMessage tone="danger" className="py-6">
              {sessionsError || "Failed to load sessions."}
            </StateMessage>
          ) : sessionSampleLoading ? (
            <StateMessage className="py-6">Loading recent sessions...</StateMessage>
          ) : list.length === 0 ? (
            <StateMessage className="py-6">
              No sessions yet.{" "}
              <Link to="/run" className="text-primary underline">
                Start a run
              </Link>
            </StateMessage>
          ) : (
            <div className="space-y-2">
              {sessionItems.slice(0, OVERVIEW_VISIBLE_LIST_LIMIT).map(({ session: s, events }) => (
                <Link
                  key={s.id}
                  to="/sessions/$sessionId"
                  params={{ sessionId: s.id }}
                  className="flex items-center justify-between gap-3 rounded-md p-3 text-foreground no-underline transition-colors hover:bg-muted"
                >
                  <div className="min-w-0">
                    <div className="truncate font-mono text-sm">{s.id}</div>
                    <div className="flex min-w-0 flex-wrap gap-x-2 gap-y-1 text-xs text-muted-foreground">
                      <span>{formatDateTime(s.updated_at || s.created_at)}</span>
                      <span className="truncate">{s.agent_name}</span>
                      <span className="truncate">{latestEventLabel(events.latest_event)}</span>
                    </div>
                  </div>
                  <div className="shrink-0">
                    <StatusBadge status={s.status} />
                  </div>
                </Link>
              ))}
            </div>
          )}
        </DataCard>

        <DataCard
          title="Recent Needs Attention"
          description={`Showing up to 20 entries from ${sessionScope}, ${OVERVIEW_SOURCE_LIMIT} recent tasks, and a pending-action page with a ${OVERVIEW_SOURCE_LIMIT}-action limit. Categories can overlap.`}
          contentClassName="p-4"
          actions={
            <Link
              to="/pending-actions"
              className={buttonVariants({ variant: "outline", size: "sm" })}
            >
              Pending
            </Link>
          }
        >
          {summary.isError || tasks.isError || pendingActions.isError ? (
            <StateMessage tone="danger" className="py-6">
              {sessionsError ||
                tasksError ||
                (pendingActions.error instanceof Error ? pendingActions.error.message : null) ||
                "Failed to load dashboard state."}
            </StateMessage>
          ) : attentionSampleLoading ? (
            <StateMessage className="py-6">Loading recent attention samples...</StateMessage>
          ) : attentionSessions.length === 0 &&
            attentionTasks.length === 0 &&
            pendingActionList.length === 0 &&
            pendingActionIssues.length === 0 ? (
            <StateMessage className="py-6">
              No sampled sessions, tasks, or pending actions need attention.
            </StateMessage>
          ) : (
            <div className="space-y-2">
              {pendingActionIssues.slice(0, 5).map((issue) => (
                <Link
                  key={`${issue.code}:${issue.session_id}`}
                  to="/sessions/$sessionId"
                  params={{ sessionId: issue.session_id }}
                  className="flex items-center justify-between gap-3 rounded-md p-3 text-foreground no-underline transition-colors hover:bg-muted"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 truncate text-sm font-medium">
                      <AlertTriangle className="h-4 w-4 shrink-0 text-destructive" />
                      <span className="truncate">Pending state needs inspection</span>
                    </div>
                    <div className="truncate text-xs text-muted-foreground">{issue.session_id}</div>
                  </div>
                  <Badge variant="outline" className="shrink-0">
                    {issue.agent_name}
                  </Badge>
                </Link>
              ))}
              {pendingActionList.slice(0, 5).map((action) => (
                <Link
                  key={action.id}
                  to="/sessions/$sessionId"
                  params={{ sessionId: action.session.id }}
                  className="flex items-center justify-between gap-3 rounded-md p-3 text-foreground no-underline transition-colors hover:bg-muted"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 truncate text-sm font-medium">
                      <AlertTriangle className="h-4 w-4 shrink-0 text-chart-1" />
                      <span className="truncate">{pendingActionLabel(action)}</span>
                    </div>
                    <div className="truncate text-xs text-muted-foreground">
                      {action.detail || action.question || action.tool_name || action.session.id}
                    </div>
                  </div>
                  <Badge variant="outline" className="shrink-0">
                    {action.session.agent_name}
                  </Badge>
                </Link>
              ))}
              {attentionSessionItems.slice(0, 4).map((item) => (
                <Link
                  key={item.session.id}
                  to="/sessions/$sessionId"
                  params={{ sessionId: item.session.id }}
                  className="flex items-center justify-between gap-3 rounded-md p-3 text-foreground no-underline transition-colors hover:bg-muted"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 truncate text-sm font-medium">
                      <AlertTriangle className="h-4 w-4 shrink-0 text-destructive" />
                      <span className="truncate">{item.session.id}</span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {attentionSessionLabel(item)} ·{" "}
                      {formatDateTime(item.session.updated_at || item.session.created_at)}
                    </div>
                  </div>
                  <StatusBadge status={item.session.status} />
                </Link>
              ))}
              {attentionTasks.slice(0, 6).map((t) => (
                <div
                  key={t.id}
                  className="flex items-center justify-between gap-3 rounded-md bg-muted/50 p-3"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 truncate text-sm font-medium">
                      <ListTodo className="h-4 w-4 shrink-0 text-muted-foreground" />
                      <span className="truncate">{t.title || t.type}</span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {t.status_reason || `Updated ${formatDateTime(t.updated_at || t.created_at)}`}
                    </div>
                  </div>
                  <StatusBadge status={t.status} />
                </div>
              ))}
            </div>
          )}
        </DataCard>

        <DataCard
          title="Recent Tasks"
          description={`Showing up to ${OVERVIEW_VISIBLE_LIST_LIMIT} of the ${OVERVIEW_SOURCE_LIMIT} most recently updated tasks; no deployment-wide task total is shown.`}
          contentClassName="p-4"
        >
          {tasks.isError ? (
            <StateMessage tone="danger" className="py-6">
              {tasksError || "Failed to load tasks."}
            </StateMessage>
          ) : taskSampleLoading ? (
            <StateMessage className="py-6">Loading recent tasks...</StateMessage>
          ) : taskList.length === 0 ? (
            <StateMessage className="py-6">No tasks yet.</StateMessage>
          ) : (
            <div className="space-y-2">
              {taskList.slice(0, OVERVIEW_VISIBLE_LIST_LIMIT).map((t) => (
                <div
                  key={t.id}
                  className="flex items-center justify-between gap-3 rounded-md bg-muted/50 p-3"
                >
                  <div className="min-w-0">
                    <div className="truncate text-sm">{t.title || t.type}</div>
                    <div className="text-xs text-muted-foreground">
                      Updated {formatDateTime(t.updated_at || t.created_at)}
                    </div>
                  </div>
                  <div className="shrink-0">
                    <StatusBadge status={t.status} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </DataCard>
      </div>
    </Page>
  )
}
