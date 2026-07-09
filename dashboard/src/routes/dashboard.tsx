import { useQuery } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import { Activity, AlertTriangle, CheckCircle, Database, ListTodo, XCircle } from "lucide-react"
import { DataCard, Page, PageHeader, StateMessage } from "../components/dashboard/layout"
import { Badge } from "../components/ui/badge"
import { buttonVariants } from "../components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card"
import {
  fetchPendingActions,
  fetchSessionsSummary,
  fetchTasks,
  type PendingAction,
  type Session,
  type SessionsSummary,
  type Task,
} from "../lib/api"
import { formatCount, formatDateTime, numericValue } from "../lib/format"
import { summarizeSessionDebugState } from "../lib/session-debug"

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

function isActiveSession(session: Session) {
  return ["pending", "running", "interrupting"].includes(session.status)
}

function isActiveTask(task: Task) {
  return ["pending", "claimed", "running", "paused", "blocked", "needs_attention"].includes(
    task.status,
  )
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

export function DashboardPage() {
  const summary = useQuery({
    queryKey: ["sessions-summary", "dashboard"],
    queryFn: () => fetchSessionsSummary({ limit: 25, order_by: "updated_at_desc" }),
    refetchInterval: 5000,
  })
  const tasks = useQuery({
    queryKey: ["tasks", "dashboard"],
    queryFn: () => fetchTasks({ limit: 25 }),
    refetchInterval: 5000,
  })
  const pendingActions = useQuery({
    queryKey: ["pending-actions", "dashboard"],
    queryFn: () => fetchPendingActions({ limit: 25 }),
    refetchInterval: 5000,
  })

  const sessionItems = summary.data?.sessions || []
  const list = sessionItems.map((item) => item.session)
  const taskList = tasks.data || []
  const pendingActionList = pendingActions.data?.actions || []
  const sessionsError = summary.error instanceof Error ? summary.error.message : null
  const tasksError = tasks.error instanceof Error ? tasks.error.message : null
  const activeSessions = list.filter(isActiveSession)
  const activeTasks = taskList.filter(isActiveTask)
  const completed = list.filter((s) => s.status === "completed").length
  const failed = list.filter((s) => s.status === "failed").length
  const attentionTasks = taskList.filter(needsAttentionTask)
  const attentionSessionItems = sessionItems.filter(needsAttentionSessionItem)
  const attentionSessions = attentionSessionItems.map((item) => item.session)
  const usage = summary.data?.usage.usage
  const totalTokens = numericValue(usage?.total_tokens)

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

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[
          {
            icon: Activity,
            label: "Active Work",
            value: activeSessions.length + activeTasks.length + pendingActionList.length,
            detail: `${activeSessions.length} sessions / ${activeTasks.length} tasks / ${pendingActionList.length} pending`,
          },
          {
            icon: CheckCircle,
            label: "Completed",
            value: completed,
            detail: "sessions in current view",
          },
          {
            icon: XCircle,
            label: "Failed",
            value: failed,
            detail: "sessions in current view",
          },
          {
            icon: Database,
            label: "Tokens",
            value: formatCount(totalTokens),
            detail: `${formatCount(usage?.input_tokens)} in / ${formatCount(usage?.output_tokens)} out`,
          },
        ].map(({ icon: Icon, label, value, detail }) => (
          <Card key={label}>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
              <Icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{value}</div>
              <div className="mt-1 truncate text-sm text-muted-foreground">{detail}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <DataCard title="Recent Sessions" contentClassName="p-4">
          {summary.isError ? (
            <StateMessage tone="danger" className="py-6">
              {sessionsError || "Failed to load sessions."}
            </StateMessage>
          ) : list.length === 0 ? (
            <StateMessage className="py-6">
              No sessions yet.{" "}
              <Link to="/run" className="text-primary underline">
                Start a run
              </Link>
            </StateMessage>
          ) : (
            <div className="space-y-2">
              {sessionItems.slice(0, 8).map(({ session: s, events }) => (
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
          title="Needs Attention"
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
          ) : attentionSessions.length === 0 &&
            attentionTasks.length === 0 &&
            pendingActionList.length === 0 ? (
            <StateMessage className="py-6">
              No sessions, tasks, or pending actions need attention in the current view.
            </StateMessage>
          ) : (
            <div className="space-y-2">
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
                      {t.status_reason || formatDateTime(t.created_at)}
                    </div>
                  </div>
                  <StatusBadge status={t.status} />
                </div>
              ))}
            </div>
          )}
        </DataCard>

        <DataCard title="Recent Tasks" contentClassName="p-4">
          {tasks.isError ? (
            <StateMessage tone="danger" className="py-6">
              {tasksError || "Failed to load tasks."}
            </StateMessage>
          ) : taskList.length === 0 ? (
            <StateMessage className="py-6">No tasks yet.</StateMessage>
          ) : (
            <div className="space-y-2">
              {taskList.slice(0, 8).map((t) => (
                <div
                  key={t.id}
                  className="flex items-center justify-between gap-3 rounded-md bg-muted/50 p-3"
                >
                  <div className="min-w-0">
                    <div className="truncate text-sm">{t.title || t.type}</div>
                    <div className="text-xs text-muted-foreground">
                      {formatDateTime(t.created_at)}
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
