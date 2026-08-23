import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import { AlertTriangle, Ban, CirclePause, Play, Search, TriangleAlert } from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"
import {
  DataCard,
  Page,
  PageHeader,
  PayloadViewer,
  StateMessage,
} from "../components/dashboard/layout"
import { useDashboardCapability } from "../components/dashboard/server-contract"
import { Badge } from "../components/ui/badge"
import { Button } from "../components/ui/button"
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
  blockTask,
  fetchForkGroup,
  fetchTask,
  fetchTasks,
  markTaskNeedsAttention,
  pauseTask,
  resumeTask,
  type Task,
  type TaskDetail,
  type TaskHold,
  type TaskListQuery,
} from "../lib/api"
import { dashboardCapabilityUnavailableText } from "../lib/dashboard-capabilities"
import { formatDateTime } from "../lib/format"
import { currentQueryParam, dashboardPath, replaceDashboardLocation } from "../lib/links"
import { taskAvailabilityDescriptor } from "../lib/task-availability"
import { cn } from "../lib/utils"

type TaskStatusFilter = "all" | NonNullable<TaskListQuery["status"]>
type TaskOrder = NonNullable<TaskListQuery["order_by"]>
type TaskAction = "pause" | "block" | "needs_attention" | "resume"
type TaskActionVariables = {
  action: TaskAction
  task: Task
  body: TaskHold
  listQuery: TaskListQuery
}

const PAGE_LIMIT = 100
const TASK_STATUSES: TaskStatusFilter[] = [
  "pending",
  "claimed",
  "running",
  "paused",
  "blocked",
  "needs_attention",
  "completed",
  "failed",
  "cancelled",
]
const HELD_STATUSES = new Set(["paused", "blocked", "needs_attention"])
const TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"])
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

function optionalFilter(value: string) {
  const trimmed = value.trim()
  return trimmed === "" ? undefined : trimmed
}

function statusBadgeVariant(status: string) {
  if (status === "completed") return "default"
  if (status === "running" || status === "claimed") return "secondary"
  if (status === "failed") return "destructive"
  return "outline"
}

function canHold(task: Task) {
  if (TERMINAL_STATUSES.has(task.status)) return false
  return !(task.status === "running" && task.session_id)
}

function canResume(task: Task) {
  return HELD_STATUSES.has(task.status)
}

function taskTitle(task: Task) {
  return task.title || task.type
}

type ForkGroupTaskLink = {
  sourceSessionId: string
  groupId: string
}

function forkGroupTaskLink(task: TaskDetail | undefined): ForkGroupTaskLink | null {
  if (!task?.fork_group) return null
  return {
    sourceSessionId: task.fork_group.source_session_id,
    groupId: task.fork_group.group_id,
  }
}

function actionLabel(action: TaskAction) {
  if (action === "pause") return "Pause"
  if (action === "block") return "Block"
  if (action === "needs_attention") return "Needs attention"
  return "Resume"
}

function taskActionBody(reason: string): TaskHold {
  const trimmed = reason.trim()
  return trimmed ? { reason: trimmed } : {}
}

function taskMatchesSearch(task: TaskDetail, query: string) {
  const needle = query.toLowerCase()
  return [
    task.id,
    task.type,
    task.title,
    task.description,
    task.status,
    task.session_id,
    task.parent_task_id,
    task.assigned_agent_name,
    task.worker_id,
    task.status_reason,
  ].some((value) => value?.toLowerCase().includes(needle))
}

function taskMatchesQuery(task: TaskDetail, query: TaskListQuery) {
  return (
    (query.q == null || taskMatchesSearch(task, query.q)) &&
    (query.status == null || task.status === query.status) &&
    (query.type == null || task.type === query.type) &&
    (query.session_id == null || task.session_id === query.session_id) &&
    (query.parent_task_id == null || task.parent_task_id === query.parent_task_id) &&
    (query.assigned_agent_name == null || task.assigned_agent_name === query.assigned_agent_name)
  )
}

function replaceSelectedTaskInLocation(taskId: string) {
  const nextUrl = new URL(window.location.href)
  nextUrl.searchParams.set("task_id", taskId)
  window.history.replaceState(null, "", `${nextUrl.pathname}${nextUrl.search}${nextUrl.hash}`)
}

export function TasksPage() {
  const queryClient = useQueryClient()
  const taskLifecycleCapability = useDashboardCapability({
    kind: "mutation",
    mutation: "task_lifecycle",
  })
  const taskLifecycleUnavailableText = dashboardCapabilityUnavailableText(taskLifecycleCapability)
  const [search, setSearch] = useState(() => currentQueryParam("q"))
  const [assignedAgentFilter, setAssignedAgentFilter] = useState(() =>
    currentQueryParam("assigned_agent_name"),
  )
  const [sessionFilter, setSessionFilter] = useState(() => currentQueryParam("session_id"))
  const [status, setStatus] = useState<TaskStatusFilter>("all")
  const [orderBy, setOrderBy] = useState<TaskOrder>("updated_at_desc")
  const [offset, setOffset] = useState(0)
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(
    () => currentQueryParam("task_id") || null,
  )
  const pinnedTaskIdRef = useRef(selectedTaskId)
  const [reason, setReason] = useState("")
  const debouncedSearch = useDebouncedValue(search, 300)

  const query = useMemo<TaskListQuery>(
    () => ({
      q: optionalFilter(debouncedSearch),
      assigned_agent_name: optionalFilter(assignedAgentFilter),
      session_id: optionalFilter(sessionFilter),
      status: status === "all" ? undefined : status,
      order_by: orderBy,
      limit: PAGE_LIMIT,
      offset,
    }),
    [assignedAgentFilter, debouncedSearch, offset, orderBy, sessionFilter, status],
  )

  const tasks = useQuery({
    queryKey: ["tasks", "queue", query],
    queryFn: () => fetchTasks(query),
    refetchInterval: 5000,
  })
  const taskList = tasks.data ?? []
  const listFallbackTask = taskList[0] ?? null
  const selectedListTask = selectedTaskId
    ? (taskList.find((task) => task.id === selectedTaskId) ?? null)
    : listFallbackTask
  const taskDetailId = selectedTaskId ?? listFallbackTask?.id ?? null
  const taskDetail = useQuery({
    queryKey: ["task", taskDetailId],
    queryFn: () => fetchTask(taskDetailId ?? ""),
    enabled: taskDetailId != null,
    refetchInterval: 5000,
  })
  const forkGroupLink = useMemo(() => forkGroupTaskLink(taskDetail.data), [taskDetail.data])
  const forkGroup = useQuery({
    queryKey: ["fork-group", forkGroupLink?.sourceSessionId, forkGroupLink?.groupId],
    queryFn: () =>
      fetchForkGroup(forkGroupLink?.sourceSessionId ?? "", forkGroupLink?.groupId ?? ""),
    enabled: forkGroupLink != null,
    refetchInterval: 5000,
  })
  const selectedTask =
    taskDetail.data ?? (selectedListTask?.id === taskDetailId ? selectedListTask : null)
  const selectedAvailabilityLabel = selectedTask ? taskAvailabilityDescriptor(selectedTask) : null
  const hasNextPage = taskList.length === PAGE_LIMIT
  const hasPreviousPage = offset > 0
  const hasFilters =
    status !== "all" ||
    search.trim() !== "" ||
    assignedAgentFilter.trim() !== "" ||
    sessionFilter.trim() !== ""

  useEffect(() => {
    if (!selectedTaskId && taskList.length > 0) {
      const firstTask = taskList[0]
      if (firstTask) {
        setReason("")
        setSelectedTaskId(firstTask.id)
      }
      return
    }
    if (
      selectedTaskId &&
      taskList.length > 0 &&
      !taskList.some((task) => task.id === selectedTaskId) &&
      pinnedTaskIdRef.current !== selectedTaskId
    ) {
      const firstTask = taskList[0]
      if (firstTask) {
        setReason("")
        setSelectedTaskId(firstTask.id)
        replaceSelectedTaskInLocation(firstTask.id)
      }
    }
  }, [selectedTaskId, taskList])

  const action = useMutation({
    mutationFn: async ({ action, body, task }: TaskActionVariables) => {
      if (action === "pause") return pauseTask(task.id, body)
      if (action === "block") return blockTask(task.id, body)
      if (action === "needs_attention") return markTaskNeedsAttention(task.id, body)
      return resumeTask(task.id)
    },
    onSuccess: async (task, variables) => {
      setSelectedTaskId(task.id)
      setReason("")
      queryClient.setQueryData<Task[]>(["tasks", "queue", variables.listQuery], (current) => {
        if (!current) return current
        if (!taskMatchesQuery(task, variables.listQuery)) {
          return current.filter((item) => item.id !== task.id)
        }
        return current.map((item) => (item.id === task.id ? { ...item, ...task } : item))
      })
      queryClient.setQueryData<TaskDetail>(["task", task.id], task)
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["tasks"] }),
        queryClient.invalidateQueries({ queryKey: ["sessions-summary", "dashboard"] }),
      ])
    },
  })

  function clearFilters() {
    pinnedTaskIdRef.current = null
    setSelectedTaskId(null)
    setSearch("")
    setAssignedAgentFilter("")
    setSessionFilter("")
    setStatus("all")
    setOrderBy("updated_at_desc")
    setOffset(0)
    replaceDashboardLocation("/tasks")
  }

  function runAction(nextAction: TaskAction, task: Task) {
    if (!taskLifecycleCapability.enabled) return
    action.mutate({ action: nextAction, body: taskActionBody(reason), task, listQuery: query })
  }

  function selectTask(taskId: string) {
    pinnedTaskIdRef.current = null
    if (taskId !== selectedTaskId) {
      setReason("")
      setSelectedTaskId(taskId)
    }
    replaceSelectedTaskInLocation(taskId)
  }

  return (
    <Page>
      <PageHeader
        title="Tasks"
        description="Operate durable task queues, held work, and worker leases."
      />

      {taskLifecycleUnavailableText && (
        <StateMessage data-testid="task-mutations-unavailable">
          Task lifecycle actions are unavailable. {taskLifecycleUnavailableText} Task inspection
          remains available.
        </StateMessage>
      )}

      <DataCard>
        <div className="flex min-w-0 flex-wrap items-center gap-2 border-b border-border p-4">
          <div className="relative min-w-64 flex-[2_1_24rem]">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(event) => {
                setSearch(event.target.value)
                setOffset(0)
              }}
              placeholder="Search tasks, types, agents, sessions, workers..."
              className="pl-8"
            />
          </div>
          <select
            value={status}
            onChange={(event) => {
              setStatus(event.target.value as TaskStatusFilter)
              setOffset(0)
            }}
            className={selectClassName}
            aria-label="Filter by task status"
          >
            <option value="all">All statuses</option>
            {TASK_STATUSES.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
          <select
            value={orderBy}
            onChange={(event) => {
              setOrderBy(event.target.value as TaskOrder)
              setOffset(0)
            }}
            className={selectClassName}
            aria-label="Sort tasks"
          >
            <option value="updated_at_desc">Updated newest</option>
            <option value="updated_at_asc">Updated oldest</option>
            <option value="created_at_desc">Created newest</option>
            <option value="created_at_asc">Created oldest</option>
          </select>
          {hasFilters && (
            <Button variant="outline" size="sm" onClick={clearFilters}>
              Clear
            </Button>
          )}
          {(assignedAgentFilter || sessionFilter) && (
            <div className="flex min-w-full flex-wrap gap-2 pt-1 text-xs">
              {assignedAgentFilter && <Badge variant="outline">agent: {assignedAgentFilter}</Badge>}
              {sessionFilter && <Badge variant="outline">session: {sessionFilter}</Badge>}
            </div>
          )}
        </div>
      </DataCard>

      <div className="grid min-h-0 grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.4fr)_minmax(22rem,0.8fr)]">
        <DataCard
          title="Task Queue"
          description={`Showing ${taskList.length} tasks${offset ? ` from offset ${offset}` : ""}.`}
          actions={
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={!hasPreviousPage || tasks.isFetching}
                onClick={() => setOffset(Math.max(0, offset - PAGE_LIMIT))}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={!hasNextPage || tasks.isFetching}
                onClick={() => setOffset(offset + PAGE_LIMIT)}
              >
                Next
              </Button>
            </div>
          }
        >
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Task</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Agent</TableHead>
                <TableHead>Worker</TableHead>
                <TableHead>Lease</TableHead>
                <TableHead>Availability</TableHead>
                <TableHead>Session</TableHead>
                <TableHead>Updated</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {tasks.isError ? (
                <TableRow>
                  <TableCell colSpan={8}>
                    <StateMessage tone="danger">
                      {tasks.error instanceof Error ? tasks.error.message : "Failed to load tasks."}
                    </StateMessage>
                  </TableCell>
                </TableRow>
              ) : tasks.isLoading ? (
                <TableRow>
                  <TableCell colSpan={8}>
                    <StateMessage>Loading tasks...</StateMessage>
                  </TableCell>
                </TableRow>
              ) : taskList.length ? (
                taskList.map((task) => {
                  const availabilityLabel = taskAvailabilityDescriptor(task)
                  return (
                    <TableRow
                      key={task.id}
                      className={cn(
                        "cursor-pointer",
                        selectedTask?.id === task.id && "bg-muted/70 hover:bg-muted/70",
                      )}
                      onClick={() => selectTask(task.id)}
                    >
                      <TableCell>
                        <div className="min-w-56">
                          <div className="truncate font-medium">{taskTitle(task)}</div>
                          <div className="truncate font-mono text-xs text-muted-foreground">
                            {task.id}
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant={statusBadgeVariant(task.status)}>{task.status}</Badge>
                      </TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {task.assigned_agent_name ?? "-"}
                      </TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {task.worker_id ?? "-"}
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {formatDateTime(task.lease_expires_at)}
                      </TableCell>
                      <TableCell>
                        {availabilityLabel ? (
                          <div className="space-y-1">
                            <Badge variant="outline">{availabilityLabel}</Badge>
                            {task.available_at && (
                              <div className="text-xs text-muted-foreground">
                                {formatDateTime(task.available_at)}
                              </div>
                            )}
                          </div>
                        ) : (
                          <span className="text-xs text-muted-foreground">
                            {task.available_at ? formatDateTime(task.available_at) : "-"}
                          </span>
                        )}
                      </TableCell>
                      <TableCell>
                        {task.session_id ? (
                          <Link
                            to="/sessions/$sessionId"
                            params={{ sessionId: task.session_id }}
                            className="font-mono text-xs text-primary hover:underline"
                            onClick={(event) => event.stopPropagation()}
                          >
                            {task.session_id}
                          </Link>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {formatDateTime(task.updated_at)}
                      </TableCell>
                    </TableRow>
                  )
                })
              ) : (
                <TableRow>
                  <TableCell colSpan={8}>
                    <StateMessage>
                      {hasFilters ? "No tasks match the current filters." : "No tasks yet."}
                    </StateMessage>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </DataCard>

        <DataCard
          title="Task Details"
          description={
            selectedTask?.id ?? taskDetailId ?? "Select a task to inspect and operate it."
          }
        >
          {selectedTask ? (
            <div className="space-y-4 p-4">
              <div className="grid gap-3 text-sm">
                <div className="flex items-start justify-between gap-3">
                  <span className="text-muted-foreground">Status</span>
                  <Badge variant={statusBadgeVariant(selectedTask.status)}>
                    {selectedTask.status}
                  </Badge>
                </div>
                <div className="flex items-start justify-between gap-3">
                  <span className="text-muted-foreground">Type</span>
                  <span className="max-w-56 truncate font-mono text-xs">{selectedTask.type}</span>
                </div>
                <div className="flex items-start justify-between gap-3">
                  <span className="text-muted-foreground">Agent</span>
                  {selectedTask.assigned_agent_name ? (
                    <a
                      href={dashboardPath("/agents", { q: selectedTask.assigned_agent_name })}
                      className="max-w-56 truncate font-mono text-xs text-primary hover:underline"
                    >
                      {selectedTask.assigned_agent_name}
                    </a>
                  ) : (
                    <span className="max-w-56 truncate font-mono text-xs">-</span>
                  )}
                </div>
                {selectedTask.parent_task_id && (
                  <div className="flex items-start justify-between gap-3">
                    <span className="text-muted-foreground">Parent</span>
                    <span className="max-w-56 truncate font-mono text-xs">
                      {selectedTask.parent_task_id}
                    </span>
                  </div>
                )}
                <div className="flex items-start justify-between gap-3">
                  <span className="text-muted-foreground">Worker</span>
                  <span className="max-w-56 truncate font-mono text-xs">
                    {selectedTask.worker_id ?? "-"}
                  </span>
                </div>
                <div className="flex items-start justify-between gap-3">
                  <span className="text-muted-foreground">Lease expires</span>
                  <span className="text-right text-xs">
                    {formatDateTime(selectedTask.lease_expires_at)}
                  </span>
                </div>
                <div className="flex items-start justify-between gap-3">
                  <span className="text-muted-foreground">Availability</span>
                  <div className="text-right">
                    {selectedAvailabilityLabel && (
                      <Badge variant="outline">{selectedAvailabilityLabel}</Badge>
                    )}
                    <div className="mt-1 text-xs">
                      {selectedTask.available_at
                        ? formatDateTime(selectedTask.available_at)
                        : "Immediate"}
                    </div>
                  </div>
                </div>
                {selectedTask.retry_series && (
                  <div className="rounded-md border border-border bg-muted/40 p-3">
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <span className="text-xs font-medium text-muted-foreground">
                        Retry series
                      </span>
                      <Badge variant="outline">{selectedTask.retry_series.disposition}</Badge>
                    </div>
                    <div className="grid gap-1.5 text-xs">
                      <div className="flex justify-between gap-3">
                        <span className="text-muted-foreground">Attempt</span>
                        <span>
                          {selectedTask.retry_series.attempt} /{" "}
                          {selectedTask.retry_series.policy.max_attempts}
                        </span>
                      </div>
                      <div className="flex justify-between gap-3">
                        <span className="text-muted-foreground">Attempts remaining</span>
                        <span>{selectedTask.retry_series.attempts_remaining}</span>
                      </div>
                      <div className="flex justify-between gap-3">
                        <span className="text-muted-foreground">Causal budget</span>
                        <span
                          className="max-w-48 truncate font-mono"
                          title={selectedTask.retry_series.causal_budget_id}
                        >
                          {selectedTask.retry_series.causal_budget_id}
                        </span>
                      </div>
                      {selectedTask.retry_series.tokens_remaining != null && (
                        <div className="flex justify-between gap-3">
                          <span className="text-muted-foreground">Tokens remaining</span>
                          <span>{selectedTask.retry_series.tokens_remaining}</span>
                        </div>
                      )}
                      {selectedTask.retry_series.estimated_cost_remaining != null && (
                        <div className="flex justify-between gap-3">
                          <span className="text-muted-foreground">Cost remaining</span>
                          <span>
                            {selectedTask.retry_series.estimated_cost_remaining}{" "}
                            {selectedTask.retry_series.policy.cost_currency}
                          </span>
                        </div>
                      )}
                      {selectedTask.retry_series.next_eligible_at && (
                        <div className="flex justify-between gap-3">
                          <span className="text-muted-foreground">Next eligible</span>
                          <span>{formatDateTime(selectedTask.retry_series.next_eligible_at)}</span>
                        </div>
                      )}
                      {selectedTask.retry_series.elapsed_deadline && (
                        <div className="flex justify-between gap-3">
                          <span className="text-muted-foreground">Series deadline</span>
                          <span>{formatDateTime(selectedTask.retry_series.elapsed_deadline)}</span>
                        </div>
                      )}
                    </div>
                    <div className="mt-2 truncate font-mono text-[11px] text-muted-foreground">
                      {selectedTask.retry_series.series_id}
                    </div>
                  </div>
                )}
                <div className="flex items-start justify-between gap-3">
                  <span className="text-muted-foreground">Created</span>
                  <span className="text-right text-xs">
                    {formatDateTime(selectedTask.created_at)}
                  </span>
                </div>
                <div className="flex items-start justify-between gap-3">
                  <span className="text-muted-foreground">Updated</span>
                  <span className="text-right text-xs">
                    {formatDateTime(selectedTask.updated_at)}
                  </span>
                </div>
                {taskDetail.data?.started_at && (
                  <div className="flex items-start justify-between gap-3">
                    <span className="text-muted-foreground">Started</span>
                    <span className="text-right text-xs">
                      {formatDateTime(taskDetail.data.started_at)}
                    </span>
                  </div>
                )}
                {selectedTask.status_reason && (
                  <div className="rounded-md border border-border bg-muted/40 p-3">
                    <div className="mb-1 text-xs font-medium text-muted-foreground">Reason</div>
                    <div className="text-sm">{selectedTask.status_reason}</div>
                  </div>
                )}
                {selectedTask.description && (
                  <div className="rounded-md border border-border bg-muted/40 p-3">
                    <div className="mb-1 text-xs font-medium text-muted-foreground">
                      Description
                    </div>
                    <div className="text-sm">{selectedTask.description}</div>
                  </div>
                )}
                {selectedTask.status_payload && (
                  <div>
                    <div className="mb-1 text-xs font-medium text-muted-foreground">Payload</div>
                    <PayloadViewer value={selectedTask.status_payload} />
                  </div>
                )}
                {taskDetail.isLoading && (
                  <StateMessage className="py-3">Loading full task payload...</StateMessage>
                )}
                {taskDetail.isError && (
                  <StateMessage tone="danger" className="py-3">
                    {taskDetail.error instanceof Error
                      ? taskDetail.error.message
                      : "Failed to load full task payload."}
                  </StateMessage>
                )}
                {taskDetail.data && (
                  <div className="grid gap-3">
                    {forkGroupLink && (
                      <div className="rounded-md border border-border bg-muted/40 p-3">
                        <div className="mb-2 flex items-center justify-between gap-3">
                          <span className="text-xs font-medium text-muted-foreground">
                            Fork group
                          </span>
                          {forkGroup.data && (
                            <Badge variant={statusBadgeVariant(forkGroup.data.state)}>
                              {forkGroup.data.state}
                            </Badge>
                          )}
                        </div>
                        {forkGroup.isLoading ? (
                          <div className="text-xs text-muted-foreground">
                            Loading durable group state...
                          </div>
                        ) : forkGroup.isError ? (
                          <StateMessage tone="danger" className="py-2">
                            {forkGroup.error instanceof Error
                              ? forkGroup.error.message
                              : "Failed to load durable group state."}
                          </StateMessage>
                        ) : forkGroup.data ? (
                          <div className="grid gap-1.5 text-xs">
                            <div className="flex justify-between gap-3">
                              <span className="text-muted-foreground">Group</span>
                              <span className="max-w-48 truncate font-mono">
                                {forkGroup.data.group_id}
                              </span>
                            </div>
                            <div className="flex justify-between gap-3">
                              <span className="text-muted-foreground">Recovery</span>
                              <Badge variant="outline">{forkGroup.data.recovery_status}</Badge>
                            </div>
                            <div className="flex justify-between gap-3">
                              <span className="text-muted-foreground">Execution</span>
                              <span>{forkGroup.data.execution_mode}</span>
                            </div>
                            <div className="flex justify-between gap-3">
                              <span className="text-muted-foreground">Attempts</span>
                              <span>{forkGroup.data.attempts.length}</span>
                            </div>
                            <div className="flex justify-between gap-3">
                              <span className="text-muted-foreground">Source</span>
                              <Link
                                to="/sessions/$sessionId"
                                params={{ sessionId: forkGroup.data.source_session_id }}
                                className="max-w-48 truncate font-mono text-primary hover:underline"
                              >
                                {forkGroup.data.source_session_id}
                              </Link>
                            </div>
                            {forkGroup.data.failure_code && (
                              <div className="flex justify-between gap-3 text-destructive">
                                <span>Failure</span>
                                <span>{forkGroup.data.failure_code}</span>
                              </div>
                            )}
                          </div>
                        ) : null}
                      </div>
                    )}
                    <div>
                      <div className="mb-1 text-xs font-medium text-muted-foreground">Input</div>
                      <PayloadViewer value={taskDetail.data.input} />
                    </div>
                    {taskDetail.data.result != null && (
                      <div>
                        <div className="mb-1 text-xs font-medium text-muted-foreground">Result</div>
                        <PayloadViewer value={taskDetail.data.result} />
                      </div>
                    )}
                    {taskDetail.data.error != null && (
                      <div>
                        <div className="mb-1 text-xs font-medium text-muted-foreground">Error</div>
                        <PayloadViewer value={taskDetail.data.error} />
                      </div>
                    )}
                    <div>
                      <div className="mb-1 text-xs font-medium text-muted-foreground">Metadata</div>
                      <PayloadViewer value={taskDetail.data.metadata} />
                    </div>
                  </div>
                )}
              </div>

              {action.isError && action.variables?.task.id === selectedTask.id && (
                <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                  {action.error instanceof Error ? action.error.message : "Task action failed."}
                </div>
              )}

              <div className="space-y-2">
                <Input
                  value={reason}
                  onChange={(event) => setReason(event.target.value)}
                  placeholder="Optional status reason"
                  disabled={action.isPending || !taskLifecycleCapability.enabled}
                  title={taskLifecycleUnavailableText ?? undefined}
                />
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    variant="outline"
                    disabled={
                      !taskLifecycleCapability.enabled || !canHold(selectedTask) || action.isPending
                    }
                    title={taskLifecycleUnavailableText ?? undefined}
                    onClick={() => runAction("pause", selectedTask)}
                  >
                    <CirclePause className="mr-1.5 h-4 w-4" />
                    {actionLabel("pause")}
                  </Button>
                  <Button
                    variant="outline"
                    disabled={
                      !taskLifecycleCapability.enabled || !canHold(selectedTask) || action.isPending
                    }
                    title={taskLifecycleUnavailableText ?? undefined}
                    onClick={() => runAction("block", selectedTask)}
                  >
                    <Ban className="mr-1.5 h-4 w-4" />
                    {actionLabel("block")}
                  </Button>
                  <Button
                    variant="outline"
                    disabled={
                      !taskLifecycleCapability.enabled || !canHold(selectedTask) || action.isPending
                    }
                    title={taskLifecycleUnavailableText ?? undefined}
                    onClick={() => runAction("needs_attention", selectedTask)}
                  >
                    <TriangleAlert className="mr-1.5 h-4 w-4" />
                    {actionLabel("needs_attention")}
                  </Button>
                  <Button
                    disabled={
                      !taskLifecycleCapability.enabled ||
                      !canResume(selectedTask) ||
                      action.isPending
                    }
                    title={taskLifecycleUnavailableText ?? undefined}
                    onClick={() => runAction("resume", selectedTask)}
                  >
                    <Play className="mr-1.5 h-4 w-4" />
                    {actionLabel("resume")}
                  </Button>
                </div>
                {selectedTask.status === "running" && selectedTask.session_id && (
                  <div className="flex items-start gap-2 rounded-md border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
                    <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                    Attached running tasks are controlled by their session and cannot be held here.
                  </div>
                )}
              </div>
            </div>
          ) : taskDetail.isLoading ? (
            <StateMessage>Loading task details...</StateMessage>
          ) : taskDetail.isError ? (
            <StateMessage tone="danger">
              {taskDetail.error instanceof Error
                ? taskDetail.error.message
                : "Failed to load task details."}
            </StateMessage>
          ) : (
            <StateMessage>No task selected.</StateMessage>
          )}
        </DataCard>
      </div>
    </Page>
  )
}
