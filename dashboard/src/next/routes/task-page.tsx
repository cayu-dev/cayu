/**
 * One task: what it's doing or waiting for, the session that ran it (every attempt's, when
 * it retries), the tasks it waits on or started, its schedule, input and result. Queue
 * internals (worker, lease, concurrency key, invocation) sit in Debug.
 */
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useParams } from "@tanstack/react-router"
import { ChevronRight, Copy, Ellipsis, Flag, Hand, Pause, Play } from "lucide-react"
import { type ReactNode, useState } from "react"
import { useViewer } from "@/components/dashboard/viewer"
import { Button, buttonVariants } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { TaskHold, TaskInfo, TaskItem, TaskScheduleEvent } from "../api/types"
import { useTask, useTaskScheduleEvents } from "../app/queries"
import { useApi, useNow, useServices } from "../app/services"
import {
  groupOf,
  HELD_STATUSES,
  holdable,
  isCayuTask,
  statusLine,
  taskName,
  when,
} from "../app/task-model"
import { useTaskAttempts, useTaskLinks, useTaskRelationPage } from "../app/task-queries"
import { ErrorState, NavLink, PageBody, StateBadge, sessionPath, TopBar, taskPath } from "./common"
import { ValueView } from "./session/value-view"
import { TaskActions } from "./task-actions"
import { GROUP_DOT, GROUP_TEXT } from "./tasks-page"

const HOLDS: { hold: TaskHold; label: string; verb: string; icon: typeof Pause }[] = [
  { hold: "pause", label: "Pause", verb: "Pause task", icon: Pause },
  { hold: "block", label: "Block", verb: "Block task", icon: Hand },
  { hold: "needs-attention", label: "Flag for attention", verb: "Flag task", icon: Flag },
]

const SCHEDULE_EVENT: Record<TaskScheduleEvent["type"], string> = {
  "task.scheduled": "Scheduled",
  "task.rescheduled": "Rescheduled",
  "task.schedule_eligible": "Became due",
  "task.schedule_misfired": "Missed its time",
  "task.schedule_expired": "Expired before it ran",
  "task.schedule_skipped": "Skipped",
  "task.schedule_dependency_skipped": "Skipped: a task it waits on didn't complete",
  "task.schedule_claimed": "Picked up by a worker",
  "task.schedule_cancellation_requested": "Cancellation requested",
  "task.schedule_cancelled": "Cancelled",
  "task.schedule_started": "Started",
  "task.schedule_held": "Held",
  "task.schedule_resumed": "Resumed",
  "task.schedule_completed": "Completed",
  "task.schedule_failed": "Failed",
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mt-8">
      <h2 className="mb-2 text-sm font-medium">{title}</h2>
      {children}
    </section>
  )
}

function Row({ k, children, mono }: { k: string; children: ReactNode; mono?: boolean }) {
  return (
    <div className="grid grid-cols-[8.5rem_1fr] gap-3 py-1.5 text-sm">
      <dt className="text-muted-foreground">{k}</dt>
      <dd className={cn("min-w-0 break-words", mono && "font-mono text-xs leading-5")}>
        {children}
      </dd>
    </div>
  )
}

/** Another task, as a row that opens it. */
function TaskLink({
  task,
  all,
  now,
  note,
}: {
  task: TaskItem
  all: TaskItem[]
  now: number
  note?: string
}) {
  const group = groupOf(task.status)
  return (
    <NavLink
      to={taskPath(task.id)}
      className="flex min-w-0 items-center gap-2 border-b border-border px-3 py-2 text-sm last:border-0 hover:bg-muted/60"
    >
      <span className={cn("size-2 shrink-0 rounded-full", GROUP_DOT[group])} aria-hidden="true" />
      {note && <span className="shrink-0 text-muted-foreground">{note}</span>}
      <span className="min-w-0 flex-1 truncate">{taskName(task)}</span>
      <span className={cn("shrink-0 truncate text-xs", GROUP_TEXT[group])}>
        {statusLine(task, all, now, undefined, false)}
      </span>
    </NavLink>
  )
}

const List = ({ children }: { children: ReactNode }) => (
  <div className="overflow-hidden rounded-xl border border-border">{children}</div>
)

const seconds = (s: number) => (s >= 3600 ? `${s / 3600}h` : s >= 60 ? `${s / 60}m` : `${s}s`)

export function TaskPage() {
  const { taskId } = useParams({ strict: false }) as { taskId: string }
  return <TaskPageView key={taskId} taskId={taskId} />
}

function TaskPageView({ taskId }: { taskId: string }) {
  const api = useApi()
  const now = useNow()
  const queryClient = useQueryClient()
  const detail = useTask(taskId)
  const services = useServices()
  const viewer = useViewer()
  const writable =
    !!services.prototype || (viewer?.full_access === true && services.taskMutationsEnabled === true)
  const [relationLimit, setRelationLimit] = useState(50)
  const linked = useTaskLinks(
    [
      ...(detail.data?.prerequisite_task_ids ?? []).slice(0, relationLimit),
      ...(detail.data?.parent_task_id ? [detail.data.parent_task_id] : []),
    ],
    true,
    true,
  )
  const relatedSupported = !!services.prototype || services.taskIndexEnabled === true
  const dependentsRead = useTaskRelationPage(
    taskId,
    "dependents",
    !!detail.data && relatedSupported,
  )
  const childrenRead = useTaskRelationPage(taskId, "children", !!detail.data && relatedSupported)
  const retryRead = useTaskAttempts(taskId, !!detail.data?.retry_series, relationLimit)
  const task: TaskInfo | undefined = detail.data
  const schedule = useTaskScheduleEvents(taskId, !!task?.schedule)
  const [menuOpen, setMenuOpen] = useState(false)
  const [holding, setHolding] = useState<TaskHold | null>(null)
  const [reason, setReason] = useState("")

  const refresh = () => void queryClient.invalidateQueries({ queryKey: ["tasks"] })
  const hold = useMutation({
    mutationFn: (h: TaskHold) => api.holdTask(taskId, h, reason.trim() || undefined),
    onSuccess: () => {
      setHolding(null)
      setReason("")
      refresh()
    },
  })
  const resume = useMutation({ mutationFn: () => api.resumeTask(taskId), onSuccess: refresh })

  if (detail.error) {
    return (
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar>Task</TopBar>
        <div className="p-6">
          <ErrorState error={detail.error} />
        </div>
      </div>
    )
  }
  if (!task)
    return (
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar>Task</TopBar>
        <p role="status" className="p-6">
          Loading task…
        </p>
      </div>
    )

  const byId = new Map(
    [
      task,
      ...(linked.data?.rows ?? []),
      ...(dependentsRead.data?.pages.flatMap((page) => page.items) ?? []),
      ...(childrenRead.data?.pages.flatMap((page) => page.items) ?? []),
      ...(retryRead.data?.rows ?? []),
    ].map((item) => [item.id, item]),
  )
  const all = [...byId.values()]
  const group = groupOf(task.status)
  const prerequisites = (task.prerequisite_task_ids ?? []).flatMap((id) => byId.get(id) ?? [])
  const dependents = all.filter((t) => t.prerequisite_task_ids?.includes(task.id))
  const parent = task.parent_task_id ? byId.get(task.parent_task_id) : undefined
  const children = all.filter((t) => t.parent_task_id === task.id)
  const attempts = retryRead.data?.rows ?? (task.retry_series ? [task] : [])
  const held = HELD_STATUSES.has(task.status)
  const at = (iso: string | null) => (iso ? when(Date.parse(iso), now()) : "–")
  const retry = task.retry_series
  const holdLabel = HOLDS.find((h) => h.hold === holding)

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <TopBar
        actions={
          <>
            {held && writable && (
              <Button
                size="sm"
                variant="outline"
                disabled={resume.isPending}
                onClick={() => resume.mutate()}
              >
                <Play />
                {resume.isPending ? "Resuming…" : "Resume"}
              </Button>
            )}
            {task.session_id && (
              <NavLink
                to={sessionPath(task.session_id)}
                className={buttonVariants({ variant: "outline", size: "sm" })}
              >
                Open session
              </NavLink>
            )}
            <div className="relative">
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label="More actions"
                aria-expanded={menuOpen}
                onKeyDown={(event) => {
                  if (event.key === "Escape") setMenuOpen(false)
                }}
                onClick={() => setMenuOpen(!menuOpen)}
              >
                <Ellipsis />
              </Button>
              {menuOpen && (
                <div className="absolute top-9 right-0 z-20 w-60 rounded-lg border border-border bg-popover p-1 text-sm shadow-lg">
                  {writable &&
                    HOLDS.map(({ hold: h, label, icon: Icon }) => (
                      <button
                        key={h}
                        type="button"
                        disabled={!holdable(task)}
                        onClick={() => {
                          setHolding(h)
                          setMenuOpen(false)
                        }}
                        className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-muted disabled:pointer-events-none disabled:opacity-50"
                      >
                        <Icon className="size-4" aria-hidden="true" />
                        {label}
                      </button>
                    ))}
                  {writable && !holdable(task) && (
                    <p className="px-2 py-1 text-xs text-muted-foreground">
                      {held
                        ? "Already held. Resume it first."
                        : task.status === "running"
                          ? "A running session can't be held; stop the session instead."
                          : "Finished tasks can't be held."}
                    </p>
                  )}
                  <div className="my-1 border-t border-border" />
                  <button
                    type="button"
                    onClick={() => {
                      void navigator.clipboard?.writeText(task.id).catch(() => undefined)
                      setMenuOpen(false)
                    }}
                    className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-muted"
                  >
                    <Copy className="size-4" aria-hidden="true" />
                    Copy task ID
                  </button>
                </div>
              )}
            </div>
          </>
        }
      >
        <span className="flex min-w-0 items-center gap-1">
          <NavLink to="/tasks" className="shrink-0 text-muted-foreground hover:text-foreground">
            Tasks
          </NavLink>
          <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
          <span className="truncate">{taskName(task)}</span>
        </span>
      </TopBar>
      <PageBody width="max-w-4xl">
        <h1 className="text-xl font-semibold">{taskName(task)}</h1>
        <p className={cn("mt-1 flex items-center gap-2 text-sm", GROUP_TEXT[group])}>
          <span
            className={cn("size-2 shrink-0 rounded-full", GROUP_DOT[group])}
            aria-hidden="true"
          />
          {statusLine(
            task,
            all,
            now(),
            undefined,
            childrenRead.isSuccess && !childrenRead.data.pages.at(-1)?.next_cursor,
          )}
        </p>
        <p className="mt-1 flex flex-wrap items-center gap-x-2 text-xs text-muted-foreground">
          <span className="font-mono">{task.type}</span>
          {task.assigned_agent_name && <span>· {task.assigned_agent_name}</span>}
          {isCayuTask(task) && (
            <span className="rounded border border-border px-1 text-[10px]">Cayu's own task</span>
          )}
        </p>

        <TaskActions task={task} />
        {!writable && (
          <p className="mt-3 text-xs text-muted-foreground">
            Your role can read tasks; task changes require full access.
          </p>
        )}
        {holding && holdLabel && writable && (
          <form
            className="mt-4 flex flex-wrap items-center gap-2 rounded-xl border border-border bg-card p-3"
            onSubmit={(e) => {
              e.preventDefault()
              hold.mutate(holding)
            }}
          >
            <label htmlFor="task-hold-reason" className="sr-only">
              Reason
            </label>
            <input
              id="task-hold-reason"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Reason (optional), shown on the task"
              className="h-8 min-w-0 flex-1 rounded-lg border border-input bg-background px-2.5 text-sm outline-none focus:border-ring"
            />
            <Button type="submit" size="sm" disabled={hold.isPending}>
              {hold.isPending ? "Saving…" : holdLabel.verb}
            </Button>
            <Button type="button" size="sm" variant="ghost" onClick={() => setHolding(null)}>
              Cancel
            </Button>
            {hold.error && <ErrorState error={hold.error} />}
          </form>
        )}
        {resume.error && <ErrorState error={resume.error} />}

        <dl className="mt-6 border-t border-border pt-2">
          <Row k="Created">{at(task.created_at)}</Row>
          {task.started_at && <Row k="Started">{at(task.started_at)}</Row>}
          {task.completed_at && <Row k="Finished">{at(task.completed_at)}</Row>}
          {!task.schedule && task.available_at && Date.parse(task.available_at) > now() && (
            <Row k={task.retry_series ? "Next attempt" : "Starts"}>{at(task.available_at)}</Row>
          )}
          {task.description && <Row k="Description">{task.description}</Row>}
        </dl>

        {attempts.length > 0 && retry ? (
          <Section title={`Attempts (${attempts.length} loaded)`}>
            <List>
              {attempts.map((a) => {
                return (
                  <div
                    key={a.id}
                    className={cn(
                      "flex min-w-0 items-center gap-3 border-b border-border px-3 py-2 text-sm last:border-0",
                      a.id === task.id && "bg-muted/50",
                    )}
                  >
                    <NavLink to={taskPath(a.id)} className="shrink-0 hover:underline">
                      Attempt {a.retry_series?.attempt}
                    </NavLink>
                    <span
                      className={cn(
                        "min-w-0 flex-1 truncate text-xs",
                        GROUP_TEXT[groupOf(a.status)],
                      )}
                    >
                      {statusLine(a, all, now(), undefined, false)}
                    </span>
                    {a.session_id ? (
                      <NavLink
                        to={sessionPath(a.session_id)}
                        className="flex shrink-0 items-center gap-2 text-xs hover:underline"
                      >
                        <StateBadge overview={undefined} />
                        Session
                      </NavLink>
                    ) : (
                      <span className="shrink-0 text-xs text-muted-foreground">No session yet</span>
                    )}
                  </div>
                )
              })}
            </List>
            {retryRead.error && <ErrorState error={retryRead.error} />}
            {retryRead.data?.incomplete && (
              <Button variant="outline" onClick={() => setRelationLimit((n) => n + 50)}>
                Load more attempts
              </Button>
            )}
            <p className="mt-2 text-xs text-muted-foreground">
              Retries up to {retry.policy.max_attempts} times, waiting{" "}
              {seconds(retry.policy.initial_backoff_seconds)} and then{" "}
              {retry.policy.backoff_multiplier}× longer each time, up to{" "}
              {seconds(retry.policy.max_backoff_seconds)}. Used so far:{" "}
              {Number(retry.cumulative_tokens).toLocaleString("en-US")} tokens, $
              {retry.cumulative_estimated_cost}.
            </p>
          </Section>
        ) : (
          <Section title="Session">
            {task.session_id ? (
              <List>
                <NavLink
                  to={sessionPath(task.session_id)}
                  className="flex min-w-0 items-center gap-3 px-3 py-2 text-sm hover:bg-muted/60"
                >
                  <span className="min-w-0 flex-1 truncate">{task.session_id}</span>
                  <StateBadge overview={undefined} />
                </NavLink>
              </List>
            ) : (
              <p className="text-sm text-muted-foreground">
                {groupOf(task.status) === "waiting"
                  ? "None yet. A session starts when a worker picks the task up."
                  : "This task ran without a session of its own."}
              </p>
            )}
          </Section>
        )}

        {linked.error && <ErrorState error={linked.error} />}
        {linked.data?.issues.map((issue) => (
          <div key={issue.id} role="status">
            <p>Related task {issue.id} is unavailable.</p>
            <ErrorState error={issue.error} />
          </div>
        ))}
        {linked.isFetching && <p role="status">Loading related tasks…</p>}
        {(task.prerequisite_task_ids?.length ?? 0) > relationLimit && (
          <Button variant="outline" onClick={() => setRelationLimit((n) => n + 50)}>
            Load more prerequisites
          </Button>
        )}
        {prerequisites.length > 0 && (
          <Section title={`Waiting on (${prerequisites.length} loaded)`}>
            <List>
              {prerequisites.map((p) => (
                <TaskLink key={p.id} task={p} all={all} now={now()} />
              ))}
            </List>
          </Section>
        )}
        {dependents.length > 0 && (
          <Section title="Needed by">
            <List>
              {dependents.map((d) => (
                <TaskLink key={d.id} task={d} all={all} now={now()} />
              ))}
            </List>
          </Section>
        )}
        {dependentsRead.error && <ErrorState error={dependentsRead.error} />}
        {dependentsRead.hasNextPage && (
          <Button
            variant="outline"
            disabled={dependentsRead.isFetching}
            onClick={() => void dependentsRead.fetchNextPage()}
          >
            Load more dependents
          </Button>
        )}
        {dependentsRead.data?.pages.at(-1)?.next_cursor && !dependentsRead.hasNextPage && (
          <p role="alert">
            Dependent paging stopped at a repeated cursor. Refresh to restart the scan.
          </p>
        )}
        {childrenRead.data?.pages.at(-1)?.next_cursor && !childrenRead.hasNextPage && (
          <p role="alert">
            Subtask paging stopped at a repeated cursor. Refresh to restart the scan.
          </p>
        )}
        {childrenRead.error && <ErrorState error={childrenRead.error} />}
        {childrenRead.hasNextPage && (
          <Button
            variant="outline"
            disabled={childrenRead.isFetching}
            onClick={() => void childrenRead.fetchNextPage()}
          >
            Load more subtasks
          </Button>
        )}
        {!relatedSupported && (
          <p role="status">Dependency and subtask pages are unavailable for this store.</p>
        )}
        {parent && (
          <Section title="Started by">
            <List>
              <TaskLink task={parent} all={all} now={now()} />
            </List>
          </Section>
        )}
        {children.length > 0 && (
          <Section title={`Subtasks (${children.length} loaded)`}>
            <List>
              {children.map((c) => (
                <TaskLink key={c.id} task={c} all={all} now={now()} />
              ))}
            </List>
          </Section>
        )}

        <Section title="Input">
          {Object.keys(task.input).length ? (
            <ValueView label={null} value={task.input} />
          ) : (
            <p className="text-sm text-muted-foreground">No input.</p>
          )}
        </Section>
        {task.result && (
          <Section title="Result">
            <ValueView label={null} value={task.result} />
          </Section>
        )}
        {task.error && (
          <Section title="Error">
            <ValueView label={null} value={task.error} />
          </Section>
        )}

        {task.schedule && (
          <Section title="Schedule">
            <dl>
              <Row k="Fires at">{at(task.available_at)}</Row>
              {task.schedule.policy.expires_at && (
                <Row k="Expires">{at(task.schedule.policy.expires_at)}</Row>
              )}
              <Row k="If missed">
                {task.schedule.policy.misfire_policy === "fire_once"
                  ? `Runs once if it's no more than ${seconds(task.schedule.policy.misfire_grace_seconds)} late`
                  : "Skipped"}
              </Row>
            </dl>
            {schedule.error && <ErrorState error={schedule.error} />}
            {schedule.isPending && <p role="status">Loading schedule history…</p>}
            {schedule.data && schedule.data.pages.flat().length > 0 && (
              <ol className="mt-3 space-y-1.5 border-l border-border pl-4 text-sm">
                {schedule.data.pages.flat().map((e) => (
                  <li key={e.sequence} className="flex gap-3">
                    <span className="w-32 shrink-0 text-muted-foreground tabular-nums">
                      {at(e.occurred_at)}
                    </span>
                    <span>
                      {SCHEDULE_EVENT[e.type] ?? e.type}
                      {e.type === "task.rescheduled" && ` to ${at(e.available_at)}`}
                    </span>
                  </li>
                ))}
              </ol>
            )}
            {schedule.hasNextPage && (
              <Button
                variant="outline"
                disabled={schedule.isFetching}
                onClick={() => void schedule.fetchNextPage()}
              >
                Load more schedule history
              </Button>
            )}
          </Section>
        )}

        <details className="mt-8 rounded-xl border border-border">
          <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-muted-foreground hover:text-foreground">
            Debug
          </summary>
          <dl className="border-t border-border px-3 py-2">
            <Row k="Task ID" mono>
              {task.id}
            </Row>
            <Row k="Status" mono>
              {task.status}
            </Row>
            {task.status_reason && <Row k="Reason">{task.status_reason}</Row>}
            <Row k="Worker" mono>
              {task.worker_id ?? "–"}
            </Row>
            <Row k="Lease expires">{task.lease_expires_at ? at(task.lease_expires_at) : "–"}</Row>
            <Row k="Concurrency key" mono>
              {task.concurrency_key ?? "–"}
            </Row>
            {task.graph_id && (
              <Row k="Graph" mono>
                {task.graph_id}
              </Row>
            )}
            <Row k="Invocation" mono>
              {task.invocation.source} · root {task.invocation.root_invocation_id}
            </Row>
            {retry && (
              <Row k="Retry series" mono>
                {retry.series_id} · {retry.disposition}
              </Row>
            )}
          </dl>
          <div className="space-y-3 border-t border-border px-3 py-3">
            {task.status_payload && (
              <div>
                <p className="mb-1 text-xs text-muted-foreground">Status payload</p>
                <ValueView label={null} value={task.status_payload} />
              </div>
            )}
            <div>
              <p className="mb-1 text-xs text-muted-foreground">Metadata</p>
              <ValueView label={null} value={task.metadata} />
            </div>
            <div>
              <p className="mb-1 text-xs text-muted-foreground">Raw task</p>
              <pre className="max-h-80 overflow-auto rounded-lg bg-muted p-3 font-mono text-xs leading-5">
                {JSON.stringify(task, null, 2)}
              </pre>
            </div>
          </div>
        </details>
      </PageBody>
    </div>
  )
}
