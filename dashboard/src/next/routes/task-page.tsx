/**
 * One task: what it's doing or waiting for, the session that ran it (every attempt's, when
 * it retries), the tasks it waits on or started, its schedule, input and result. Queue
 * internals (worker, lease, concurrency key, invocation) sit in Debug.
 */
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useParams } from "@tanstack/react-router"
import { ChevronRight, Copy, Ellipsis, Flag, Hand, Pause, Play } from "lucide-react"
import { type ReactNode, useState } from "react"
import { Button, buttonVariants } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { TaskHold, TaskInfo, TaskItem, TaskScheduleEvent } from "../api/types"
import { useSessionRows, useTask, useTaskScheduleEvents, useTasks } from "../app/queries"
import { useApi, useNow } from "../app/services"
import {
  groupOf,
  HELD_STATUSES,
  holdable,
  isCayuTask,
  statusLine,
  taskName,
  when,
} from "../app/task-model"
import { ErrorState, NavLink, PageBody, StateBadge, sessionPath, TopBar, taskPath } from "./common"
import { ValueView } from "./session/value-view"
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
        {statusLine(task, all, now)}
      </span>
    </NavLink>
  )
}

const List = ({ children }: { children: ReactNode }) => (
  <div className="overflow-hidden rounded-xl border border-border">{children}</div>
)

/** Every attempt in the task's retry series, first to last. */
function attemptsOf(task: TaskItem, all: TaskItem[]): TaskItem[] {
  const byId = new Map(all.map((t) => [t.id, t]))
  let first = task
  for (let prev = task.retry_series?.predecessor_task_id; prev; ) {
    const p = byId.get(prev)
    if (!p) break
    first = p
    prev = p.retry_series?.predecessor_task_id
  }
  const chain = [first]
  for (let next = first.retry_series?.successor_task_id; next; ) {
    const n = byId.get(next)
    if (!n) break
    chain.push(n)
    next = n.retry_series?.successor_task_id
  }
  return chain
}

const seconds = (s: number) => (s >= 3600 ? `${s / 3600}h` : s >= 60 ? `${s / 60}m` : `${s}s`)

export function TaskPage() {
  const { taskId } = useParams({ strict: false }) as { taskId: string }
  const api = useApi()
  const now = useNow()
  const queryClient = useQueryClient()
  const detail = useTask(taskId)
  const tasks = useTasks()
  const sessions = useSessionRows()
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
      </div>
    )

  const all = tasks.data ?? [task]
  const byId = new Map(all.map((t) => [t.id, t]))
  const group = groupOf(task.status)
  const rows = sessions.data?.rows ?? []
  const sessionRow = rows.find((r) => r.session.id === task.session_id)
  const prerequisites = (task.prerequisite_task_ids ?? []).flatMap((id) => byId.get(id) ?? [])
  const dependents = all.filter((t) => t.prerequisite_task_ids?.includes(task.id))
  const parent = task.parent_task_id ? byId.get(task.parent_task_id) : undefined
  const children = all.filter((t) => t.parent_task_id === task.id)
  const attempts = task.retry_series ? attemptsOf(task, all) : []
  const held = HELD_STATUSES.has(task.status)
  const at = (iso: string | null) => (iso ? when(Date.parse(iso), now()) : "–")
  const retry = task.retry_series
  const holdLabel = HOLDS.find((h) => h.hold === holding)

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <TopBar
        actions={
          <>
            {held && (
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
                onClick={() => setMenuOpen(!menuOpen)}
              >
                <Ellipsis />
              </Button>
              {menuOpen && (
                <div className="absolute top-9 right-0 z-20 w-60 rounded-lg border border-border bg-popover p-1 text-sm shadow-lg">
                  {HOLDS.map(({ hold: h, label, icon: Icon }) => (
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
                  {!holdable(task) && (
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
          {statusLine(task, all, now())}
        </p>
        <p className="mt-1 flex flex-wrap items-center gap-x-2 text-xs text-muted-foreground">
          <span className="font-mono">{task.type}</span>
          {task.assigned_agent_name && <span>· {task.assigned_agent_name}</span>}
          {isCayuTask(task) && (
            <span className="rounded border border-border px-1 text-[10px]">Cayu's own task</span>
          )}
        </p>

        {holding && holdLabel && (
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
          <Section title={`Attempts (${attempts.length} of ${retry.policy.max_attempts})`}>
            <List>
              {attempts.map((a) => {
                const s = rows.find((r) => r.session.id === a.session_id)
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
                      {statusLine(a, all, now())}
                    </span>
                    {a.session_id ? (
                      <NavLink
                        to={sessionPath(a.session_id)}
                        className="flex shrink-0 items-center gap-2 text-xs hover:underline"
                      >
                        <StateBadge overview={s?.overview} />
                        Session
                      </NavLink>
                    ) : (
                      <span className="shrink-0 text-xs text-muted-foreground">No session yet</span>
                    )}
                  </div>
                )
              })}
            </List>
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
                  <span className="min-w-0 flex-1 truncate">
                    {sessionRow?.overview?.title ?? task.session_id}
                  </span>
                  <StateBadge overview={sessionRow?.overview} />
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

        {prerequisites.length > 0 && (
          <Section
            title={`Waiting on (${prerequisites.filter((p) => p.status !== "completed").length} of ${prerequisites.length} still open)`}
          >
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
        {parent && (
          <Section title="Started by">
            <List>
              <TaskLink task={parent} all={all} now={now()} />
            </List>
          </Section>
        )}
        {children.length > 0 && (
          <Section title={`Subtasks (${children.length})`}>
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
            {schedule.data && schedule.data.length > 0 && (
              <ol className="mt-3 space-y-1.5 border-l border-border pl-4 text-sm">
                {schedule.data.map((e) => (
                  <li key={e.sequence} className="flex gap-3">
                    <span className="w-32 shrink-0 text-muted-foreground tabular-nums">
                      {at(e.occurred_at)}
                    </span>
                    <span>
                      {SCHEDULE_EVENT[e.type]}
                      {e.type === "task.rescheduled" && ` to ${at(e.available_at)}`}
                    </span>
                  </li>
                ))}
              </ol>
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
