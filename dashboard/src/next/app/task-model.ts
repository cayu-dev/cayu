/**
 * How tasks read on the Tasks page: five plain groups instead of twelve raw statuses, and
 * one line per task saying what it's doing or waiting for.
 */
import type { TaskItem } from "../api/types"

export type TaskGroup = "active" | "waiting" | "held" | "failed" | "done"

export const TASK_GROUPS: { id: TaskGroup; label: string; statuses: string[] }[] = [
  { id: "active", label: "Active", statuses: ["claimed", "running"] },
  {
    id: "waiting",
    label: "Waiting",
    statuses: ["pending", "waiting_dependencies", "waiting_group"],
  },
  { id: "held", label: "Held", statuses: ["paused", "blocked", "needs_attention"] },
  { id: "failed", label: "Failed", statuses: ["failed", "dependency_skipped"] },
  { id: "done", label: "Done", statuses: ["completed", "cancelled"] },
]

export function groupOf(status: string): TaskGroup {
  return TASK_GROUPS.find((g) => g.statuses.includes(status))?.id ?? "waiting"
}

export const HELD_STATUSES = new Set(["paused", "blocked", "needs_attention"])
export const FINISHED_STATUSES = new Set(["completed", "failed", "cancelled", "dependency_skipped"])

/** Runtime housekeeping (dispatch, knowledge enrichment, wait deadlines) uses `cayu.*` types. */
export const isCayuTask = (task: Pick<TaskItem, "type">) => task.type.startsWith("cayu.")

export const taskName = (task: Pick<TaskItem, "title" | "type">) => task.title || task.type

const TZ = "America/Los_Angeles"

/** "in 6m", "4:00 PM", "Nov 1, 6:00 AM": near times relative, later ones as a clock time. */
export function when(ms: number, now: number, timeZone = TZ): string {
  if (Math.abs(ms - now) < 60_000) return "just now"
  const minutes = Math.round((ms - now) / 60_000)
  if (minutes > 0 && minutes < 60) return `in ${minutes}m`
  const day = (t: number) => new Date(t).toLocaleDateString("en-CA", { timeZone })
  const time = new Date(ms).toLocaleTimeString("en-US", {
    timeZone,
    hour: "numeric",
    minute: "2-digit",
  })
  if (day(ms) === day(now)) return time
  const date = new Date(ms).toLocaleDateString("en-US", {
    timeZone,
    month: "short",
    day: "numeric",
  })
  return `${date}, ${time}`
}

const HOLD_WORD: Record<string, string> = {
  paused: "Paused",
  blocked: "Blocked",
  needs_attention: "Needs attention",
}

/**
 * What the task is doing or waiting for, in plain words: "Waiting on 2 tasks", "Scheduled
 * for Nov 1, 6:00 AM", "Blocked: vendor file late", "Retry 2 of 3 in 6m".
 */
export function statusLine(task: TaskItem, all: TaskItem[], now: number, timeZone = TZ): string {
  const byId = new Map(all.map((t) => [t.id, t]))
  const retry = task.retry_series
  const reason = task.status_reason ? `: ${task.status_reason}` : ""
  const children = all.filter((t) => t.parent_task_id === task.id).length
  const subtasks = children ? ` · ${children} subtask${children === 1 ? "" : "s"}` : ""
  switch (task.status) {
    case "running":
      return `Running${subtasks}`
    case "claimed":
      return "Starting"
    case "pending": {
      const at = retry?.next_eligible_at ?? task.available_at
      const time = at && Date.parse(at) > now ? when(Date.parse(at), now, timeZone) : null
      const phrase = time ? (time.startsWith("in ") ? ` ${time}` : ` at ${time}`) : ""
      if (retry && retry.attempt > 1)
        return `Retry ${retry.attempt} of ${retry.policy.max_attempts}${phrase}`
      if (time && task.schedule)
        return time.startsWith("in ") ? `Scheduled ${time}` : `Scheduled for ${time}`
      return time ? `Starts${phrase}` : "Queued"
    }
    case "waiting_dependencies": {
      const open = (task.prerequisite_task_ids ?? []).filter(
        (id) => byId.get(id)?.status !== "completed",
      ).length
      return open ? `Waiting on ${open} task${open === 1 ? "" : "s"}` : "Waiting on other tasks"
    }
    case "waiting_group":
      return "Waiting for its group"
    case "paused":
    case "blocked":
    case "needs_attention":
      return `${HOLD_WORD[task.status]}${reason}`
    case "completed":
      return `Completed${subtasks}`
    case "failed": {
      const next = retry?.successor_task_id ? byId.get(retry.successor_task_id) : undefined
      if (!next) return `Failed${reason}`
      const outcome = FINISHED_STATUSES.has(next.status) ? next.status : "queued"
      return `Failed · attempt ${next.retry_series?.attempt ?? (retry?.attempt ?? 1) + 1} ${outcome}`
    }
    case "dependency_skipped":
      return "Skipped: a task it waits on didn't complete"
    case "cancelled":
      return "Cancelled"
    default:
      return task.status
  }
}

/** Whether a hold or resume applies, following the runtime's rules (_lifecycle.py). */
export function holdable(task: TaskItem): boolean {
  if (FINISHED_STATUSES.has(task.status) || HELD_STATUSES.has(task.status)) return false
  return !(task.status === "running" && task.session_id)
}
