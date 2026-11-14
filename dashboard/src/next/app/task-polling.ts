import { FINISHED_STATUSES } from "./task-model.ts"

export const TASK_POLL_MS = 5000
export const LIVE_TASK = new Set([
  "pending",
  "claimed",
  "running",
  "waiting_dependencies",
  "waiting_group",
])

/**
 * Whether a task can still change. Held tasks (paused, blocked, needing attention) can
 * be resumed, so only a finished task stops a poll that watches it.
 */
export const taskCanChange = (status: string) => !FINISHED_STATUSES.has(status)
