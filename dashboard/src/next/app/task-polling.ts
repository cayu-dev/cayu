export const TASK_POLL_MS = 5000
export const LIVE_TASK = new Set([
  "pending",
  "claimed",
  "running",
  "waiting_dependencies",
  "waiting_group",
])
