type TaskAvailabilityInput = {
  status: string
  session_id: string | null
  available_at: string | null
}

export function taskAvailabilityDescriptor(task: TaskAvailabilityInput): string | null {
  if (task.status !== "pending") return null
  if (task.session_id) return "Session-bound"
  return task.available_at ? "Time-gated" : "Immediate"
}
