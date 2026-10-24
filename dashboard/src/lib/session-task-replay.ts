import { ApiClientError, fetchSessionTaskReplay, fetchTask, replayTask } from "./api.ts"
import type { TaskReplayBody } from "./generated/server-api"

export type TaskReplayAcceptance = {
  phase: "accepted" | "rejected" | "unknown"
  taskId: string | null
  message: string | null
}

/** One durable task creation; uncertain retries preserve the exact operation tuple. */
class SessionTaskReplay {
  private uncertain = false
  private inFlight = false
  private accepted: TaskReplayAcceptance | null = null
  private readonly body: TaskReplayBody
  constructor(body: TaskReplayBody) {
    this.body = body
  }

  async submit(signal: AbortSignal): Promise<TaskReplayAcceptance> {
    if (this.accepted) return this.accepted
    if (this.inFlight) throw new Error("Task replay is already being submitted.")
    this.inFlight = true
    try {
      const result = await replayTask(this.body, signal)
      if (typeof result.task_id !== "string" || !result.task_id)
        throw new Error("Missing task receipt.")
      this.accepted = { phase: "accepted", taskId: result.task_id, message: null }
      return this.accepted
    } catch (error) {
      if (error instanceof ApiClientError && error.status < 500 && !this.uncertain)
        return { phase: "rejected", taskId: null, message: error.message }
      this.uncertain = true
      return {
        phase: "unknown",
        taskId: null,
        message: "Task creation is unconfirmed. Retry this exact replay to recover its receipt.",
      }
    } finally {
      this.inFlight = false
    }
  }
}

export async function prepareSessionTaskReplay(sessionId: string, signal: AbortSignal) {
  const eligibility = await fetchSessionTaskReplay(sessionId, signal)
  if (!eligibility.available || !eligibility.source)
    throw new Error(
      `Task replay is unavailable: ${eligibility.unavailable_reason ?? "missing source identity"}.`,
    )
  const task = await fetchTask(eligibility.source.task_id, signal)
  if (!task.type) throw new Error("Task replay needs a declared task type.")
  return new SessionTaskReplay(
    Object.freeze({
      operation_id: crypto.randomUUID(),
      type: task.type,
      source: Object.freeze({ ...eligibility.source }),
    }),
  )
}
