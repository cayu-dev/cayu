/** One exact task creation body survives response loss and view unmounts in this tab. */
export type TaskCreationInput = { type: string; input: Record<string, unknown> }
export type TaskOperationState = {
  phase: "idle" | "pending" | "unknown" | "rejected" | "succeeded"
  body?: TaskCreationInput
  operationId?: string
  taskId?: string
  error?: unknown
}
export class TaskOperation {
  state: TaskOperationState = { phase: "idle" }
  listeners = new Set<() => void>()
  private readonly send: (
    body: TaskCreationInput,
    operationId: string,
  ) => Promise<{ task_id: string }>
  constructor(
    send: (body: TaskCreationInput, operationId: string) => Promise<{ task_id: string }>,
  ) {
    this.send = send
  }
  subscribe = (listener: () => void) => {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }
  snapshot = () => this.state
  private publish(state: TaskOperationState) {
    this.state = state
    for (const listener of this.listeners) listener()
  }
  clear() {
    if (this.state.phase !== "pending" && this.state.phase !== "unknown")
      this.publish({ phase: "idle" })
  }
  async submit(body?: TaskCreationInput) {
    if (this.state.phase === "pending") return
    const retained = this.state.phase === "unknown"
    const request = retained ? this.state.body! : structuredClone(body!)
    const operationId = retained ? this.state.operationId! : crypto.randomUUID()
    this.publish({ phase: "pending", body: request, operationId })
    try {
      const result = await this.send(request, operationId)
      if (typeof result.task_id !== "string" || !result.task_id.trim())
        throw new Error("Task response has no task identity; recover the same operation.")
      this.publish({ phase: "succeeded", body: request, operationId, taskId: result.task_id })
    } catch (error) {
      const status =
        typeof error === "object" && error !== null && "status" in error ? error.status : null
      const definitive =
        !retained && typeof status === "number" && [400, 401, 403, 409, 413, 422].includes(status)
      this.publish({
        phase: definitive ? "rejected" : "unknown",
        body: request,
        operationId,
        error,
      })
    }
  }
}
