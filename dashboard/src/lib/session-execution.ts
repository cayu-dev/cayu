import type { SessionExecutionState } from "./generated/server-api"
import { objectPayload } from "./session-debug.ts"

export function sessionExecutionLabel(execution: SessionExecutionState): string {
  switch (execution.state) {
    case "executing":
      return execution.local_owner ? "Running here" : "Running in another worker"
    case "owner_lost":
      return "Execution owner lost"
    case "waiting":
      return "Waiting"
    case "idle":
      return "Idle"
    case "terminal":
      return "Execution finished"
    case "unknown":
      return "Execution owner unknown"
  }
}

export function sessionDeadlineState(
  metadata: Record<string, unknown>,
  terminalEvent?: { type: string; payload: Record<string, unknown> } | null,
): { expiresAt: string | null; timedOut: boolean } {
  const expiry = objectPayload(metadata["cayu:execution_deadline"])?.expires_at
  const expiresAt =
    typeof expiry === "string" && expiry.length <= 128 && Number.isFinite(Date.parse(expiry))
      ? expiry
      : null
  const timedOut =
    (terminalEvent?.type === "session.failed" || terminalEvent?.type === "session.interrupted") &&
    (objectPayload(terminalEvent.payload.failure_evidence)?.classification === "deadline" ||
      terminalEvent.payload.error_type === "ExecutionDeadlineExceeded")
  return { expiresAt, timedOut }
}
