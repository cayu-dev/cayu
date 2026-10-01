import type { SessionExecutionState } from "./generated/server-api"

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
