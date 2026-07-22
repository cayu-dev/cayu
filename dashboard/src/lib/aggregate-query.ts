const TRANSIENT_HTTP_STATUSES = new Set([408, 425, 429])

type AggregateSnapshotWarningInput = {
  sessions: { accuracy: { kind: string } }
  task_snapshot_status: string
  tasks: { accuracy: { kind: string } } | null
}

function httpStatus(error: unknown): number | null {
  if (error === null || typeof error !== "object" || !("status" in error)) return null
  const status = (error as { status?: unknown }).status
  return typeof status === "number" && Number.isInteger(status) ? status : null
}

export function aggregateRequestFailureIsPermanent(error: unknown): boolean {
  const status = httpStatus(error)
  if (status === null || TRANSIENT_HTTP_STATUSES.has(status)) return false
  return (status >= 400 && status < 500) || status === 501
}

export function retryAggregateRequest(failureCount: number, error: Error): boolean {
  return !aggregateRequestFailureIsPermanent(error) && failureCount < 3
}

export function aggregateSnapshotNeedsWarning(
  snapshot: AggregateSnapshotWarningInput | undefined,
  requestFailed: boolean,
): boolean {
  if (requestFailed) return true
  if (!snapshot) return false
  if (snapshot.sessions.accuracy.kind !== "exact") return true
  if (snapshot.task_snapshot_status === "unsupported") return true
  return snapshot.tasks !== null && snapshot.tasks.accuracy.kind !== "exact"
}
