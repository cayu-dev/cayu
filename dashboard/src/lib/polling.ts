import type { OperationalSnapshot } from "./api"
import { sumCounts } from "./format.ts"

// Pages poll on their fast interval only while the runtime is doing work, back off
// while they wait on a human, and otherwise rely on focus refetches and slow polling.
export const IDLE_POLL_INTERVAL_MS = 30_000
const WAITING_POLL_CAP_MS = 60_000
export const WAITING_POLL_BACKOFF_MS = [5_000, 10_000, 30_000, WAITING_POLL_CAP_MS] as const

const ACTIVE_SESSION_STATUSES = new Set(["pending", "running", "interrupting"])
const ACTIVE_TASK_STATUSES = new Set(["pending", "claimed", "running"])

export function sessionStatusIsActive(status: string): boolean {
  return ACTIVE_SESSION_STATUSES.has(status)
}

export function taskStatusIsActive(status: string): boolean {
  return ACTIVE_TASK_STATUSES.has(status)
}

export function operationalSnapshotIsActive(snapshot: OperationalSnapshot | undefined): boolean {
  if (snapshot === undefined) return false
  const sessions = snapshot.sessions.counts_by_status
  if (sumCounts(sessions.pending, sessions.running, sessions.interrupting) !== "0") return true
  const tasks = snapshot.tasks?.counts_by_status
  return tasks !== undefined && sumCounts(tasks.pending, tasks.claimed, tasks.running) !== "0"
}

export function activityPollInterval(active: boolean, activeIntervalMs: number): number {
  return active ? activeIntervalMs : IDLE_POLL_INTERVAL_MS
}

export function waitingPollInterval(unchangedResponses: number): number {
  return WAITING_POLL_BACKOFF_MS[Math.max(0, unchangedResponses)] ?? WAITING_POLL_CAP_MS
}

/**
 * Counts consecutive successful responses whose data did not change, so a
 * `refetchInterval` callback can back off while nothing happens. TanStack Query
 * evaluates `refetchInterval` several times per response, so each response is
 * observed once by its `dataUpdateCount`. A new query hash starts a new count.
 */
export class UnchangedResponseCounter {
  #queryHash: string | null = null
  #dataUpdateCount = -1
  #fingerprint: string | null = null
  #unchanged = 0

  observe(queryHash: string, dataUpdateCount: number, fingerprint: string | null): number {
    if (queryHash !== this.#queryHash) {
      this.#queryHash = queryHash
      this.#dataUpdateCount = dataUpdateCount
      this.#fingerprint = fingerprint
      this.#unchanged = 0
    } else if (dataUpdateCount !== this.#dataUpdateCount) {
      this.#dataUpdateCount = dataUpdateCount
      if (fingerprint !== null && fingerprint === this.#fingerprint) {
        this.#unchanged += 1
      } else {
        this.#fingerprint = fingerprint
        this.#unchanged = 0
      }
    }
    return this.#unchanged
  }

  /** Restart the backoff; the next response counts as a change. */
  reset(): void {
    this.#fingerprint = null
    this.#unchanged = 0
  }
}

export function responseFingerprint(data: unknown): string | null {
  return data === undefined ? null : JSON.stringify(data)
}
