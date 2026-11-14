/**
 * Polling that backs off while a read keeps answering the same data, so an open page
 * that nothing changes settles to one read a minute.
 */
import { useCallback, useState } from "react"

/** The longest any backed-off poll waits between reads. */
export const POLL_BACKOFF_CAP_MS = 60_000

/** `baseMs`, doubled for each response in a row that didn't change, capped at a minute. */
export function unchangedBackoffInterval(unchanged: number, baseMs: number): number {
  return Math.min(POLL_BACKOFF_CAP_MS, baseMs * 2 ** Math.max(0, unchanged))
}

type PolledQuery = { queryHash: string; state: { data: unknown; dataUpdateCount: number } }

/**
 * A `refetchInterval` that backs off while a polled read keeps answering the same data.
 * React Query's structural sharing keeps the previous `data` reference when a response
 * is deep-equal to it, so "unchanged" is a reference check, with no serialization. Any
 * change, or a new query key, returns to `baseMs`.
 */
export class UnchangedDataBackoff {
  #queryHash: string | null = null
  #dataUpdateCount = -1
  #data: unknown
  #unchanged = 0

  interval(query: PolledQuery, baseMs: number): number {
    const { data, dataUpdateCount } = query.state
    if (query.queryHash !== this.#queryHash) {
      this.#queryHash = query.queryHash
      this.#dataUpdateCount = dataUpdateCount
      this.#data = data
      this.#unchanged = 0
    } else if (dataUpdateCount !== this.#dataUpdateCount) {
      this.#dataUpdateCount = dataUpdateCount
      if (data !== undefined && data === this.#data) this.#unchanged += 1
      else {
        this.#data = data
        this.#unchanged = 0
      }
    }
    return unchangedBackoffInterval(this.#unchanged, baseMs)
  }
}

/**
 * A `refetchInterval` for reads that can change at any time but usually don't: poll every
 * `baseMs` while the data changes, and back off to a minute while it stays the same.
 * Pass `false` to stop polling.
 */
export function useUnchangedBackoff(baseMs: number | false) {
  const [backoff] = useState(() => new UnchangedDataBackoff())
  return useCallback(
    (query: PolledQuery) => (baseMs === false ? false : backoff.interval(query, baseMs)),
    [backoff, baseMs],
  )
}
