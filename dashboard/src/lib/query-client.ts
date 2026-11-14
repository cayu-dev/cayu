/**
 * The one React Query client setup the production dashboard and the prototype share.
 *
 * - Data stays fresh for `QUERY_STALE_TIME_MS`, so remounts and route changes reuse it,
 *   except a session's live views (`LIVE_SESSION_QUERY_KEYS`), which a return to the
 *   session reads again.
 * - Window focus doesn't refetch, except the reads that decide what the viewer may do
 *   (`FOCUS_REFRESHED_QUERY_KEYS`), which are refetched whenever the window regains focus,
 *   so a changed role or availability shows without a reload.
 * - A failed read is retried at most once, and never for a 4xx (413 included), a
 *   timeout, or a cancelled request: those fail the same way again.
 */
import { QueryClient } from "@tanstack/react-query"

export const QUERY_STALE_TIME_MS = 10_000

/** Reads that still refresh on every window focus: identity, eval targets, a session's detail. */
export const FOCUS_REFRESHED_QUERY_KEYS: readonly (readonly string[])[] = [
  ["dashboard-identity"],
  ["evals", "targets"],
  ["session"],
]

/** A session's live views: never reused from the cache on mount, so a return shows its latest. */
export const LIVE_SESSION_QUERY_KEYS: readonly (readonly string[])[] = [
  ["transcript"],
  ["overview"],
  ["state"],
  ["session-state"],
]

export const QUERY_MAX_RETRIES = 1

const TIMEOUT_STATUSES = new Set([408, 504])

function errorStatus(error: unknown): number | null {
  if (!error || typeof error !== "object" || !("status" in error)) return null
  const status = (error as { status: unknown }).status
  return typeof status === "number" ? status : null
}

/** Whether a read failed in a way that retrying the same request can't fix. */
export function queryErrorIsFinal(error: unknown): boolean {
  const status = errorStatus(error)
  if (status !== null && ((status >= 400 && status < 500) || TIMEOUT_STATUSES.has(status)))
    return true
  if (error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError"))
    return true
  return false
}

export function shouldRetryQuery(failureCount: number, error: unknown): boolean {
  return failureCount < QUERY_MAX_RETRIES && !queryErrorIsFinal(error)
}

export function createDashboardQueryClient(): QueryClient {
  const client = new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: QUERY_STALE_TIME_MS,
        retry: shouldRetryQuery,
        refetchOnWindowFocus: false,
      },
      mutations: { retry: 0 },
    },
  })
  // A query's own options still win, so a longer staleTime set at the call site holds.
  for (const queryKey of FOCUS_REFRESHED_QUERY_KEYS)
    client.setQueryDefaults(queryKey, { staleTime: 0, refetchOnWindowFocus: true })
  for (const queryKey of LIVE_SESSION_QUERY_KEYS)
    client.setQueryDefaults(queryKey, { staleTime: 0 })
  return client
}
