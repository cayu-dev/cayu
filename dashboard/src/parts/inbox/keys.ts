/**
 * The Inbox's query keys, all under `inboxKeys.all`, and what a part does after it settled
 * something: refresh exactly those, then let the host refresh its own.
 */
import type { QueryClient } from "@tanstack/react-query"
import type { InboxCountsQuery, PartChange, PartsHost } from "../host.tsx"
import { partKeys } from "../keys.ts"
import type { InboxQuery } from "../types.ts"

/** The filters a list is read with; the cursor is the last element of a page's key. */
export type InboxListFilters = Pick<
  InboxQuery,
  "session_id" | "agent_name" | "q" | "kind" | "limit"
>

const all = [...partKeys.all, "inbox"] as const

export const inboxKeys = {
  all,
  /** One page of the list (with `cursor`), or every page of these filters. */
  list: (filters: InboxListFilters, cursor?: string | null) =>
    cursor === undefined
      ? ([...all, "list", filters] as const)
      : ([...all, "list", filters, cursor] as const),
  /** Exact counts by kind, or every count (without filters). */
  counts: (filters?: InboxCountsQuery) =>
    filters === undefined ? ([...all, "counts"] as const) : ([...all, "counts", filters] as const),
  sessionTitles: (ids: string[]) => [...all, "session-titles", ids] as const,
}

/**
 * After a settle or a recovery: refresh the Inbox's own queries, then await the host's
 * `onSettled` once, so the host refreshes what it shows by contract.
 */
export async function refreshAfterChange(
  client: Pick<QueryClient, "invalidateQueries">,
  host: Pick<PartsHost, "onSettled">,
  change: PartChange,
): Promise<void> {
  await client.invalidateQueries({ queryKey: inboxKeys.all })
  await host.onSettled?.(change)
}
