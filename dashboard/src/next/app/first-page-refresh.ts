/**
 * Paged lists poll only their first page.
 *
 * React Query refetches every loaded page of an infinite query on each interval, so a
 * list with five pages loaded makes five requests per tick. With these two pieces the
 * infinite query polls itself only while it holds a single page. Once the reader loads
 * more, a separate query polls page 1 alone and writes it over the first loaded page
 * (only when it changed), leaving the older pages as they were read. Once a minute
 * (`OLDER_PAGES_REFRESH_MS`) that poll re-reads the whole list instead, so rows that
 * moved between pages settle. The page-1 poll owns that schedule: a write to the
 * infinite query restarts the query's own interval timer, so that timer can't. Lists that
 * merge pages keep each row's first occurrence (`firstOccurrences`): a refreshed first
 * page can repeat a row an older page still holds in its earlier state.
 */
import { type InfiniteData, type QueryKey, useQuery, useQueryClient } from "@tanstack/react-query"
import { useEffect, useRef } from "react"

type Interval = number | false

/** How often a list with older pages loaded re-reads all of them, while it polls at all. */
export const OLDER_PAGES_REFRESH_MS = 60_000

/**
 * The infinite query's `refetchInterval` when nothing else polls it: poll at `interval`
 * while one page is loaded, and re-read every loaded page at most once a minute after.
 */
export function firstPageOnly<TPage>(interval: (pages: readonly TPage[]) => Interval) {
  return (query: { state: { data: InfiniteData<TPage, unknown> | undefined } }): Interval => {
    const pages = query.state.data?.pages ?? []
    const every = interval(pages)
    if (pages.length <= 1 || every === false) return every
    return Math.max(every, OLDER_PAGES_REFRESH_MS)
  }
}

/**
 * The infinite query's `refetchInterval` beside `useFirstPageRefresh`: poll at `interval`
 * while one page is loaded; once more are, the page-1 poll takes over, full re-reads too.
 */
export function singlePageOnly<TPage>(interval: (pages: readonly TPage[]) => Interval) {
  return (query: { state: { data: InfiniteData<TPage, unknown> | undefined } }): Interval => {
    const pages = query.state.data?.pages ?? []
    return pages.length > 1 ? false : interval(pages)
  }
}

/**
 * What a page-1 poll does with a fresh page 1: a full re-read when due, or when rows of
 * the old page 1 fell out of the fresh one (they moved past the boundary into a page
 * that was read before they got there), else a write.
 */
export function firstPageAction(
  now: number,
  lastFullRead: number,
  changed: boolean,
  shifted = false,
): "refetch-all" | "write" | "none" {
  if (shifted || now - lastFullRead >= OLDER_PAGES_REFRESH_MS) return "refetch-all"
  return changed ? "write" : "none"
}

/** Whether a row of the old first page is missing from the fresh one. */
export function shiftedOut(oldKeys: readonly string[], freshKeys: readonly string[]): boolean {
  const fresh = new Set(freshKeys)
  return oldKeys.some((key) => !fresh.has(key))
}

/** Each key's first occurrence, in order: the newest copy when page 1 was just refreshed. */
export function firstOccurrences<T>(items: Iterable<T>, key: (item: T) => string): T[] {
  const seen = new Set<string>()
  const kept: T[] = []
  for (const item of items) {
    const id = key(item)
    if (seen.has(id)) continue
    seen.add(id)
    kept.push(item)
  }
  return kept
}

/** Replace the first loaded page, keeping the rest. */
export function withFirstPage<TPage, TParam>(
  data: InfiniteData<TPage, TParam> | undefined,
  first: TPage,
): InfiniteData<TPage, TParam> | undefined {
  if (!data || data.pages.length === 0 || data.pages[0] === first) return data
  return { pages: [first, ...data.pages.slice(1)], pageParams: data.pageParams }
}

/**
 * Poll page 1 of an infinite query while it holds more than one page. `fetchFirst` reads
 * the first page exactly as the infinite query's `queryFn` would; `rowKeys` names a
 * page's rows, so a poll can tell when rows moved past page 1.
 */
export function useFirstPageRefresh<TPage, TParam>({
  queryKey,
  data,
  dataUpdatedAt,
  fetchFirst,
  interval,
  rowKeys,
  enabled = true,
}: {
  queryKey: QueryKey
  data: InfiniteData<TPage, TParam> | undefined
  dataUpdatedAt: number
  fetchFirst: (signal: AbortSignal) => Promise<TPage>
  interval: (pages: readonly TPage[]) => Interval
  rowKeys: (page: TPage) => readonly string[]
  enabled?: boolean
}) {
  const client = useQueryClient()
  const paged = enabled && (data?.pages.length ?? 0) > 1
  const head = useQuery({
    queryKey: [...queryKey, "first-page"],
    enabled: paged,
    queryFn: ({ signal }) => fetchFirst(signal),
    // Start from the page already loaded, so switching modes doesn't re-read it.
    initialData: () => data?.pages[0],
    initialDataUpdatedAt: () => dataUpdatedAt,
    refetchInterval: () => {
      const pages = client.getQueryData<InfiniteData<TPage, TParam>>(queryKey)?.pages ?? []
      return pages.length > 1 ? interval(pages) : false
    },
    refetchOnWindowFocus: false,
    gcTime: 0,
  })
  const first = head.data
  const updated = head.dataUpdatedAt
  const key = useRef(queryKey)
  key.current = queryKey
  const keysOf = useRef(rowKeys)
  keysOf.current = rowKeys
  // When the list last re-read every page; starts when it began holding several.
  const lastFullRead = useRef(0)
  const handled = useRef(0)
  useEffect(() => {
    if (!paged) {
      lastFullRead.current = 0
      return
    }
    if (lastFullRead.current === 0) lastFullRead.current = Date.now()
    if (first === undefined || updated === handled.current) return
    handled.current = updated
    const current = client.getQueryData<InfiniteData<TPage, TParam>>(key.current)
    const old = current?.pages[0]
    const changed = old !== first
    const shifted =
      changed && old !== undefined && shiftedOut(keysOf.current(old), keysOf.current(first))
    const action = firstPageAction(Date.now(), lastFullRead.current, changed, shifted)
    if (action === "refetch-all") {
      lastFullRead.current = Date.now()
      // A full re-read still running finishes; restarting it on every poll could
      // keep a slow multi-page list from ever settling.
      void client.refetchQueries({ queryKey: key.current, exact: true }, { cancelRefetch: false })
    } else if (action === "write")
      client.setQueryData<InfiniteData<TPage, TParam>>(key.current, (data) =>
        withFirstPage(data, first),
      )
  }, [paged, first, updated, client])
}
