/**
 * Pure paging helpers for the reads that span several server pages: a session's events
 * and the inbox. They take the page reader as an argument, so real.ts and the tests use
 * the same code.
 */
import type { InboxItem, InboxPage, InboxQuery, SessionEvent, SessionEventsPage } from "./types.ts"

/** The server's page caps (`_EVENT_PAGE_LIMIT_MAX`, `_TRANSCRIPT_PAGE_LIMIT_MAX` in routes.py). */
export const EVENT_PAGE_MAX = 1000
export const TRANSCRIPT_PAGE_MAX = 1000
/** How many events the trace keeps in memory; older ones are dropped and noted. */
export const EVENT_CAP = 10_000
/** How long a live tail read waits for new events (the server allows up to 25s). */
export const EVENT_WAIT_SECONDS = 10

/** A session's events as the trace holds them: oldest first, at most `EVENT_CAP`. */
export type EventHistory = {
  events: SessionEvent[]
  /** Older events exist that weren't loaded (the cap was hit). */
  truncated: boolean
  /** The tail cursor: the highest sequence read or scanned so far. */
  through: number | null
}

type EventPageReader = (query: {
  limit: number
  order_by: "sequence_asc" | "sequence_desc"
  after_sequence?: number
  before_sequence?: number
  wait_seconds?: number
}) => Promise<SessionEventsPage>

/**
 * Merge events into a history: dedupe by sequence, keep sequence order, and keep only
 * the newest `cap`, marking the history truncated when anything was dropped.
 */
export function mergeEvents(
  history: EventHistory,
  incoming: SessionEvent[],
  cap = EVENT_CAP,
  through: number | null = null,
): EventHistory {
  const bySequence = new Map<number, SessionEvent>()
  for (const e of history.events) bySequence.set(e.sequence, e)
  for (const e of incoming) bySequence.set(e.sequence, e)
  const merged = [...bySequence.values()].sort((a, b) => a.sequence - b.sequence)
  const dropped = Math.max(0, merged.length - cap)
  const events = dropped ? merged.slice(dropped) : merged
  const newest = events.at(-1)?.sequence ?? null
  const candidates = [history.through, through, newest].filter((n): n is number => n != null)
  return {
    events,
    truncated: history.truncated || dropped > 0,
    through: candidates.length ? Math.max(...candidates) : null,
  }
}

/**
 * Load a session's history newest first, a page of at most `EVENT_PAGE_MAX` at a time,
 * walking back with `before_sequence` until the start or the cap. Paging backwards means
 * the cap drops the oldest events, so the live tail (`after_sequence`) continues from
 * the newest one.
 */
export async function loadEventHistory(
  read: EventPageReader,
  cap = EVENT_CAP,
  pageSize = EVENT_PAGE_MAX,
): Promise<EventHistory> {
  let history: EventHistory = { events: [], truncated: false, through: null }
  let before: number | undefined
  for (;;) {
    const limit = Math.min(pageSize, EVENT_PAGE_MAX, cap - history.events.length)
    if (limit <= 0) return { ...history, truncated: true }
    const page = await read({
      limit,
      order_by: "sequence_desc",
      ...(before === undefined ? {} : { before_sequence: before }),
    })
    history = mergeEvents(history, page.events, cap)
    const oldest = page.events.reduce<number | null>(
      (min, e) => (min === null || e.sequence < min ? e.sequence : min),
      null,
    )
    if (!page.has_more || oldest === null) return history
    before = oldest
  }
}

/**
 * Read the events after the history's newest one and merge them in. While the session is
 * live, the first read long-polls (`wait_seconds`) so a quiet session costs one request
 * per wait instead of one per poll.
 */
export async function loadEventTail(
  read: EventPageReader,
  history: EventHistory,
  options: { wait?: number; cap?: number; maxPages?: number } = {},
): Promise<EventHistory> {
  const cap = options.cap ?? EVENT_CAP
  let next = history
  for (let page = 0; page < (options.maxPages ?? 10); page++) {
    const result = await read({
      limit: EVENT_PAGE_MAX,
      order_by: "sequence_asc",
      ...(next.through == null ? {} : { after_sequence: next.through }),
      ...(page === 0 && options.wait ? { wait_seconds: options.wait } : {}),
    })
    next = mergeEvents(next, result.events, cap, result.scan_through_sequence ?? null)
    if (!result.has_more || result.events.length === 0) break
  }
  return next
}

/** Keep the items an inbox query asks for, for servers that ignore the filters. */
export function filterInboxItems(items: InboxItem[], query: InboxQuery): InboxItem[] {
  return items.filter(
    (i) =>
      (!query.session_id || i.session_id === query.session_id) &&
      (!query.agent_name || i.agent_name === query.agent_name) &&
      (!query.kind || i.kind === query.kind),
  )
}

/** The most inbox pages one read follows before it stops and reports `has_more`. */
export const INBOX_MAX_PAGES = 50

/**
 * Read the whole inbox: follow `next_cursor` until it is null, including across empty
 * pages (a caller the inbox access rule limits gets those, with sealed cursors). A 422
 * means the cursor expired or can't be opened; restart once from no cursor, dropping
 * the pages read so far, as `client.watchInbox` does. `issues` from every page are kept.
 */
export async function readAllInbox(
  read: (cursor: string | null) => Promise<InboxPage>,
  options: { maxPages?: number; isStaleCursor?: (error: unknown) => boolean } = {},
): Promise<InboxPage> {
  const maxPages = options.maxPages ?? INBOX_MAX_PAGES
  const stale = options.isStaleCursor ?? (() => false)
  let restarted = false
  for (;;) {
    const items: InboxItem[] = []
    const issues: NonNullable<InboxPage["issues"]> = []
    let cursor: string | null = null
    try {
      for (let page = 0; page < maxPages; page++) {
        const result: InboxPage = await read(cursor)
        items.push(...result.items)
        issues.push(...(result.issues ?? []))
        cursor = result.next_cursor
        if (cursor === null) break
      }
    } catch (error) {
      if (cursor !== null && !restarted && stale(error)) {
        restarted = true
        continue
      }
      throw error
    }
    const seen = new Set<string>()
    const unique = items.filter((i) => !seen.has(i.id) && seen.add(i.id))
    return { items: unique, issues, next_cursor: cursor, has_more: cursor !== null }
  }
}
