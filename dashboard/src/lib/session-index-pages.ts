import type { SessionsSummary, SessionsSummaryQuery } from "./api"

export const MAX_AUTOMATIC_SESSION_PAGES = 25

export type SessionIndexPage = {
  page: SessionsSummary
  searchPaused: boolean
}

/** Find a visible page without treating an empty candidate window as exhaustion. */
export async function loadSessionIndexPage(
  query: SessionsSummaryQuery,
  read: (query: SessionsSummaryQuery, signal: AbortSignal) => Promise<SessionsSummary>,
  signal: AbortSignal,
): Promise<SessionIndexPage> {
  const seen = new Set<string>()
  if (query.cursor) seen.add(query.cursor)
  let current = query
  for (let index = 0; index < MAX_AUTOMATIC_SESSION_PAGES; index += 1) {
    signal.throwIfAborted()
    const page = await read(current, signal)
    signal.throwIfAborted()
    const next = page.next_cursor
    if (next && seen.has(next)) {
      throw Object.assign(new Error("Session search did not advance. Restart the search."), {
        status: 409,
      })
    }
    if (page.sessions.length || !next || page.total_count !== null) {
      return { page, searchPaused: false }
    }
    if (index + 1 === MAX_AUTOMATIC_SESSION_PAGES) return { page, searchPaused: true }
    seen.add(next)
    current = { ...query, cursor: next }
  }
  throw new Error("Session search did not produce a page.")
}
