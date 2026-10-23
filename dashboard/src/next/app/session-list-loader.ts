import type {
  SessionListQuery,
  SessionsPage,
  SessionTranscriptPage,
  SessionTranscriptQuery,
} from "../../lib/api"
import type { SessionOverviewResponse } from "../../lib/generated/server-api"
import { loadSessionIndexPage } from "../../lib/session-index-pages.ts"
import type { Session } from "../api/types"
import { sessionDisplayTitle } from "./session-model.ts"

export type SessionListRow = {
  session: Session
  overview: SessionOverviewResponse
  title: string
  titleUnavailable: boolean
}

/** Read one visible candidate window; bounded transcript fallbacks never define search membership. */
export async function loadSessionList(
  query: SessionListQuery,
  readPage: (query: SessionListQuery, signal: AbortSignal) => Promise<SessionsPage>,
  readFirstUser: (
    id: string,
    query: SessionTranscriptQuery,
    signal: AbortSignal,
  ) => Promise<SessionTranscriptPage>,
  signal: AbortSignal,
) {
  const result = await loadSessionIndexPage(query, readPage, signal)
  const byId = new Map(result.page.overviews?.map((overview) => [overview.session_id, overview]))
  const rows: SessionListRow[] = result.page.sessions.map((session) => {
    const overview = byId.get(session.id)
    if (!overview) throw new Error("Session overview is unavailable. Refresh the list.")
    return {
      session,
      overview,
      title: sessionDisplayTitle(session, overview.title, undefined),
      titleUnavailable: false,
    }
  })
  const missing = rows.filter(
    (row) => !row.overview.title?.trim() && Object.keys(row.session.labels).length === 0,
  )
  let index = 0
  await Promise.all(
    Array.from({ length: Math.min(4, missing.length) }, async () => {
      while (index < missing.length) {
        signal.throwIfAborted()
        const row = missing[index++]
        if (!row) return
        try {
          const transcript = await readFirstUser(row.session.id, { role: "user", limit: 1 }, signal)
          signal.throwIfAborted()
          row.title = sessionDisplayTitle(row.session, row.overview.title, transcript.messages[0])
        } catch {
          signal.throwIfAborted()
          row.titleUnavailable = true
        }
      }
    }),
  )
  signal.throwIfAborted()
  return { ...result, rows }
}
