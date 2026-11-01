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

export type SessionListOptions = {
  /** The server returns each row's first user text with the overview (`include=first_message`). */
  firstMessage?: boolean
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
  options: SessionListOptions = {},
) {
  const firstMessage = options.firstMessage === true && !!query.include?.includes("overview")
  const selection: SessionListQuery =
    firstMessage && !query.include?.includes("first_message")
      ? { ...query, include: [...(query.include ?? []), "first_message"] }
      : query
  const result = await loadSessionIndexPage(selection, readPage, signal)
  const byId = new Map(result.page.overviews?.map((overview) => [overview.session_id, overview]))
  const rows: SessionListRow[] = result.page.sessions.map((session) => {
    const overview = byId.get(session.id)
    if (!overview) throw new Error("Session overview is unavailable. Refresh the list.")
    return {
      session,
      overview,
      title: sessionDisplayTitle(
        session,
        overview.title,
        firstMessage && overview.first_message
          ? { role: "user", content: [{ type: "text", text: overview.first_message }] }
          : undefined,
      ),
      titleUnavailable:
        firstMessage &&
        overview.first_message_unavailable === true &&
        !overview.title?.trim() &&
        Object.keys(session.labels).length === 0,
    }
  })
  // The server already answered for every row; older servers need one bounded read per row.
  const missing = firstMessage
    ? []
    : rows.filter(
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
