import type { SessionDetailSearch } from "./session-detail-search.ts"
import { tabStorage } from "./tab-storage.ts"

/**
 * Which of a session's two views is on screen, and the one it opens on. A session opens on
 * Trace, except where the reader needs Conversation: a link that asks for it or carries only
 * conversation filters, a return to a session last read as a conversation in this browser tab
 * (a session the person just started there counts), or a session waiting on someone (its
 * question and the composer are in Conversation).
 * Only Conversation is remembered, and choosing Trace forgets it, so a session that opened on
 * Trace by default still opens on Conversation once it waits on someone.
 * Storage can be missing or blocked; the last view is then not remembered.
 */
export type SessionViewName = "trace" | "conversation"

export const SESSION_VIEW_PREFIX = "cayu-dashboard-session-view:"

type ViewStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">

/**
 * `opened` describes the session when the page opened: whether it was last read as a
 * conversation in this tab, and whether it was waiting. `waiting` is undefined until the
 * session's state is known, and while it is, the result can be too.
 */
export function sessionView(
  search: SessionDetailSearch,
  opened: { readAsConversation: boolean; waiting: boolean | undefined },
): SessionViewName | undefined {
  if (search.view === "conversation" || search.view === "trace") return search.view
  if (search.transcript_role || search.include_thinking === false) return "conversation"
  if (opened.readAsConversation) return "conversation"
  if (opened.waiting === undefined) return undefined
  return opened.waiting ? "conversation" : "trace"
}

export function readAsConversation(
  sessionId: string,
  storage: ViewStorage | null = tabStorage(),
): boolean {
  try {
    return storage?.getItem(SESSION_VIEW_PREFIX + sessionId) === "conversation"
  } catch {
    return false
  }
}

/** Remember that the session was read as a conversation, or forget it. */
export function rememberConversation(
  sessionId: string,
  remembered: boolean,
  storage: ViewStorage | null = tabStorage(),
): void {
  try {
    if (remembered) storage?.setItem(SESSION_VIEW_PREFIX + sessionId, "conversation")
    else storage?.removeItem(SESSION_VIEW_PREFIX + sessionId)
  } catch {
    // Storage can be unavailable or full; the view then isn't remembered.
  }
}
