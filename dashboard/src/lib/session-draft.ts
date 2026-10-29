/**
 * A session's unsent composer text, kept per session in this browser tab so leaving the
 * session or reloading the page doesn't lose it. The draft is cleared when it is sent.
 * Storage can be missing or blocked; the draft then lasts while the page stays open.
 */
export const SESSION_DRAFT_PREFIX = "cayu-dashboard-draft:"

type DraftStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">

function tabStorage(): DraftStorage | null {
  try {
    return window.sessionStorage
  } catch {
    return null
  }
}

export function readSessionDraft(
  sessionId: string,
  storage: DraftStorage | null = tabStorage(),
): string {
  try {
    return storage?.getItem(SESSION_DRAFT_PREFIX + sessionId) ?? ""
  } catch {
    return ""
  }
}

/** Remember `text` for the session; an empty draft is forgotten. */
export function writeSessionDraft(
  sessionId: string,
  text: string,
  storage: DraftStorage | null = tabStorage(),
): void {
  try {
    if (text) storage?.setItem(SESSION_DRAFT_PREFIX + sessionId, text)
    else storage?.removeItem(SESSION_DRAFT_PREFIX + sessionId)
  } catch {
    // Storage can be unavailable or full; the draft then lasts for this page only.
  }
}

export function clearSessionDraft(
  sessionId: string,
  storage: DraftStorage | null = tabStorage(),
): void {
  writeSessionDraft(sessionId, "", storage)
}
