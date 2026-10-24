import { fetchPendingActions } from "./api.ts"

/** Resolve one current attention identity, including later native candidate pages. */
export async function fetchSessionRecovery(
  sessionId: string,
  attentionId: string | null | undefined,
  signal?: AbortSignal,
) {
  if (typeof attentionId !== "string" || !attentionId.trim())
    throw new Error("Manual recovery requires the exact attention identity.")
  const seen = new Set<string>()
  let cursor: string | null = null
  do {
    const page = await fetchPendingActions(
      { session_id: sessionId, kind: "manual_recovery", limit: 50, cursor },
      signal,
    )
    const action = page.actions.find(
      (candidate) => candidate.kind === "manual_recovery" && candidate.attention_id === attentionId,
    )
    if (action) return action
    cursor = page.next_cursor
    if (cursor && seen.has(cursor))
      throw new Error("Recovery paging did not advance. Refresh the session.")
    if (cursor) seen.add(cursor)
  } while (cursor)
  return null
}
