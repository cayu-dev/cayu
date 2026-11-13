/**
 * The query keys parts use. Every one starts with "cayu", so a part's caches never meet
 * the host's own keys (an app's "overview", say) and a host can refresh a part's data by
 * key instead of by coincidence.
 *
 * Protected review gets no keys: protected material stays out of shared caches.
 */

export const partKeys = {
  all: ["cayu"] as const,
  /** A knowledge proposal's full text at one revision. */
  knowledgeEntry: (entryId: string | undefined, revision: number | undefined) =>
    ["cayu", "knowledge-entry", entryId, revision] as const,
  /** A session's pending recoveries, or all of them; for invalidating. */
  sessionRecoveries: (sessionId?: string) =>
    sessionId === undefined
      ? (["cayu", "session-recovery"] as const)
      : (["cayu", "session-recovery", sessionId] as const),
  /** The pending recovery behind one inbox item. */
  sessionRecovery: (sessionId: string | null | undefined, attentionId: string | null | undefined) =>
    ["cayu", "session-recovery", sessionId, attentionId] as const,
}
