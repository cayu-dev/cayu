import { useQuery, useQueryClient } from "@tanstack/react-query"
import { useEffect, useRef } from "react"
import { sessionStatusIsActive } from "@/lib/polling"
import { useUnchangedBackoff } from "../../parts/backoff"
import type { SessionsPage } from "../api/types"
import { useApi } from "./services"

const ACTIVITY_POLL_MS = 15_000
const ACTIVITY_WINDOW = 20

/** Newest `updated_at` per agent among the most recently updated sessions. */
export function agentActivity(page: Pick<SessionsPage, "sessions">): Map<string, string> {
  const latest = new Map<string, string>()
  for (const session of page.sessions) {
    const seen = latest.get(session.agent_name)
    if (!seen || session.updated_at > seen) latest.set(session.agent_name, session.updated_at)
  }
  return latest
}

/** Agents whose newest activity differs between two reads (new agents included). */
export function changedAgents(before: Map<string, string>, after: Map<string, string>): string[] {
  return [...after].filter(([agent, at]) => before.get(agent) !== at).map(([agent]) => agent)
}

/**
 * One poll for the Specialists page instead of one per card: read the most recently
 * updated sessions, and refresh only the cards whose agent shows new activity. It polls
 * every 15 s while a session runs and backs off to a minute while nothing changes.
 */
export function useSpecialistActivity(enabled: boolean) {
  const api = useApi()
  const client = useQueryClient()
  const backoff = useUnchangedBackoff(ACTIVITY_POLL_MS)
  const activity = useQuery({
    queryKey: ["specialist-activity"],
    enabled,
    queryFn: ({ signal }) =>
      api.sessions(
        {
          limit: ACTIVITY_WINDOW,
          order_by: "updated_at_desc",
          include_eval: true,
          include_forks: true,
        },
        signal,
      ),
    refetchInterval: (query) =>
      query.state.data?.sessions.some((session) => sessionStatusIsActive(session.status))
        ? ACTIVITY_POLL_MS
        : backoff(query),
  })
  const previous = useRef<Map<string, string> | null>(null)
  useEffect(() => {
    if (!activity.data) return
    const next = agentActivity(activity.data)
    const before = previous.current
    previous.current = next
    if (!before) return
    for (const agent of changedAgents(before, next)) {
      void client.invalidateQueries({ queryKey: ["specialist-recent", agent] })
      void client.invalidateQueries({ queryKey: ["specialist-tasks", agent] })
    }
  }, [activity.data, client])
  return activity
}
