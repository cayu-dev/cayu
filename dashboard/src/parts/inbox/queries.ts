/**
 * The Inbox's reads: one page of items at a time, refreshing the newest, and the exact
 * counts by kind. Both read the host's inbox client (../host), so the Developer dashboard
 * and an app that imports the Inbox share them.
 */
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { useEffect, useMemo, useRef, useState } from "react"
import { type InboxCountsQuery, useInboxHost } from "../host"
import type { InboxQuery } from "../types"
import { inboxKeys } from "./keys"

/**
 * One page of the Inbox. `poll` refreshes the newest page while it shows: every
 * `pollMs` (default 5 s), and on each change `api.watch` reports when the host passes it.
 */
export function useInbox(query: InboxQuery = {}, poll = true) {
  const { api, pollMs = 5000 } = useInboxHost().inbox
  const client = useQueryClient()
  const key = useMemo(
    () =>
      inboxKeys.list({
        session_id: query.session_id,
        agent_name: query.agent_name,
        q: query.q,
        kind: query.kind,
        limit: query.limit,
      }),
    [query.session_id, query.agent_name, query.q, query.kind, query.limit],
  )
  const identity = JSON.stringify(key)
  const [position, setPosition] = useState({
    identity,
    cursor: null as string | null,
    visited: [] as string[],
    number: 1,
  })
  const current =
    position.identity === identity ? position : { identity, cursor: null, visited: [], number: 1 }
  const result = useQuery({
    queryKey: [...key, current.cursor],
    queryFn: ({ signal }) => api.list({ limit: 50, ...query, cursor: current.cursor }, signal),
    retry: false,
    refetchInterval: poll && current.cursor === null ? pollMs : false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })
  const page = result.data
  const next = page?.next_cursor
  const cycleError =
    next && current.visited.includes(next)
      ? new Error("The Inbox source did not advance. Refresh the list.")
      : null
  const advance = () => {
    if (next && !cycleError && !result.isFetching)
      setPosition({
        identity,
        cursor: next,
        visited: [...current.visited, next],
        number: current.number + 1,
      })
  }
  const watching = poll && current.cursor === null && !!api.watch
  const { session_id, agent_name, q, kind } = query
  useEffect(() => {
    if (!watching || !api.watch) return
    return api.watch(
      { session_id, agent_name, q, kind },
      () => void client.invalidateQueries({ queryKey: inboxKeys.all }),
    )
  }, [watching, api, client, session_id, agent_name, q, kind])
  const restarted = useRef<string | null>(null)
  useEffect(() => {
    if (
      (result.error as { status?: unknown } | null)?.status === 422 &&
      current.cursor &&
      restarted.current !== identity
    ) {
      restarted.current = identity
      setPosition({ identity, cursor: null, visited: [], number: 1 })
      void client.resetQueries({ queryKey: [...key, null], exact: true })
    }
  }, [result.error, current.cursor, identity, client, key])
  const refresh = async () => {
    await client.cancelQueries({ queryKey: key })
    setPosition({ identity, cursor: null, visited: [], number: 1 })
    await client.resetQueries({ queryKey: [...key, null], exact: true })
  }
  return {
    ...result,
    refetch: refresh,
    fetchNextPage: advance,
    isFetchingNextPage: result.isFetching,
    historical: current.cursor !== null,
    error: result.error ?? cycleError,
    isError: result.isError || !!cycleError,
    isSuccess: result.isSuccess && !cycleError,
    hasNextPage: !!next && !cycleError,
    data: page ? { ...page, issues: page.issues ?? [] } : undefined,
  }
}

/** Exact counts are separate from list pagination and may deliberately be unavailable. */
export function useInboxCounts(query: InboxCountsQuery = {}) {
  const { api, enabled } = useInboxHost().inbox
  return useQuery({
    enabled,
    queryKey: inboxKeys.counts(query),
    queryFn: ({ signal }) => api.counts(query, signal),
    refetchInterval: 30000,
  })
}
