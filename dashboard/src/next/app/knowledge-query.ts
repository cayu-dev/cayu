import { useQuery, useQueryClient } from "@tanstack/react-query"
import { useCallback, useState } from "react"
import { useUnchangedBackoff } from "../../parts/backoff"
import { useApi } from "./services"

const firstPage = { cursor: null as string | null, visited: [] as string[], number: 1 }

/** One active page bounds timer, focus, and mutation-driven refresh work alike. */
export function useKeptKnowledge(enabled: boolean) {
  const api = useApi()
  const client = useQueryClient()
  const [position, setPosition] = useState(firstPage)
  const backoff = useUnchangedBackoff(position.cursor === null ? 10_000 : false)
  const key = ["knowledge", "kept", position.cursor]
  const result = useQuery({
    queryKey: key,
    enabled,
    queryFn: ({ signal }) => api.knowledge({ limit: 50, cursor: position.cursor }, signal),
    retry: false,
    refetchInterval: backoff,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })
  const next = result.data?.next_cursor
  const cycle = !!next && position.visited.includes(next)
  const advance = useCallback(() => {
    if (!next || cycle || result.isFetching) return
    setPosition({ cursor: next, visited: [...position.visited, next], number: position.number + 1 })
  }, [next, cycle, result.isFetching, position])
  const refresh = async () => {
    // Reset only the first page, not every previously visited cursor query.
    await client.cancelQueries({ queryKey: ["knowledge", "kept"] })
    setPosition(firstPage)
    await client.resetQueries({ queryKey: ["knowledge", "kept", null], exact: true })
  }
  return {
    ...result,
    refetch: refresh,
    fetchNextPage: advance,
    entries: result.data?.entries ?? [],
    historical: position.cursor !== null,
    error:
      result.error ??
      (cycle ? new Error("The knowledge source did not advance. Refresh the list.") : null),
    hasNextPage: !!next && !cycle,
    complete: result.isSuccess && !result.isFetching && !result.error && !cycle && !next,
    usageUnavailable: !!result.data?.usage_unavailable_reason,
  }
}
