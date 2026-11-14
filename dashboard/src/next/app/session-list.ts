import { useQuery, useQueryClient } from "@tanstack/react-query"
import { activityPollInterval } from "@/lib/polling"
import { shouldRetryQuery } from "@/lib/query-client"
import { sessionIndexErrorState } from "@/lib/session-index-error"
import { type SessionIndexSearch, sessionOverviewQuery } from "@/lib/session-index-search"
import { useApi, useServices } from "./services"
import { loadSessionList } from "./session-list-loader"

export function useSessionList(search: SessionIndexSearch) {
  const api = useApi()
  const { overviewFirstMessageEnabled } = useServices()
  const client = useQueryClient()
  const selection = sessionOverviewQuery(search)
  return useQuery({
    queryKey: ["session-list", selection, overviewFirstMessageEnabled === true],
    queryFn: ({ signal }) =>
      loadSessionList(
        selection,
        (query, signal) => api.sessions(query, signal),
        (id, query) =>
          client.fetchQuery({
            queryKey: ["session-list-first-user", id],
            queryFn: ({ signal }) => api.transcript(id, query, signal),
            staleTime: 30_000,
            retry: false,
          }),
        signal,
        { firstMessage: overviewFirstMessageEnabled },
      ),
    retry: (count, error) =>
      shouldRetryQuery(count, error) && sessionIndexErrorState(error).retryable,
    refetchOnWindowFocus: true,
    refetchInterval: (query) =>
      activityPollInterval(
        query.state.data?.rows.some((row) => row.overview.state === "running") ?? false,
        5000,
      ),
  })
}
