import { useQuery, useQueryClient } from "@tanstack/react-query"
import { ApiClientError, fetchEnvironments } from "./api"

const ENVIRONMENTS_QUERY_KEY = ["environments"] as const

/** One inventory cache for pages, filters and scenario launch settings. */
export function useEnvironments(enabled = true) {
  const client = useQueryClient()
  return useQuery({
    queryKey: ENVIRONMENTS_QUERY_KEY,
    enabled,
    staleTime: 15_000,
    retry: (count, error) => count < 3 && !readDenied(error),
    queryFn: async () => {
      try {
        return await fetchEnvironments()
      } catch (error) {
        if (readDenied(error)) {
          // Clearing the shared cache is permanent until a successful authorized read.
          // A later network/server failure must not revive the denied inventory.
          client.getQueryCache().find({ queryKey: ENVIRONMENTS_QUERY_KEY, exact: true })?.setState({
            data: undefined,
            dataUpdatedAt: 0,
            error,
            status: "error",
          })
        }
        throw error
      }
    },
  })
}

const readDenied = (error: unknown): error is ApiClientError =>
  error instanceof ApiClientError && [401, 403].includes(error.status)
