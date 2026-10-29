import { hashKey, type QueryClient, type QueryKey } from "@tanstack/react-query"

/** A denial is cached without the protected value, so later transient errors stay empty. */
export type UsageRead<T> = { value: T; denied: null } | { value: null; denied: Error }

export async function readUsage<T>(
  client: QueryClient,
  key: QueryKey,
  signal: AbortSignal,
  read: () => Promise<T>,
): Promise<UsageRead<T>> {
  try {
    const value = await read()
    signal.throwIfAborted()
    return { value, denied: null }
  } catch (error) {
    signal.throwIfAborted()
    if (
      !(error instanceof Error) ||
      !("status" in error) ||
      (error.status !== 401 && error.status !== 403)
    )
      throw error
    const denied: UsageRead<never> = { value: null, denied: error }
    const report = (query: { queryKey: QueryKey }) =>
      query.queryKey[0] === "usage" &&
      (query.queryKey[1] === "rollup" || query.queryKey[1] === "counts")
    // Cancellation synchronously fences TanStack's result publication, even when the
    // underlying read cannot abort. Do not cancel this query before it publishes denial.
    const currentKeyHash = hashKey(key)
    void client.cancelQueries(
      { predicate: (query) => report(query) && hashKey(query.queryKey) !== currentKeyHash },
      { revert: false },
    )
    client.setQueriesData({ predicate: report }, () => denied)
    return denied
  }
}
