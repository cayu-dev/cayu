import {
  hashKey,
  type QueryClient,
  type QueryKey,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query"
import { type ArtifactReadQuery, type ArtifactsQuery, fetchArtifact, fetchArtifacts } from "./api"

type ArtifactResult<T> = { value: T; denied: null } | { value: null; denied: Error }

/** Lists and exact previews share authority; a denial discards both cached surfaces. */
export async function readArtifactList<T>(
  client: QueryClient,
  key: QueryKey,
  signal: AbortSignal,
  read: () => Promise<T>,
): Promise<ArtifactResult<T>> {
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
    const denied: ArtifactResult<never> = { value: null, denied: error }
    const inventory = (query: { queryKey: QueryKey }) =>
      query.queryKey[0] === "artifact-list" || query.queryKey[0] === "artifact-preview"
    const current = hashKey(key)
    void client.cancelQueries(
      { predicate: (query) => inventory(query) && hashKey(query.queryKey) !== current },
      { revert: false },
    )
    client.setQueriesData({ predicate: inventory }, () => denied)
    return denied
  }
}

export function useArtifacts(query: ArtifactsQuery, enabled = true) {
  const client = useQueryClient()
  const key = ["artifact-list", query] as const
  const result = useQuery({
    queryKey: key,
    enabled,
    queryFn: ({ signal }) =>
      readArtifactList(client, key, signal, () => fetchArtifacts(query, signal)),
  })
  const data = enabled ? (result.data?.value ?? undefined) : undefined
  return {
    ...result,
    data,
    error: result.data?.denied ?? result.error,
    isSuccess: result.isSuccess && data !== undefined,
  }
}

export function useArtifact(artifactId: string, query: ArtifactReadQuery, enabled = true) {
  const client = useQueryClient()
  const key = ["artifact-preview", artifactId, query] as const
  const result = useQuery({
    queryKey: key,
    enabled,
    queryFn: ({ signal }) =>
      readArtifactList(client, key, signal, () => fetchArtifact(artifactId, query, signal)),
  })
  return {
    ...result,
    data: enabled ? (result.data?.value ?? undefined) : undefined,
    error: result.data?.denied ?? result.error,
  }
}
