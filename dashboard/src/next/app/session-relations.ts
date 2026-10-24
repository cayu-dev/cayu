import { useInfiniteQuery, useQuery } from "@tanstack/react-query"
import { overviewForDisplay } from "../api/overview"
import type { SessionListQuery } from "../api/types"
import type { SessionRow } from "./queries"
import { useApi } from "./services"
import { loadSessionList } from "./session-list-loader"

/** Parent identity and direct children are queried explicitly, never from a capped global list. */
export function useSessionRelations(id: string, parentId: string | null | undefined) {
  const api = useApi()
  const read = (query: SessionListQuery, signal: AbortSignal) =>
    loadSessionList(
      query,
      (selection, signal) => api.sessions(selection, signal),
      (sessionId, selection, signal) => api.transcript(sessionId, selection, signal),
      signal,
    )
  const parent = useQuery({
    queryKey: ["session-parent", id, parentId],
    enabled: !!parentId,
    queryFn: async ({ signal }): Promise<SessionRow | null> => {
      if (!parentId) return null
      const result = await read({ ids: [parentId], include: ["overview"] }, signal)
      const row = result.rows[0]
      return row ? { ...row, overview: overviewForDisplay(row.overview) } : null
    },
  })
  const children = useInfiniteQuery({
    queryKey: ["session-children", id],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) =>
      read(
        {
          parent_session_id: id,
          include: ["overview"],
          include_forks: false,
          limit: 100,
          cursor: pageParam,
        },
        signal,
      ),
    getNextPageParam: (result) => result.page.next_cursor ?? undefined,
  })
  const rows: SessionRow[] = (children.data?.pages ?? []).flatMap((page) =>
    page.rows.map((row) => ({ ...row, overview: overviewForDisplay(row.overview) })),
  )
  return { parent, children, rows }
}
