import { useQuery, useQueryClient } from "@tanstack/react-query"
import { useEffect, useRef, useState } from "react"
import type { SessionTracePage } from "@/lib/api"
import { useApi } from "./services"
import { mergeTracePages } from "./trace-runtime"

/** Historical pages keep a fixed cutoff; live deltas use a separate forward cursor. */
export function useTraceHistory(sessionId: string, enabled: boolean) {
  const api = useApi()
  const client = useQueryClient()
  const current = useRef<{
    id: string
    page?: SessionTracePage
    after: number
    through: number
    more: boolean
    live: number
  }>({ id: sessionId, after: 0, through: 0, more: false, live: 0 })
  if (current.current.id !== sessionId)
    current.current = { id: sessionId, after: 0, through: 0, more: false, live: 0 }
  const [loadingMore, setLoadingMore] = useState(false)
  const [pageError, setPageError] = useState<unknown>(null)
  const olderAbort = useRef<AbortController | null>(null)
  // biome-ignore lint/correctness/useExhaustiveDependencies: changing session must abort its historical read.
  useEffect(() => () => olderAbort.current?.abort(), [sessionId])
  const query = useQuery({
    queryKey: ["trace", sessionId],
    enabled,
    queryFn: async ({ signal }) => {
      if (!api.trace) throw new Error("This server does not provide exact session traces.")
      const owner = current.current
      let incoming = await api.trace(
        sessionId,
        {
          after_sequence: owner.page ? owner.live : 0,
          limit: 1000,
          session_instance_id: owner.page?.session_instance_id,
        },
        signal,
      )
      if (signal.aborted || current.current !== owner)
        throw new DOMException("Aborted", "AbortError")
      if (!owner.page) {
        owner.after = incoming.next_sequence
        owner.through = incoming.through_sequence
        owner.more = incoming.has_more
        owner.live = incoming.through_sequence
        owner.page = incoming
      } else {
        // Drain in bounded server pages without dropping a fast producer's backlog.
        while (true) {
          owner.page = mergeTracePages(owner.page, incoming)
          owner.live = Math.max(owner.live, incoming.next_sequence)
          if (!incoming.has_more) break
          incoming = await api.trace(
            sessionId,
            {
              after_sequence: owner.live,
              through_sequence: incoming.through_sequence,
              session_instance_id: owner.page.session_instance_id,
              limit: 1000,
            },
            signal,
          )
          if (signal.aborted || current.current !== owner)
            throw new DOMException("Aborted", "AbortError")
        }
      }
      return { page: owner.page, hasMore: owner.more }
    },
    retry: false,
  })
  const loadMore = async () => {
    if (loadingMore || !api.trace) return
    const owner = current.current
    if (!owner.page || !owner.more) return
    olderAbort.current?.abort()
    const controller = new AbortController()
    olderAbort.current = controller
    setLoadingMore(true)
    setPageError(null)
    try {
      const incoming = await api.trace(
        sessionId,
        {
          after_sequence: owner.after,
          through_sequence: owner.through,
          session_instance_id: owner.page.session_instance_id,
          limit: 1000,
        },
        controller.signal,
      )
      if (controller.signal.aborted || current.current !== owner) return
      owner.page = mergeTracePages(owner.page, incoming)
      owner.after = incoming.next_sequence
      owner.more = incoming.has_more
      client.setQueryData(["trace", sessionId], { page: owner.page, hasMore: owner.more })
    } catch (error) {
      if (!controller.signal.aborted) setPageError(error)
    } finally {
      if (olderAbort.current === controller) setLoadingMore(false)
    }
  }
  return { ...query, loadMore, loadingMore, pageError }
}
