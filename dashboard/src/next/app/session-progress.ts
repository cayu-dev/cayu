import { useEffect, useRef, useState } from "react"
import { ApiClientError } from "@/lib/api"
import { followSessionProgress, type SessionProgress } from "@/lib/session-history"
import { useApi } from "./services"

export type { SessionProgress } from "@/lib/session-history"

/** Mount once in Session; Trace and Conversation share the resulting invalidations. */
export function useSessionProgress({
  sessionId,
  enabled,
  active,
  mutationActive,
  onProgress,
}: {
  sessionId: string
  enabled: boolean
  active: boolean
  mutationActive: boolean
  onProgress: (progress: SessionProgress) => void | Promise<void>
}) {
  const api = useApi()
  const callback = useRef(onProgress)
  callback.current = onProgress
  const [error, setError] = useState<unknown>(null)
  const [revision, setRevision] = useState(0)
  const [visible, setVisible] = useState(() =>
    typeof document === "undefined" ? true : document.visibilityState !== "hidden",
  )
  useEffect(() => {
    const update = () => setVisible(document.visibilityState !== "hidden")
    document.addEventListener("visibilitychange", update)
    update()
    return () => document.removeEventListener("visibilitychange", update)
  }, [])
  // biome-ignore lint/correctness/useExhaustiveDependencies: revision explicitly restarts observation after retry.
  useEffect(() => {
    if (!enabled || !visible) return
    const controller = new AbortController()
    let retryTimer: ReturnType<typeof setTimeout> | undefined
    setError(null)
    void followSessionProgress({
      signal: controller.signal,
      active: () => active || mutationActive,
      read: (query, signal) => api.events(sessionId, query, signal),
      onProgress: async (progress) => {
        if (!controller.signal.aborted) await callback.current(progress)
      },
    }).catch((failure: unknown) => {
      if (controller.signal.aborted) return
      setError(failure)
      if (
        !(failure instanceof ApiClientError && [401, 403, 404, 409, 422].includes(failure.status))
      ) {
        retryTimer = setTimeout(() => setRevision((value) => value + 1), 5000)
      }
    })
    return () => {
      controller.abort()
      clearTimeout(retryTimer)
    }
  }, [api, sessionId, enabled, active, mutationActive, revision, visible])
  return { error, retry: () => setRevision((value) => value + 1) }
}
