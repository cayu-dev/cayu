import { useInfiniteQuery } from "@tanstack/react-query"
import { useEffect, useRef, useState } from "react"
import { ApiClientError, inspectSessionMessages, withdrawSessionMessage } from "@/lib/api"
import type {
  SessionMessageActionBody,
  SessionMessageCursor,
  SessionMessageInspectionRecord,
} from "@/lib/generated/server-api"
import { createMutationId } from "@/lib/mutation-transport"
import { type MessageAcceptance, SessionMessageAcceptance } from "@/lib/session-message-queue"

export function useSessionMessageQueue(sessionId: string, enabled: boolean) {
  const [busy, setBusy] = useState(false)
  const [acceptance, setAcceptance] = useState<MessageAcceptance | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const pending = useRef<{ owner: SessionMessageAcceptance; content: string } | null>(null)
  const abort = useRef<AbortController | null>(null)
  const mounted = useRef(true)
  const withdrawals = useRef(new Map<string, SessionMessageActionBody>())
  const inspection = useInfiniteQuery({
    queryKey: ["session-messages", sessionId],
    initialPageParam: undefined as SessionMessageCursor | undefined,
    queryFn: async ({ pageParam, signal }) => {
      const page = await inspectSessionMessages(sessionId, pageParam, signal)
      if (pageParam && page.session_instance_id !== pageParam.session_instance_id)
        throw new Error("The session changed while reading queued messages. Reload the session.")
      if (page.next_cursor && JSON.stringify(page.next_cursor) === JSON.stringify(pageParam))
        throw new Error("Queued-message pagination did not advance.")
      return page
    },
    getNextPageParam: (page) => page.next_cursor ?? undefined,
    enabled,
    retry: false,
  })
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      abort.current?.abort()
    }
  }, [])
  const unavailable =
    inspection.error instanceof ApiClientError && [401, 403].includes(inspection.error.status)
  const guarded = async <T,>(run: (signal: AbortSignal) => Promise<T>): Promise<T> => {
    const controller = new AbortController()
    abort.current = controller
    setBusy(true)
    setActionError(null)
    const timeout = globalThis.setTimeout(() => controller.abort(), 30_000)
    try {
      return await run(controller.signal)
    } finally {
      globalThis.clearTimeout(timeout)
      if (abort.current === controller) abort.current = null
      if (mounted.current) setBusy(false)
    }
  }
  const submit = async (content?: string): Promise<string | null> => {
    if (abort.current) return null
    if (!pending.current) {
      if (!enabled || !inspection.data || !content?.trim()) return null
      pending.current = {
        owner: new SessionMessageAcceptance(sessionId, content, createMutationId()),
        content,
      }
    }
    const request = pending.current
    return guarded(async (signal) => {
      const result = await request.owner.submit(signal)
      if (!mounted.current) return null
      setAcceptance(result)
      if (result.phase !== "unknown") pending.current = null
      if (result.phase === "accepted") {
        void inspection.refetch()
        return request.content
      }
      return null
    })
  }
  const withdraw = async (record: SessionMessageInspectionRecord, instance: string) => {
    if (abort.current || pending.current) return
    const body =
      withdrawals.current.get(record.queue_id) ??
      Object.freeze({
        session_instance_id: instance,
        expected_revision: record.revision,
        idempotency_key: createMutationId(),
      })
    withdrawals.current.set(record.queue_id, body)
    return guarded(async (signal) => {
      try {
        await withdrawSessionMessage(sessionId, record.queue_id, body, signal)
        withdrawals.current.delete(record.queue_id)
        if (mounted.current) void inspection.refetch()
      } catch {
        if (mounted.current)
          setActionError("Withdrawal was not confirmed. Retry withdrawal using the same request.")
      }
    })
  }
  const first = inspection.data?.pages[0]
  const unreadable =
    inspection.data?.pages.some((page) =>
      (page.records ?? []).some((record) => record.validity !== "valid"),
    ) ?? false
  const rows =
    inspection.data?.pages.flatMap((page) =>
      page.session_instance_id === first?.session_instance_id
        ? (page.records ?? []).filter(
            (record) => record.validity === "valid" && record.status === "queued",
          )
        : [],
    ) ?? []
  return {
    inspection,
    rows,
    instance: first?.session_instance_id,
    unavailable,
    unreadable,
    busy,
    acceptance,
    actionError,
    submit,
    withdraw,
    fenced: busy || acceptance?.phase === "unknown",
  }
}
