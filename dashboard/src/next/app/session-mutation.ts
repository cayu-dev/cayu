import { useEffect, useRef, useState } from "react"
import type { MutationExecutionOptions } from "@/lib/mutation-browser"
import type {
  MutationTransportController,
  MutationTransportSnapshot,
} from "@/lib/mutation-transport"

export type SessionMutationKind = "reply" | "interrupt" | "recovery" | "provider" | "rerun"
const settled = (snapshot: MutationTransportSnapshot | undefined) =>
  !snapshot || ["terminal", "request_failed", "runtime_failed"].includes(snapshot.phase)

/** React lifetime and single-action admission around the existing transport owner. */
export function useSessionMutation() {
  const observer = useRef<AbortController | null>(null)
  const transport = useRef<MutationTransportController | null>(null)
  const mounted = useRef(true)
  const [snapshot, setSnapshot] = useState<MutationTransportSnapshot | null>(null)
  const [kind, setKind] = useState<SessionMutationKind | null>(null)
  const [busy, setBusy] = useState(false)
  const [fenced, setFenced] = useState(false)
  const [error, setError] = useState<unknown>(null)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      observer.current?.abort()
      transport.current?.dispose()
      observer.current = null
      transport.current = null
    }
  }, [])
  const releaseSettled = (next: MutationTransportSnapshot | undefined) => {
    if (!settled(next)) return
    transport.current?.dispose()
    transport.current = null
    setFenced(false)
  }
  const currentSnapshot = () => transport.current?.snapshot()
  const execute = async (
    action: SessionMutationKind,
    run: (options: MutationExecutionOptions) => Promise<MutationTransportSnapshot | undefined>,
  ): Promise<MutationTransportSnapshot | null | undefined> => {
    if (observer.current || transport.current) {
      throw new Error("Resolve the previous action's outcome before starting another action.")
    }
    const abort = new AbortController()
    observer.current = abort
    setKind(action)
    setBusy(true)
    setFenced(true)
    setSnapshot(null)
    setError(null)
    let owner: MutationTransportController | null = null
    try {
      const result = await run({
        signal: abort.signal,
        onController: (next) => {
          owner = next
          if (mounted.current && observer.current === abort) transport.current = next
          else next.dispose()
        },
        onChange: (next) => {
          if (mounted.current && !abort.signal.aborted && transport.current === owner)
            setSnapshot(next)
        },
      })
      if (!mounted.current || observer.current !== abort || abort.signal.aborted) return null
      setSnapshot(result ?? null)
      releaseSettled(result)
      return result
    } catch (failure) {
      if (!mounted.current || observer.current !== abort || abort.signal.aborted) return null
      setError(failure)
      // A pre-dispatch failure has no owner. Once dispatched, retain its authority.
      const current = currentSnapshot()
      if (current) setSnapshot(current)
      releaseSettled(current)
      throw failure
    } finally {
      if (mounted.current && observer.current === abort) {
        observer.current = null
        setBusy(false)
      }
    }
  }
  const reobserve = async () => {
    const owner = transport.current
    if (!owner || observer.current) return null
    const abort = new AbortController()
    observer.current = abort
    setBusy(true)
    setError(null)
    try {
      const result = await owner.reobserve(abort.signal)
      if (!mounted.current || observer.current !== abort || abort.signal.aborted) return null
      setSnapshot(result)
      releaseSettled(result)
      return result
    } catch (failure) {
      if (mounted.current && observer.current === abort && !abort.signal.aborted) setError(failure)
      return null
    } finally {
      if (mounted.current && observer.current === abort) {
        observer.current = null
        setBusy(false)
      }
    }
  }
  return { execute, reobserve, snapshot, kind, busy, fenced, error }
}
