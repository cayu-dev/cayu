/** Bounded, read-only observation. Stopping never claims native quiescence. */
export function startBrowserPageObserver<T>(
  read: () => Promise<T>,
  publish: (value: T) => void,
  unavailable: () => void,
  canRead: () => boolean,
) {
  let stopped = false
  let timer: ReturnType<typeof setTimeout> | undefined
  const expires = performance.now() + 300_000

  async function sample() {
    timer = undefined
    if (stopped) return
    if (performance.now() >= expires) {
      stopped = true
      unavailable()
      return
    }
    if (canRead()) {
      try {
        // The existing control client owns the request's ten-second deadline.
        // Never overlap reads or replay a failed read under newer authority.
        const result = await read()
        if (!stopped && performance.now() >= expires) {
          stopped = true
          unavailable()
        } else if (!stopped && canRead()) publish(result)
      } catch {
        if (!stopped) {
          stopped = true
          unavailable()
        }
      }
    }
    if (!stopped) timer = setTimeout(sample, 2_000)
  }

  timer = setTimeout(sample, 2_000)
  return {
    stop() {
      stopped = true
      clearTimeout(timer)
      // An already dispatched census retains its existing request owner until
      // settlement/deadline. Its late result cannot publish through this owner.
    },
  }
}
