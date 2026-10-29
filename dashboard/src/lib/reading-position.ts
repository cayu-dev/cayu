/**
 * Where a person was reading in a scrolling view, remembered in this browser tab so
 * returning to the view puts them back there. `view` names what was on screen (a
 * session's transcript filters, say); a position is only restored into the same view.
 * `top` is measured from an anchor inside the view (the conversation itself, below its
 * header), so content above the anchor that arrives late doesn't move the reader.
 * Storage can be missing or blocked; positions then aren't remembered.
 */
export const READING_POSITION_PREFIX = "cayu-dashboard-reading:"
/** How close to the bottom still counts as following the end of the view. */
export const READING_END_SLACK = 32

export type ReadingPosition = {
  view: string
  top: number
  atEnd: boolean
  transcriptOffset?: number
}
type PositionStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">
type ScrollMetrics = { scrollTop: number; scrollHeight: number; clientHeight: number }

function tabStorage(): PositionStorage | null {
  try {
    return window.sessionStorage
  } catch {
    return null
  }
}

export function readingPositionFromScroll(
  metrics: ScrollMetrics,
  view: string,
  anchorTop = 0,
): ReadingPosition {
  const top = Math.round(metrics.scrollTop - anchorTop)
  return {
    view,
    top,
    atEnd: metrics.scrollHeight - metrics.scrollTop - metrics.clientHeight <= READING_END_SLACK,
  }
}

export function readReadingPosition(
  key: string,
  storage: PositionStorage | null = tabStorage(),
): ReadingPosition | null {
  try {
    const raw = storage?.getItem(READING_POSITION_PREFIX + key)
    if (!raw) return null
    const value: unknown = JSON.parse(raw)
    if (
      value === null ||
      typeof value !== "object" ||
      !("view" in value) ||
      typeof value.view !== "string" ||
      !("top" in value) ||
      typeof value.top !== "number" ||
      !Number.isFinite(value.top) ||
      !("atEnd" in value) ||
      typeof value.atEnd !== "boolean"
    )
      return null
    const transcriptOffset = "transcriptOffset" in value ? value.transcriptOffset : undefined
    if (
      transcriptOffset !== undefined &&
      (typeof transcriptOffset !== "number" ||
        !Number.isSafeInteger(transcriptOffset) ||
        transcriptOffset < 0)
    )
      return null
    return {
      view: value.view,
      top: value.top,
      atEnd: value.atEnd,
      ...(transcriptOffset === undefined ? {} : { transcriptOffset }),
    }
  } catch {
    return null
  }
}

/** Remember a position, or forget the key's position when `position` is null. */
export function writeReadingPosition(
  key: string,
  position: ReadingPosition | null,
  storage: PositionStorage | null = tabStorage(),
): void {
  try {
    if (position) storage?.setItem(READING_POSITION_PREFIX + key, JSON.stringify(position))
    else storage?.removeItem(READING_POSITION_PREFIX + key)
  } catch {
    // Storage can be unavailable or full; the position then isn't remembered.
  }
}

/** Where `anchor` starts within `element`'s scrolling content; 0 without an anchor. */
export function anchorOffset(element: HTMLElement, anchor: Element | null | undefined): number {
  if (!anchor) return 0
  return (
    anchor.getBoundingClientRect().top - element.getBoundingClientRect().top + element.scrollTop
  )
}

/**
 * Scroll `element` back to `top` past `anchor` once its content is tall enough. Content
 * often arrives after the view mounts, so this retries as `content` resizes, and stops when the
 * position is reached, the person scrolls themselves, or `timeoutMs` passes. Calls
 * `onDone` once either way and returns a function that cancels it.
 */
export function restoreScrollPosition(
  element: HTMLElement,
  content: Element,
  top: number,
  onDone: () => void,
  { anchor, timeoutMs = 3000 }: { anchor?: () => Element | null; timeoutMs?: number } = {},
): () => void {
  let finished = false
  const finish = () => {
    if (finished) return
    finished = true
    observer?.disconnect()
    clearTimeout(timer)
    for (const event of USER_SCROLL_EVENTS) element.removeEventListener(event, finish)
    onDone()
  }
  const attempt = () => {
    if (finished) return
    const target = Math.max(0, top + anchorOffset(element, anchor?.()))
    element.scrollTop = target
    if (Math.abs(element.scrollTop - target) <= 1) finish()
  }
  const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(attempt)
  const timer = setTimeout(finish, timeoutMs)
  for (const event of USER_SCROLL_EVENTS) element.addEventListener(event, finish, { passive: true })
  observer?.observe(content)
  attempt()
  return finish
}

const USER_SCROLL_EVENTS = ["wheel", "touchstart", "keydown", "pointerdown"] as const
