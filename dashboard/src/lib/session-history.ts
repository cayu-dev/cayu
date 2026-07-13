const ACTIVE_SESSION_STATUSES = new Set(["pending", "running", "interrupting"])
const TERMINAL_STATE_POLL_INTERVAL_MS = 15_000

export type PageNavigation<T> = {
  current: T
  newer: T[]
}

export type TranscriptPageParam = { offset: number; limit: number } | null

export type SummaryRevisionState = {
  status: string
  lastActivityAt: string
  interruptionCascade: "none" | "pending" | "failed"
}

export type StatePollInput = {
  status: string | undefined
  interruptionCascade: "none" | "pending" | "failed" | undefined
  interruptRequested: boolean
  mutationActive: boolean
  hasError: boolean
  failureCount: number
  nonRetryableError: boolean
}

type SequencedRecord = {
  id: string
  sequence: number
}

type EventScanPage = {
  events: readonly Pick<SequencedRecord, "sequence">[]
  scan_through_sequence: number | null
}

type IndexedRecord = {
  index: number
}

type TranscriptTailPage = {
  has_more: boolean
  total_messages: number
}

export type BoundedWindow<T> = {
  records: T[]
  dropped: boolean
}

export type TailActivityAction = "ignore" | "defer" | "reconcile"

export type PendingTailInput = {
  pending: boolean
  atLatest: boolean
  pinned: boolean
  visible: boolean
  mutationActive: boolean
  queryFetching: boolean
  hasError: boolean
}

/** Return true to drain one request queued while the task was active. */
export type CoalescedAsyncTask = () => Promise<boolean>

/** Serializes async work and retains at most one follow-up request. */
export class CoalescedAsyncRunner {
  private activePromise: Promise<void> | null = null
  private pending = false
  private task: CoalescedAsyncTask | null = null

  request(task: CoalescedAsyncTask): Promise<void> {
    this.task = task
    this.pending = true
    if (this.activePromise === null) {
      this.activePromise = this.drain()
    }
    return this.activePromise
  }

  clearPending(): void {
    this.pending = false
  }

  get active(): boolean {
    return this.activePromise !== null
  }

  private async drain(): Promise<void> {
    try {
      while (this.pending) {
        this.pending = false
        const task = this.task
        if (task === null) return
        const continueDraining = await task()
        if (!continueDraining) this.pending = false
      }
    } catch (error) {
      this.pending = false
      throw error
    } finally {
      // Clear synchronously in the drain continuation, before the returned
      // promise settles, so a new request can never attach to a completed run.
      this.activePromise = null
    }
  }
}

export function initialPageNavigation<T>(current: T): PageNavigation<T> {
  return { current, newer: [] }
}

export function navigateToOlderPage<T>(navigation: PageNavigation<T>, next: T): PageNavigation<T> {
  return {
    current: next,
    newer: [...navigation.newer, navigation.current],
  }
}

export function navigateToNewerPage<T>(navigation: PageNavigation<T>): PageNavigation<T> {
  const next = navigation.newer.at(-1)
  if (next === undefined) return navigation
  return {
    current: next,
    newer: navigation.newer.slice(0, -1),
  }
}

export function jumpToLatestPage<T>(current: T): PageNavigation<T> {
  return initialPageNavigation(current)
}

export function olderTranscriptPage(
  currentOffset: number,
  pageSize: number,
): TranscriptPageParam | undefined {
  if (currentOffset <= 0) return undefined
  const offset = Math.max(0, currentOffset - pageSize)
  return { offset, limit: currentOffset - offset }
}

export function sessionSummaryRevision(state: SummaryRevisionState): string {
  if (ACTIVE_SESSION_STATUSES.has(state.status) || state.interruptionCascade === "pending") {
    return `${state.status}:${state.interruptionCascade}:snapshot`
  }
  return `${state.status}:${state.interruptionCascade}:${state.lastActivityAt}`
}

export function summaryStateIsStable(state: SummaryRevisionState): boolean {
  return !ACTIVE_SESSION_STATUSES.has(state.status) && state.interruptionCascade !== "pending"
}

export function statePollInterval(input: StatePollInput): number | false {
  if (input.nonRetryableError) return false
  if (input.hasError) {
    return Math.min(30_000, 2000 * 2 ** Math.min(input.failureCount, 4))
  }
  if (input.interruptRequested || input.interruptionCascade === "pending") return 1000
  if (input.mutationActive) return 2000
  return input.status === undefined || ACTIVE_SESSION_STATUSES.has(input.status)
    ? 2000
    : TERMINAL_STATE_POLL_INTERVAL_MS
}

export function queryReadErrorIsFatal(isError: boolean, hasData: boolean): boolean {
  return isError && !hasData
}

export function sessionMetadataNeedsRefresh(
  detailUpdatedAt: string | undefined,
  stateUpdatedAt: string | undefined,
): boolean {
  return (
    detailUpdatedAt !== undefined &&
    stateUpdatedAt !== undefined &&
    detailUpdatedAt !== stateUpdatedAt
  )
}

export function mergeLatestEventWindow<T extends SequencedRecord>(
  current: readonly T[],
  incoming: readonly T[],
  limit: number,
): BoundedWindow<T> {
  const byId = new Map<string, T>()
  for (const event of current) byId.set(event.id, event)
  for (const event of incoming) byId.set(event.id, event)
  const merged = [...byId.values()].sort((left, right) => right.sequence - left.sequence)
  return {
    records: merged.slice(0, limit),
    dropped: merged.length > limit,
  }
}

export function eventScanCursor(page: EventScanPage, previous = 0): number {
  let cursor = Math.max(previous, page.scan_through_sequence ?? 0)
  for (const event of page.events) cursor = Math.max(cursor, event.sequence)
  return cursor
}

export function mergeStoredAndLiveRecords<T extends { id: string }>(
  stored: readonly T[],
  live: readonly T[],
): T[] {
  const storedIds = new Set(stored.map((record) => record.id))
  return [...stored, ...live.filter((record) => !storedIds.has(record.id))]
}

export function mergeLatestTranscriptWindow<T extends IndexedRecord>(
  current: readonly T[],
  incoming: readonly T[],
  limit: number,
): BoundedWindow<T> {
  const byIndex = new Map<number, T>()
  for (const message of current) byIndex.set(message.index, message)
  for (const message of incoming) byIndex.set(message.index, message)
  const merged = [...byIndex.values()].sort((left, right) => left.index - right.index)
  const dropped = merged.length > limit
  return {
    records: dropped ? merged.slice(-limit) : merged,
    dropped,
  }
}

export async function loadStableTranscriptTailPage<T extends TranscriptTailPage>(
  totalMessages: number,
  pageSize: number,
  readLimit: number,
  loadPage: (offset: number) => Promise<T>,
): Promise<T> {
  let offset = Math.max(0, totalMessages - pageSize)
  for (let attempt = 0; attempt < readLimit; attempt += 1) {
    const page = await loadPage(offset)
    if (!page.has_more) return page

    // The append-only transcript grew after its total was observed. Move the
    // bounded window to the new tail, but never chase active writes forever.
    offset = Math.max(0, page.total_messages - pageSize)
  }
  throw new Error(
    "The transcript is growing too quickly to capture a stable latest page. Retry shortly.",
  )
}

export function tailActivityAction({
  atLatest,
  pinned,
  visible,
}: {
  atLatest: boolean
  pinned: boolean
  visible: boolean
}): TailActivityAction {
  if (!atLatest) return "ignore"
  return pinned && visible ? "reconcile" : "defer"
}

export function pendingTailCanReconcile(input: PendingTailInput): boolean {
  return (
    input.pending &&
    input.atLatest &&
    input.pinned &&
    input.visible &&
    !input.mutationActive &&
    !input.queryFetching &&
    !input.hasError
  )
}

export function historyModeLabel({
  atLatest,
  active,
  pinned,
}: {
  atLatest: boolean
  active: boolean
  pinned: boolean
}): "History" | "Latest" | "Live tail" | "Paused" {
  if (!atLatest) return "History"
  if (!pinned) return "Paused"
  return active ? "Live tail" : "Latest"
}

export function durationEndTimestamp({
  terminalEventAt,
  lastActivityAt,
  updatedAt,
}: {
  terminalEventAt: string | null | undefined
  lastActivityAt: string | null | undefined
  updatedAt: string | null | undefined
}): string | null {
  return terminalEventAt ?? lastActivityAt ?? updatedAt ?? null
}

export function durationDetail(status: string, usesTerminalEvent: boolean): string {
  if (!usesTerminalEvent) return "started to last activity"
  if (status === "completed") return "started to completion"
  if (status === "failed") return "started to failure"
  if (status === "interrupted") return "started to interruption"
  return "started to final activity"
}
