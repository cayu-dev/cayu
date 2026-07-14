import type { ApiEventRecord, SseEventEnvelope } from "./generated/server-api"

export const MUTATION_LIVE_EVENT_LIMIT = 100
export const MUTATION_LIVE_EVENT_MAX_BYTES = 2 * 1024 * 1024
export const MUTATION_RECENT_EVENT_ID_LIMIT = 2_048
export const MUTATION_RECENT_EVENT_ID_MAX_BYTES = 2 * 1024 * 1024
export const MUTATION_OUT_OF_ORDER_RECORD_LIMIT = 100
export const MUTATION_LIVE_TEXT_LIMIT = 262_144

const TERMINAL_EVENT_TYPES = new Set(["session.completed", "session.failed", "session.interrupted"])
const TEXT_ENCODER = new TextEncoder()

type AccumulatedEvent = {
  event: SseEventEnvelope
  sequence: number | null
  arrival: number
  bytes: number
}

type TerminalBoundary = {
  id: string
  sequence: number | null
  arrival: number
}

type RecentEventId = {
  id: string
  bytes: number
}

export type MutationEventAccumulatorSnapshot = {
  events: SseEventEnvelope[]
  eventsTruncated: boolean
  liveText: string
  liveTextTruncated: boolean
  latestSequence: number
}

export type MutationReconciliationResult = {
  observedBoundaryFound: boolean
}

function eventWithoutSequence(record: ApiEventRecord): SseEventEnvelope {
  const { sequence: _sequence, ...event } = record
  return event
}

function serializedBytes(event: SseEventEnvelope): number {
  return TEXT_ENCODER.encode(JSON.stringify(event)).byteLength
}

export class MutationEventAccumulator {
  private readonly eventsById = new Map<string, AccumulatedEvent>()
  private readonly recentEventIds = new Set<string>()
  private readonly recentEventIdOrder: RecentEventId[] = []
  private recentEventIdHead = 0
  private recentEventIdBytes = 0
  private readonly terminalBoundaries = new Map<string, TerminalBoundary>()
  private arrival = 0
  private eventBytes = 0
  private eventWindowTruncated = false
  private text = ""
  private textWindowTruncated = false
  private durableSequence = 0

  addSseEvent(event: SseEventEnvelope): boolean {
    return this.addEvent(event, null)
  }

  reconcile(
    records: readonly ApiEventRecord[],
    observedThroughEventId: string | null = null,
  ): MutationReconciliationResult {
    if (records.length > MUTATION_OUT_OF_ORDER_RECORD_LIMIT) {
      throw new RangeError(
        `Mutation reconciliation exceeded ${MUTATION_OUT_OF_ORDER_RECORD_LIMIT} records.`,
      )
    }
    const ordered = [...records].sort((left, right) => left.sequence - right.sequence)
    const observedBoundary =
      observedThroughEventId === null
        ? undefined
        : ordered.find((record) => record.id === observedThroughEventId)
    for (const record of ordered) {
      const wasObserved =
        observedThroughEventId !== null &&
        (observedBoundary === undefined || record.sequence <= observedBoundary.sequence)
      this.addEvent(eventWithoutSequence(record), record.sequence, wasObserved)
      this.durableSequence = Math.max(this.durableSequence, record.sequence)
    }
    return { observedBoundaryFound: observedBoundary !== undefined }
  }

  advanceDurableSequence(sequence: number | null | undefined): void {
    if (sequence === null || sequence === undefined) return
    if (!Number.isSafeInteger(sequence) || sequence < 0) {
      throw new RangeError("Mutation durable sequence must be a non-negative safe integer.")
    }
    this.durableSequence = Math.max(this.durableSequence, sequence)
  }

  snapshot(): MutationEventAccumulatorSnapshot {
    return {
      events: [...this.eventsById.values()]
        .sort((left, right) => {
          if (left.sequence !== null && right.sequence !== null) {
            return left.sequence - right.sequence
          }
          if (left.sequence !== null) return -1
          if (right.sequence !== null) return 1
          return left.arrival - right.arrival
        })
        .map(({ event }) => event),
      eventsTruncated: this.eventWindowTruncated,
      liveText: this.text,
      liveTextTruncated: this.textWindowTruncated,
      latestSequence: this.durableSequence,
    }
  }

  private addEvent(event: SseEventEnvelope, sequence: number | null, wasObserved = false): boolean {
    const existing = this.eventsById.get(event.id)
    if (existing !== undefined) {
      if (sequence !== null) {
        const bytes = serializedBytes(event)
        this.eventBytes += bytes - existing.bytes
        existing.sequence = sequence
        existing.event = event
        existing.bytes = bytes
        this.enforceEventWindow()
      }
      this.recordTerminalBoundary(event, sequence, existing.arrival)
      return false
    }
    if (this.recentEventIds.has(event.id)) return false

    const arrival = this.arrival
    this.arrival += 1
    this.recordTerminalBoundary(event, sequence, arrival)
    if (wasObserved) return false

    this.rememberEventId(event.id)

    if (event.type === "model.text.delta") {
      const delta = event.payload.delta
      if (typeof delta === "string" && delta.length > 0) this.appendText(delta)
      return true
    }

    const bytes = serializedBytes(event)
    this.eventsById.set(event.id, { event, sequence, arrival, bytes })
    this.eventBytes += bytes
    this.enforceEventWindow()
    return true
  }

  private rememberEventId(eventId: string): void {
    const bytes = TEXT_ENCODER.encode(eventId).byteLength
    this.recentEventIds.add(eventId)
    this.recentEventIdOrder.push({ id: eventId, bytes })
    this.recentEventIdBytes += bytes
    while (
      this.recentEventIdOrder.length - this.recentEventIdHead > MUTATION_RECENT_EVENT_ID_LIMIT ||
      this.recentEventIdBytes > MUTATION_RECENT_EVENT_ID_MAX_BYTES
    ) {
      const removed = this.recentEventIdOrder[this.recentEventIdHead]
      this.recentEventIdHead += 1
      if (removed !== undefined) {
        this.recentEventIds.delete(removed.id)
        this.recentEventIdBytes -= removed.bytes
      }
    }
    if (this.recentEventIdHead >= MUTATION_RECENT_EVENT_ID_LIMIT) {
      this.recentEventIdOrder.splice(0, this.recentEventIdHead)
      this.recentEventIdHead = 0
    }
  }

  private appendText(delta: string): void {
    this.text += delta
    if (this.text.length <= MUTATION_LIVE_TEXT_LIMIT) return
    this.text = this.text.slice(-MUTATION_LIVE_TEXT_LIMIT)
    this.textWindowTruncated = true
  }

  private recordTerminalBoundary(
    event: SseEventEnvelope,
    sequence: number | null,
    arrival: number,
  ): void {
    if (!TERMINAL_EVENT_TYPES.has(event.type)) return
    const current = this.terminalBoundaries.get(event.type)
    if (
      current === undefined ||
      (sequence !== null && (current.sequence === null || sequence > current.sequence)) ||
      (sequence === null && current.sequence === null && arrival > current.arrival)
    ) {
      this.terminalBoundaries.set(event.type, { id: event.id, sequence, arrival })
    }
  }

  private enforceEventWindow(): void {
    while (
      this.eventsById.size > MUTATION_LIVE_EVENT_LIMIT ||
      this.eventBytes > MUTATION_LIVE_EVENT_MAX_BYTES
    ) {
      const entries = [...this.eventsById.values()].sort(
        (left, right) => left.arrival - right.arrival,
      )
      const latestTerminalIds = new Set(
        [...this.terminalBoundaries.values()].map((boundary) => boundary.id),
      )
      const removable =
        entries.find(({ event }) => !latestTerminalIds.has(event.id)) ?? entries.at(0)
      if (removable === undefined) break
      this.eventsById.delete(removable.event.id)
      this.eventBytes -= removable.bytes
      this.eventWindowTruncated = true
    }
  }
}
