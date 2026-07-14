import assert from "node:assert/strict"
import test from "node:test"

import {
  MUTATION_LIVE_EVENT_LIMIT,
  MUTATION_LIVE_EVENT_MAX_BYTES,
  MUTATION_LIVE_TEXT_LIMIT,
  MUTATION_OUT_OF_ORDER_RECORD_LIMIT,
  MUTATION_RECENT_EVENT_ID_MAX_BYTES,
  MutationEventAccumulator,
} from "../src/lib/mutation-event-accumulator.ts"
import {
  expectedMutationSseId,
  MUTATION_EVENT_ID_MAX_CHARS,
  MutationStreamProtocolError,
  parseMutationSseError,
  parseMutationSseEvent,
} from "../src/lib/mutation-stream-protocol.ts"
import {
  createMutationId,
  createRunSessionId,
  MutationTransportController,
  mutationReconnectDelay,
  newRunReplayBaseline,
} from "../src/lib/mutation-transport.ts"

const SESSION_ID = "session-test"
const MUTATION_ID = "mutation-test"

function event(id, type = "custom.test", payload = {}) {
  return {
    id,
    type,
    session_id: SESSION_ID,
    agent_name: "assistant",
    environment_name: null,
    workflow_name: null,
    tool_name: null,
    payload,
    timestamp: `2026-07-14T00:00:${String(id.length).padStart(2, "0")}Z`,
  }
}

function record(sequence, id, type = "custom.test", payload = {}) {
  return { ...event(id, type, payload), sequence }
}

function acceptanceEvent(id, acceptedEventId, acceptedEventType = "session.resumed") {
  return event(id, "server.mutation.accepted", {
    mutation_id: MUTATION_ID,
    mutation_kind: "resume",
    accepted_event_id: acceptedEventId,
    accepted_event_type: acceptedEventType,
  })
}

function acceptanceRecord(sequence, id, acceptedEventId, acceptedEventType = "session.resumed") {
  return { ...acceptanceEvent(id, acceptedEventId, acceptedEventType), sequence }
}

function eventsPage(events = [], { hasMore = false, scanThrough = null } = {}) {
  return {
    session_id: SESSION_ID,
    events,
    order_by: "sequence_asc",
    has_more: hasMore,
    next_sequence: events.at(-1)?.sequence ?? null,
    scan_through_sequence: scanThrough,
  }
}

function state(status) {
  return {
    session_id: SESSION_ID,
    status,
    interruption_cascade: "none",
    updated_at: "2026-07-14T00:00:00Z",
    last_activity_at: "2026-07-14T00:00:00Z",
  }
}

function runtimeError() {
  return {
    type: "stream.error",
    kind: "runtime",
    code: "runtime_failed",
    error: "runtime failed",
    error_type: "RuntimeError",
    retryable: false,
    session_id: SESSION_ID,
  }
}

function observerError(retryable) {
  return {
    type: "stream.error",
    kind: "observer",
    code: retryable ? "observer_lagged" : "event_frame_too_large",
    error: "observer failed",
    error_type: "ObserverError",
    retryable,
    session_id: SESSION_ID,
  }
}

function request(overrides = {}) {
  return {
    path: "/resume",
    body: { session_id: SESSION_ID, prompt: "continue" },
    sessionId: SESSION_ID,
    mutationId: MUTATION_ID,
    baseline: { marker: `${SESSION_ID}:event-baseline`, sequence: 1 },
    ...overrides,
  }
}

function controllerOptions(options) {
  return options
}

function transportController(options) {
  return new MutationTransportController(controllerOptions(options))
}

function immediateIo(overrides = {}) {
  return {
    delay: async (_milliseconds, signal) => {
      if (signal.aborted) throw new DOMException("aborted", "AbortError")
    },
    watchdog: (_milliseconds, signal) =>
      new Promise((_resolve, reject) => {
        if (signal.aborted) {
          reject(new DOMException("aborted", "AbortError"))
          return
        }
        signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")), {
          once: true,
        })
      }),
    now: () => 0,
    random: () => 0.5,
    ...overrides,
  }
}

function controlledWatchdogs() {
  const waits = []
  return {
    wait(milliseconds, signal) {
      return new Promise((resolve, reject) => {
        const entry = { milliseconds, signal, resolve, settled: false }
        waits.push(entry)
        if (signal.aborted) {
          entry.settled = true
          reject(new DOMException("aborted", "AbortError"))
          return
        }
        signal.addEventListener(
          "abort",
          () => {
            if (entry.settled) return
            entry.settled = true
            reject(new DOMException("aborted", "AbortError"))
          },
          { once: true },
        )
      })
    },
    expireLatest() {
      const entry = waits.findLast((item) => !item.settled && !item.signal.aborted)
      assert.ok(entry, "expected an armed watchdog")
      entry.settled = true
      entry.resolve()
      return entry
    },
    waits,
  }
}

function visibility(initiallyVisible = true) {
  let visible = initiallyVisible
  const listeners = new Set()
  return {
    source: {
      isVisible: () => visible,
      subscribe(listener) {
        listeners.add(listener)
        return () => listeners.delete(listener)
      },
    },
    set(next) {
      visible = next
      for (const listener of listeners) listener()
    },
    listenerCount: () => listeners.size,
  }
}

async function tick() {
  await new Promise((resolve) => setTimeout(resolve, 0))
}

async function waitFor(predicate, message) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const value = predicate()
    if (value) return value
    await tick()
  }
  assert.fail(message)
}

test("strict stream protocol parsing preserves generated envelopes", () => {
  const value = event("event-1")
  assert.deepEqual(parseMutationSseEvent(JSON.stringify(value)), value)
  assert.equal(expectedMutationSseId(value), `${SESSION_ID}:event-1`)
  assert.deepEqual(parseMutationSseError(JSON.stringify(runtimeError())), runtimeError())

  assert.throws(() => parseMutationSseEvent("not-json"), MutationStreamProtocolError)
  assert.throws(
    () => parseMutationSseEvent(JSON.stringify({ ...value, payload: [] })),
    /Malformed SSE event/,
  )
  assert.throws(
    () => parseMutationSseEvent(JSON.stringify({ ...value, agent_name: 42 })),
    /Malformed SSE event/,
  )
  assert.throws(
    () => parseMutationSseEvent(JSON.stringify({ ...value, id: "event\n1" })),
    /Malformed SSE event/,
  )
  assert.throws(
    () =>
      parseMutationSseEvent(
        JSON.stringify({ ...value, id: "e".repeat(MUTATION_EVENT_ID_MAX_CHARS + 1) }),
      ),
    /Malformed SSE event/,
  )
  assert.throws(
    () => parseMutationSseError(JSON.stringify({ ...runtimeError(), retryable: "no" })),
    /Malformed SSE error/,
  )
  assert.throws(
    () =>
      parseMutationSseError(
        JSON.stringify({ ...runtimeError(), kind: "observer", code: "runtime_failed" }),
      ),
    /Malformed SSE error/,
  )
})

test("mutation accumulator deduplicates, orders, and bounds live state", () => {
  const accumulator = new MutationEventAccumulator()
  for (let index = 0; index < MUTATION_LIVE_EVENT_LIMIT + 5; index += 1) {
    accumulator.addSseEvent(event(`event-${index}`))
  }
  assert.equal(accumulator.snapshot().events.length, MUTATION_LIVE_EVENT_LIMIT)
  assert.equal(accumulator.snapshot().eventsTruncated, true)
  assert.equal(accumulator.addSseEvent(event("event-104")), false)

  const delta = "x".repeat(MUTATION_LIVE_TEXT_LIMIT + 7)
  accumulator.addSseEvent(event("delta-1", "model.text.delta", { delta }))
  assert.equal(accumulator.snapshot().liveText.length, MUTATION_LIVE_TEXT_LIMIT)
  assert.equal(accumulator.snapshot().liveTextTruncated, true)

  const ordered = new MutationEventAccumulator()
  ordered.reconcile([record(3, "event-3"), record(2, "event-2")])
  assert.deepEqual(
    ordered.snapshot().events.map((item) => item.id),
    ["event-2", "event-3"],
  )
  assert.equal(ordered.snapshot().latestSequence, 3)

  assert.throws(
    () =>
      ordered.reconcile(
        Array.from({ length: MUTATION_OUT_OF_ORDER_RECORD_LIMIT + 1 }, (_, index) =>
          record(index + 4, `overflow-${index}`),
        ),
      ),
    /reconciliation exceeded/,
  )
})

test("mutation accumulator applies a serialized-byte ceiling and retains terminal state", () => {
  const accumulator = new MutationEventAccumulator()
  const payload = { value: "x".repeat(40_000) }
  for (let index = 0; index < MUTATION_LIVE_EVENT_LIMIT; index += 1) {
    accumulator.addSseEvent(event(`large-${index}`, "custom.large", payload))
  }
  accumulator.addSseEvent(event("terminal", "session.completed", payload))
  const snapshot = accumulator.snapshot()
  const retainedBytes = snapshot.events.reduce(
    (total, item) => total + new TextEncoder().encode(JSON.stringify(item)).byteLength,
    0,
  )
  assert.ok(retainedBytes <= MUTATION_LIVE_EVENT_MAX_BYTES)
  assert.equal(snapshot.eventsTruncated, true)
  assert.ok(snapshot.events.some((item) => item.id === "terminal"))
})

test("mutation accumulator byte-bounds retained event identities", () => {
  const accumulator = new MutationEventAccumulator()
  const largeIdBytes = new TextEncoder().encode(
    `0000${"😀".repeat(MUTATION_EVENT_ID_MAX_CHARS - 4)}`,
  ).byteLength
  const eventCount = Math.floor(MUTATION_RECENT_EVENT_ID_MAX_BYTES / largeIdBytes) + 2
  const eventIds = Array.from({ length: eventCount }, (_, index) => {
    const prefix = String(index).padStart(4, "0")
    return `${prefix}${"😀".repeat(MUTATION_EVENT_ID_MAX_CHARS - prefix.length)}`
  })
  for (const eventId of eventIds) {
    assert.equal(accumulator.addSseEvent(event(eventId, "model.text.delta", { delta: "x" })), true)
  }

  assert.equal(
    accumulator.addSseEvent(event(eventIds.at(-1), "model.text.delta", { delta: "x" })),
    false,
  )
  assert.equal(
    accumulator.addSseEvent(event(eventIds[0], "model.text.delta", { delta: "x" })),
    true,
  )
})

test("durable reconciliation does not duplicate an SSE prefix beyond the ID cache", () => {
  const accumulator = new MutationEventAccumulator()
  const total = 2_100
  for (let index = 1; index <= total; index += 1) {
    accumulator.addSseEvent(event(`delta-${index}`, "model.text.delta", { delta: "x" }))
  }

  let boundaryFound = false
  for (let start = 1; start <= total; start += MUTATION_OUT_OF_ORDER_RECORD_LIMIT) {
    const end = Math.min(total, start + MUTATION_OUT_OF_ORDER_RECORD_LIMIT - 1)
    const page = Array.from({ length: end - start + 1 }, (_, offset) => {
      const sequence = start + offset
      return record(sequence, `delta-${sequence}`, "model.text.delta", { delta: "x" })
    }).reverse()
    const result = accumulator.reconcile(page, `delta-${total}`)
    boundaryFound ||= result.observedBoundaryFound
  }

  const snapshot = accumulator.snapshot()
  assert.equal(boundaryFound, true)
  assert.equal(snapshot.liveText.length, total)
  assert.equal(snapshot.latestSequence, total)
})

test("out-of-order durable pages retain the newest bounded event window", () => {
  const accumulator = new MutationEventAccumulator()
  accumulator.reconcile(
    Array.from({ length: MUTATION_LIVE_EVENT_LIMIT }, (_, index) =>
      record(MUTATION_LIVE_EVENT_LIMIT - index, `event-${MUTATION_LIVE_EVENT_LIMIT - index}`),
    ),
  )
  accumulator.reconcile(
    Array.from({ length: 5 }, (_, index) => record(105 - index, `event-${105 - index}`)),
  )

  assert.deepEqual(
    accumulator.snapshot().events.map((item) => item.id),
    Array.from({ length: MUTATION_LIVE_EVENT_LIMIT }, (_, index) => `event-${index + 6}`),
  )
})

test("accepted disconnect reconnects from the latest durable marker and deduplicates overlap", async () => {
  const attemptRequests = []
  const phases = []
  const attemptCallbacks = []
  let attemptIndex = 0
  let eventRead = 0
  const controller = new MutationTransportController(
    controllerOptions({
      request: request(),
      onChange: (snapshot) => phases.push(snapshot.phase),
      policy: { reconnectJitterRatio: 0 },
      io: immediateIo({
        async attempt(attemptRequest, callbacks) {
          attemptRequests.push(attemptRequest)
          attemptCallbacks.push(callbacks)
          attemptIndex += 1
          callbacks.onAccepted()
          if (attemptIndex === 1) {
            callbacks.onEvent(event("event-2"), `${SESSION_ID}:event-2`)
            return {
              kind: "network_failed",
              accepted: true,
              error: new Error("socket closed"),
            }
          }
          attemptCallbacks[0].onEvent(event("event-stale"), `${SESSION_ID}:event-stale`)
          callbacks.onEvent(event("event-4", "session.completed"), `${SESSION_ID}:event-4`)
          return { kind: "closed", accepted: true }
        },
        async readEvents(_sessionId, _afterSequence) {
          eventRead += 1
          return eventRead === 1
            ? eventsPage([record(2, "event-2"), record(3, "event-3")], { scanThrough: 3 })
            : eventsPage([record(4, "event-4", "session.completed")], { scanThrough: 4 })
        },
        async readState() {
          return eventRead === 1 ? state("running") : state("completed")
        },
      }),
    }),
  )

  const snapshot = await controller.start()

  assert.equal(snapshot.phase, "terminal")
  assert.deepEqual(
    snapshot.events.map((item) => item.id),
    ["event-2", "event-3", "event-4"],
  )
  assert.equal(attemptRequests.length, 2)
  assert.equal(attemptRequests[0].lastEventId, null)
  assert.equal(attemptRequests[1].lastEventId, `${SESSION_ID}:event-3`)
  assert.equal(snapshot.latestSseId, `${SESSION_ID}:event-4`)
  assert.equal(snapshot.replayMarker, `${SESSION_ID}:event-4`)
  assert.ok(phases.includes("reconnecting"))
  assert.equal(snapshot.reconnectAttempt, 0)
  assert.equal(snapshot.failure, null)
})

test("accepted pre-frame replay waits for the exact operation epoch before trusting terminal events", async () => {
  const attemptRequests = []
  let reads = 0
  let abortedByHistoricalTerminal = null
  const controller = transportController({
    request: request(),
    policy: { reconnectMaxAttempts: 1, reconnectJitterRatio: 0 },
    io: immediateIo({
      async attempt(attemptRequest, callbacks, signal) {
        attemptRequests.push(attemptRequest)
        callbacks.onAccepted()
        if (attemptRequests.length === 1) {
          return {
            kind: "network_failed",
            accepted: true,
            error: new Error("body disconnected before its first frame"),
          }
        }
        callbacks.onEvent(
          event("event-old-terminal", "session.completed"),
          `${SESSION_ID}:event-old-terminal`,
        )
        abortedByHistoricalTerminal = signal.aborted
        callbacks.onEvent(event("event-resumed", "session.resumed"), `${SESSION_ID}:event-resumed`)
        callbacks.onEvent(
          acceptanceEvent("event-acceptance", "event-resumed"),
          `${SESSION_ID}:event-acceptance`,
        )
        callbacks.onEvent(
          event("event-current-terminal", "session.completed"),
          `${SESSION_ID}:event-current-terminal`,
        )
        return { kind: "closed", accepted: true }
      },
      async readEvents() {
        reads += 1
        if (reads === 1) return eventsPage([], { scanThrough: 1 })
        return eventsPage(
          [
            record(2, "event-old-terminal", "session.completed"),
            record(3, "event-resumed", "session.resumed"),
            acceptanceRecord(4, "event-acceptance", "event-resumed"),
            record(5, "event-current-terminal", "session.completed"),
          ],
          { scanThrough: 5 },
        )
      },
      async readState() {
        return reads === 1 ? state("running") : state("completed")
      },
    }),
  })

  const snapshot = await controller.start()

  assert.equal(abortedByHistoricalTerminal, false)
  assert.equal(attemptRequests.length, 2)
  assert.equal(attemptRequests[1].lastEventId, `${SESSION_ID}:event-baseline`)
  assert.equal(snapshot.phase, "terminal")
  assert.equal(snapshot.failure, null)
})

test("an accepted replay open clears the active connection failure while preserving retry bounds", async () => {
  let attempts = 0
  let announceReplayOpen
  const replayOpened = new Promise((resolve) => {
    announceReplayOpen = resolve
  })
  const controller = transportController({
    request: request(),
    policy: { reconnectMaxAttempts: 1, reconnectJitterRatio: 0 },
    io: immediateIo({
      async attempt(_request, callbacks, signal) {
        attempts += 1
        callbacks.onAccepted()
        if (attempts === 1) {
          callbacks.onEvent(event("event-2"), `${SESSION_ID}:event-2`)
          return {
            kind: "network_failed",
            accepted: true,
            error: new Error("socket closed"),
          }
        }
        announceReplayOpen()
        return new Promise((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true },
          )
        })
      },
      async readEvents() {
        return eventsPage([], { scanThrough: 2 })
      },
      async readState() {
        return state("running")
      },
    }),
  })

  const running = controller.start()
  await replayOpened

  assert.equal(controller.snapshot().phase, "streaming")
  assert.equal(controller.snapshot().failure, null)
  assert.equal(controller.snapshot().reconnectAttempt, 1)

  controller.dispose()
  const snapshot = await running
  assert.equal(snapshot.phase, "cancelled")
})

test("duplicate replay frames do not regress markers or reset retry bounds", async () => {
  let attempts = 0
  let eventReads = 0
  let markerAfterDuplicate = null
  let controller
  controller = transportController({
    request: request(),
    policy: { reconnectMaxAttempts: 1, reconnectJitterRatio: 0 },
    io: immediateIo({
      async attempt(_request, callbacks) {
        attempts += 1
        callbacks.onAccepted()
        if (attempts === 1) {
          callbacks.onEvent(event("event-2"), `${SESSION_ID}:event-2`)
          callbacks.onEvent(event("event-3"), `${SESSION_ID}:event-3`)
        } else if (attempts === 2) {
          callbacks.onEvent(event("event-2"), `${SESSION_ID}:event-2`)
          markerAfterDuplicate = controller.snapshot().replayMarker
        }
        return {
          kind: "network_failed",
          accepted: true,
          error: new Error("socket closed"),
        }
      },
      async readEvents() {
        eventReads += 1
        if (eventReads === 1) {
          return eventsPage([record(2, "event-2"), record(3, "event-3")], {
            scanThrough: 3,
          })
        }
        if (eventReads === 2) return eventsPage([], { scanThrough: 3 })
        return eventsPage([record(4, "event-terminal", "session.completed")], {
          scanThrough: 4,
        })
      },
      async readState() {
        return eventReads < 3 ? state("running") : state("completed")
      },
    }),
  })

  const snapshot = await controller.start()

  assert.equal(attempts, 2)
  assert.equal(markerAfterDuplicate, `${SESSION_ID}:event-3`)
  assert.equal(snapshot.phase, "terminal")
  assert.equal(snapshot.failure, null)
})

test("a historical duplicate terminal frame cannot stop a newly accepted mutation", async () => {
  const snapshots = []
  let attempts = 0
  let eventReads = 0
  let abortedAfterDuplicate = null
  let emissionsFromDuplicate = null
  const controller = transportController({
    request: request(),
    onChange: (snapshot) => snapshots.push(snapshot),
    policy: { reconnectMaxAttempts: 1, reconnectJitterRatio: 0 },
    io: immediateIo({
      async attempt(_request, callbacks, signal) {
        attempts += 1
        if (attempts === 1) {
          return {
            kind: "network_failed",
            accepted: false,
            error: new Error("initial response lost"),
          }
        }
        callbacks.onAccepted()
        callbacks.onEvent(
          event("event-old-terminal", "session.completed"),
          `${SESSION_ID}:event-old-terminal`,
        )
        callbacks.onEvent(event("event-resumed", "session.resumed"), `${SESSION_ID}:event-resumed`)
        callbacks.onEvent(
          acceptanceEvent("event-acceptance", "event-resumed"),
          `${SESSION_ID}:event-acceptance`,
        )
        const emissionsBeforeDuplicate = snapshots.length
        callbacks.onEvent(
          event("event-old-terminal", "session.completed"),
          `${SESSION_ID}:event-old-terminal`,
        )
        emissionsFromDuplicate = snapshots.length - emissionsBeforeDuplicate
        abortedAfterDuplicate = signal.aborted
        return { kind: "observer_failed", accepted: true, error: observerError(false) }
      },
      async readEvents() {
        eventReads += 1
        if (eventReads === 1) return eventsPage([], { scanThrough: 1 })
        if (eventReads === 2) {
          return eventsPage(
            [
              record(2, "event-old-terminal", "session.completed"),
              record(3, "event-resumed", "session.resumed"),
              acceptanceRecord(4, "event-acceptance", "event-resumed"),
            ],
            { scanThrough: 4 },
          )
        }
        if (eventReads === 3) return eventsPage([], { scanThrough: 4 })
        return eventsPage([record(5, "event-terminal", "session.completed")], {
          scanThrough: 5,
        })
      },
      async readState() {
        return eventReads < 3 ? state("running") : state("completed")
      },
    }),
  })

  const snapshot = await controller.start()

  assert.equal(abortedAfterDuplicate, false)
  assert.equal(emissionsFromDuplicate, 0)
  assert.equal(attempts, 2)
  assert.equal(eventReads, 4)
  assert.equal(snapshot.phase, "terminal")
  assert.equal(snapshot.failure, null)
})

test("a rejected pre-acceptance attempt is never retried as a mutation", async () => {
  const attemptRequests = []
  let reconciliations = 0
  const controller = new MutationTransportController(
    controllerOptions({
      request: request(),
      io: immediateIo({
        async attempt(attemptRequest, callbacks) {
          attemptRequests.push(attemptRequest)
          if (attemptRequests.length === 1) {
            throw new Error("connection reset")
          }
          callbacks.onAccepted()
          callbacks.onEvent(event("event-2", "session.resumed"), `${SESSION_ID}:event-2`)
          callbacks.onEvent(acceptanceEvent("event-3", "event-2"), `${SESSION_ID}:event-3`)
          callbacks.onEvent(event("event-4", "session.completed"), `${SESSION_ID}:event-4`)
          return { kind: "closed", accepted: true }
        },
        async readEvents() {
          reconciliations += 1
          return reconciliations === 1
            ? eventsPage([], { scanThrough: 1 })
            : eventsPage(
                [
                  record(2, "event-2", "session.resumed"),
                  acceptanceRecord(3, "event-3", "event-2"),
                  record(4, "event-4", "session.completed"),
                ],
                { scanThrough: 4 },
              )
        },
        async readState() {
          return reconciliations === 1 ? state("running") : state("completed")
        },
      }),
    }),
  )

  const snapshot = await controller.start()

  assert.equal(snapshot.phase, "terminal")
  assert.equal(attemptRequests.length, 2)
  assert.equal(attemptRequests[0].lastEventId, null)
  assert.equal(attemptRequests[1].lastEventId, `${SESSION_ID}:event-baseline`)
  assert.equal(reconciliations, 2)
})

test("REST reconciliation can prove ambiguous acceptance with the matching durable event", async () => {
  let attempts = 0
  let reads = 0
  const controller = transportController({
    request: request(),
    io: immediateIo({
      async attempt() {
        attempts += 1
        throw new Error("response lost")
      },
      async readEvents() {
        reads += 1
        return eventsPage(
          [
            record(2, "event-2", "session.resumed"),
            acceptanceRecord(3, "event-3", "event-2"),
            record(4, "event-4", "session.completed"),
          ],
          { scanThrough: 4 },
        )
      },
      async readState() {
        return state("completed")
      },
    }),
  })

  const snapshot = await controller.start()

  assert.equal(snapshot.phase, "terminal")
  assert.equal(snapshot.failure, null)
  assert.equal(attempts, 1)
  assert.equal(reads, 1)
})

test("replay-only recovery does not accept unrelated session activity", async () => {
  const attemptRequests = []
  const phases = []
  let eventReads = 0
  const controller = transportController({
    request: request({
      path: `/sessions/${SESSION_ID}/interrupt`,
      body: { reason: "operator request" },
    }),
    onChange: (snapshot) => phases.push(snapshot.phase),
    policy: { reconnectMaxAttempts: 1, fallbackMaxConsecutiveFailures: 2 },
    io: immediateIo({
      async attempt(attemptRequest, callbacks) {
        attemptRequests.push(attemptRequest)
        if (attemptRequests.length === 1) {
          return {
            kind: "network_failed",
            accepted: false,
            error: new Error("connection reset"),
          }
        }
        callbacks.onAccepted()
        if (attemptRequests.length === 2) {
          callbacks.onEvent(event("event-2", "model.completed"), `${SESSION_ID}:event-2`)
          callbacks.onEvent(event("event-3", "session.completed"), `${SESSION_ID}:event-3`)
        }
        return { kind: "closed", accepted: true }
      },
      async readEvents() {
        eventReads += 1
        if (eventReads === 1) return eventsPage([], { scanThrough: 1 })
        if (eventReads === 2) {
          return eventsPage(
            [record(2, "event-2", "model.completed"), record(3, "event-3", "session.completed")],
            { scanThrough: 3 },
          )
        }
        return eventsPage([], { scanThrough: 3 })
      },
      async readState() {
        return eventReads === 1 ? state("running") : state("completed")
      },
    }),
  })

  const snapshot = await controller.start()

  assert.equal(snapshot.phase, "transport_failed")
  assert.equal(snapshot.failure.kind, "network")
  assert.deepEqual(
    snapshot.events.map((item) => item.id),
    ["event-2", "event-3"],
  )
  assert.equal(attemptRequests[0].lastEventId, null)
  assert.equal(attemptRequests[1].lastEventId, `${SESSION_ID}:event-baseline`)
  assert.ok(attemptRequests.every((item, index) => index === 0 || item.lastEventId !== null))
  assert.ok(!phases.includes("streaming"))
})

test("an unconfirmed initial mutation fails after bounded replay-only probes", async () => {
  const attemptRequests = []
  let reads = 0
  const controller = new MutationTransportController(
    controllerOptions({
      request: request(),
      policy: { reconnectMaxAttempts: 1, fallbackMaxConsecutiveFailures: 2 },
      io: immediateIo({
        async attempt(attemptRequest, callbacks) {
          attemptRequests.push(attemptRequest)
          if (attemptRequests.length === 1) {
            return {
              kind: "network_failed",
              accepted: false,
              error: new Error("connection reset"),
            }
          }
          callbacks.onAccepted()
          return { kind: "closed", accepted: true }
        },
        async readEvents() {
          reads += 1
          return eventsPage([], { scanThrough: 1 })
        },
        async readState() {
          return state("running")
        },
      }),
    }),
  )

  const snapshot = await controller.start()
  assert.equal(snapshot.phase, "transport_failed")
  assert.equal(snapshot.failure.kind, "network")
  assert.equal(attemptRequests.length, 2)
  assert.equal(attemptRequests[0].lastEventId, null)
  assert.equal(attemptRequests[1].lastEventId, `${SESSION_ID}:event-baseline`)
  assert.equal(reads, 4)
})

test("a hung initial connection is aborted and reconciled without repeating the mutation", async () => {
  const timers = controlledWatchdogs()
  let attempts = 0
  let attemptSignal = null
  const controller = transportController({
    request: request(),
    io: immediateIo({
      watchdog: timers.wait,
      async attempt(_request, _callbacks, signal) {
        attempts += 1
        attemptSignal = signal
        return new Promise((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true },
          )
        })
      },
      async readEvents() {
        return eventsPage(
          [
            record(2, "event-terminal", "session.completed"),
            acceptanceRecord(3, "event-acceptance", "event-terminal", "session.completed"),
          ],
          { scanThrough: 3 },
        )
      },
      async readState() {
        return state("completed")
      },
    }),
  })

  const running = controller.start()
  await tick()
  const timer = timers.expireLatest()
  const snapshot = await running

  assert.equal(timer.milliseconds, 15_000)
  assert.equal(attemptSignal.aborted, true)
  assert.equal(attempts, 1)
  assert.equal(snapshot.phase, "terminal")
})

test("an accepted initial stream has a deadline for its first operation event", async () => {
  const timers = controlledWatchdogs()
  let attempts = 0
  let attemptSignal = null
  const controller = transportController({
    request: request(),
    io: immediateIo({
      watchdog: timers.wait,
      async attempt(_request, callbacks, signal) {
        attempts += 1
        attemptSignal = signal
        callbacks.onAccepted()
        return new Promise((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true },
          )
        })
      },
      async readEvents() {
        return eventsPage(
          [
            record(2, "event-terminal", "session.completed"),
            acceptanceRecord(3, "event-acceptance", "event-terminal", "session.completed"),
          ],
          { scanThrough: 3 },
        )
      },
      async readState() {
        return state("completed")
      },
    }),
  })

  const running = controller.start()
  await waitFor(
    () => timers.waits.some((item) => item.milliseconds === 30_000 && !item.signal.aborted),
    "initial acceptance confirmation watchdog was not armed",
  )
  const timer = timers.expireLatest()
  const snapshot = await running

  assert.equal(timer.milliseconds, 30_000)
  assert.equal(attemptSignal.aborted, true)
  assert.equal(attempts, 1)
  assert.equal(snapshot.phase, "terminal")
})

test("an accepted replay without the exact mutation marker has a confirmation deadline", async () => {
  const timers = controlledWatchdogs()
  const attemptSignals = []
  let attempts = 0
  let reads = 0
  const controller = transportController({
    request: request(),
    policy: { reconnectJitterRatio: 0 },
    io: immediateIo({
      watchdog: timers.wait,
      async attempt(_request, callbacks, signal) {
        attempts += 1
        attemptSignals.push(signal)
        if (attempts === 1) {
          return {
            kind: "network_failed",
            accepted: false,
            error: new Error("response lost"),
          }
        }
        callbacks.onAccepted()
        callbacks.onEvent(event("event-unrelated"), `${SESSION_ID}:event-unrelated`)
        return new Promise((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true },
          )
        })
      },
      async readEvents() {
        reads += 1
        if (reads === 1) return eventsPage([], { scanThrough: 1 })
        return eventsPage(
          [
            record(2, "event-terminal", "session.completed"),
            acceptanceRecord(3, "event-acceptance", "event-terminal", "session.completed"),
          ],
          { scanThrough: 3 },
        )
      },
      async readState() {
        return reads === 1 ? state("running") : state("completed")
      },
    }),
  })

  const running = controller.start()
  await waitFor(
    () => attempts >= 2 && timers.waits.some((item) => !item.signal.aborted),
    "replay confirmation watchdog was not armed",
  )
  const timer = timers.expireLatest()
  const snapshot = await running

  assert.equal(timer.milliseconds, 30_000)
  assert.equal(attemptSignals[1].aborted, true)
  assert.equal(attempts, 2)
  assert.equal(snapshot.phase, "terminal")
})

test("a durable terminal event aborts the live reader and reconciles immediately", async () => {
  let streamSignal = null
  let announceStateRead
  let resolveStateRead
  const stateReadStarted = new Promise((resolve) => {
    announceStateRead = resolve
  })
  const snapshots = []
  const controller = transportController({
    request: request(),
    onChange: (snapshot) => snapshots.push(snapshot),
    io: immediateIo({
      async attempt(_request, callbacks, signal) {
        streamSignal = signal
        callbacks.onAccepted()
        callbacks.onEvent(
          event("event-terminal", "session.completed"),
          `${SESSION_ID}:event-terminal`,
        )
        return new Promise((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true },
          )
        })
      },
      async readEvents(_sessionId, afterSequence) {
        return eventsPage(
          afterSequence < 2 ? [record(2, "event-terminal", "session.completed")] : [],
          { scanThrough: 2 },
        )
      },
      async readState() {
        announceStateRead()
        return new Promise((resolve) => {
          resolveStateRead = resolve
        })
      },
    }),
  })

  const running = controller.start()
  await stateReadStarted

  assert.equal(streamSignal.aborted, true)
  assert.ok(snapshots.every((snapshot) => snapshot.failure === null))

  resolveStateRead(state("completed"))
  const snapshot = await running

  assert.equal(snapshot.phase, "terminal")
  assert.equal(snapshot.failure, null)
})

test("terminal recovery stays on REST reconciliation after a transient state timeout", async () => {
  const timers = controlledWatchdogs()
  const snapshots = []
  let attemptCalls = 0
  let stateReads = 0
  let stalledStateSignal = null
  const controller = transportController({
    request: request(),
    onChange: (snapshot) => snapshots.push(snapshot),
    policy: { reconciliationRequestTimeoutMs: 321 },
    io: immediateIo({
      watchdog: timers.wait,
      async attempt(_request, callbacks, signal) {
        attemptCalls += 1
        callbacks.onAccepted()
        callbacks.onEvent(
          event("event-terminal", "session.completed"),
          `${SESSION_ID}:event-terminal`,
        )
        return new Promise((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true },
          )
        })
      },
      async readEvents(_sessionId, afterSequence) {
        return eventsPage(
          afterSequence < 2 ? [record(2, "event-terminal", "session.completed")] : [],
          { scanThrough: 2 },
        )
      },
      async readState(_sessionId, signal) {
        stateReads += 1
        if (stateReads > 1) return state("completed")
        stalledStateSignal = signal
        return new Promise((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true },
          )
        })
      },
    }),
  })

  const running = controller.start()
  const timer = await waitFor(
    () =>
      timers.waits.find(
        (item) => item.milliseconds === 321 && !item.settled && !item.signal.aborted,
      ),
    "terminal state reconciliation watchdog was not armed",
  )
  timer.settled = true
  timer.resolve()
  const snapshot = await running

  assert.equal(stalledStateSignal.aborted, true)
  assert.equal(attemptCalls, 1)
  assert.ok(!snapshots.some((item) => item.failure?.kind === "network"))
  assert.equal(snapshot.phase, "terminal")
  assert.equal(snapshot.failure, null)
})

test("a buffered terminal event remains REST-only when the page becomes hidden", async () => {
  const pageVisibility = visibility(true)
  const snapshots = []
  let attempts = 0
  let announcePausedAttempt
  const pausedAttempt = new Promise((resolve) => {
    announcePausedAttempt = resolve
  })
  const controller = transportController({
    request: request(),
    visibility: pageVisibility.source,
    onChange: (snapshot) => snapshots.push(snapshot),
    io: immediateIo({
      async attempt(_request, callbacks) {
        attempts += 1
        callbacks.onAccepted()
        if (attempts === 1) {
          pageVisibility.set(false)
          callbacks.onEvent(
            event("event-terminal", "session.completed"),
            `${SESSION_ID}:event-terminal`,
          )
          announcePausedAttempt()
          return { kind: "aborted", accepted: true }
        }
        return {
          kind: "network_failed",
          accepted: true,
          error: new Error("unexpected replacement stream"),
        }
      },
      async readEvents() {
        return eventsPage([record(2, "event-terminal", "session.completed")], {
          scanThrough: 2,
        })
      },
      async readState() {
        return state("completed")
      },
    }),
  })

  const running = controller.start()
  await pausedAttempt
  await tick()
  assert.equal(controller.snapshot().phase, "paused")

  pageVisibility.set(true)
  const snapshot = await running

  assert.equal(attempts, 1)
  assert.ok(!snapshots.some((item) => item.failure?.kind === "network"))
  assert.equal(snapshot.phase, "terminal")
  assert.equal(snapshot.failure, null)
  assert.equal(pageVisibility.listenerCount(), 0)
})

test("stalled event and state reconciliation reads are aborted by bounded deadlines", async () => {
  for (const stalledRead of ["events", "state"]) {
    const timers = controlledWatchdogs()
    let eventReads = 0
    let stateReads = 0
    let stalledSignal = null
    const stall = (signal) => {
      stalledSignal = signal
      return new Promise(() => undefined)
    }
    const controller = transportController({
      request: request(),
      policy: {
        reconciliationRequestTimeoutMs: 321,
        fallbackMaxConsecutiveFailures: 1,
      },
      io: immediateIo({
        watchdog: timers.wait,
        async attempt(_request, callbacks) {
          callbacks.onAccepted()
          return { kind: "observer_failed", accepted: true, error: observerError(false) }
        },
        async readEvents(_sessionId, _afterSequence, _limit, signal) {
          eventReads += 1
          if (stalledRead === "events" && eventReads === 2) return stall(signal)
          return eventsPage([], { scanThrough: 1 })
        },
        async readState(_sessionId, signal) {
          stateReads += 1
          if (stalledRead === "state" && stateReads === 2) return stall(signal)
          return state("running")
        },
      }),
    })

    const running = controller.start()
    const timer = await waitFor(
      () =>
        timers.waits.find(
          (item) => item.milliseconds === 321 && !item.settled && !item.signal.aborted,
        ),
      `${stalledRead} reconciliation watchdog was not armed`,
    )
    timer.settled = true
    timer.resolve()
    const snapshot = await running

    assert.equal(stalledSignal.aborted, true)
    assert.equal(snapshot.phase, "transport_failed")
    assert.equal(snapshot.failure.kind, "reconciliation")
    assert.equal(snapshot.failure.error.name, "MutationTransportTimeoutError")
  }
})

test("a structured stream error from another session is treated as protocol failure", async () => {
  const snapshots = []
  const controller = transportController({
    request: request(),
    onChange: (snapshot) => snapshots.push(snapshot),
    io: immediateIo({
      async attempt(_request, callbacks) {
        callbacks.onAccepted()
        return {
          kind: "runtime_failed",
          accepted: true,
          error: { ...runtimeError(), session_id: "session-other" },
        }
      },
      async readEvents() {
        return eventsPage(
          [
            record(2, "event-resumed", "session.resumed"),
            acceptanceRecord(3, "event-accepted", "event-resumed"),
            record(4, "event-terminal", "session.failed"),
          ],
          { scanThrough: 4 },
        )
      },
      async readState() {
        return state("failed")
      },
    }),
  })

  const snapshot = await controller.start()

  assert.equal(snapshot.phase, "terminal")
  assert.ok(snapshots.some((item) => item.failure?.kind === "protocol"))
  assert.ok(!snapshots.some((item) => item.phase === "runtime_failed"))
})

test("an initial HTTP rejection is a request failure", async () => {
  let reads = 0
  const controller = transportController({
    request: request(),
    io: immediateIo({
      async attempt() {
        return {
          kind: "http_failed",
          accepted: false,
          status: 422,
          error: new Error("HTTP 422"),
        }
      },
      async readEvents() {
        reads += 1
        return eventsPage()
      },
      async readState() {
        return state("running")
      },
    }),
  })

  const snapshot = await controller.start()
  assert.equal(snapshot.phase, "request_failed")
  assert.equal(snapshot.failure.kind, "http")
  assert.equal(reads, 0)
})

test("an ambiguous initial HTTP failure recovers only through the durable baseline", async () => {
  const attemptRequests = []
  const phases = []
  let reads = 0
  const controller = transportController({
    request: request(),
    onChange: (snapshot) => phases.push(snapshot.phase),
    policy: { reconnectMaxAttempts: 1, reconnectJitterRatio: 0 },
    io: immediateIo({
      async attempt(attemptRequest, callbacks) {
        attemptRequests.push(attemptRequest)
        if (attemptRequests.length === 1) {
          return {
            kind: "http_failed",
            accepted: false,
            status: 504,
            error: new Error("HTTP 504"),
          }
        }
        callbacks.onAccepted()
        callbacks.onEvent(event("event-2", "session.resumed"), `${SESSION_ID}:event-2`)
        callbacks.onEvent(acceptanceEvent("event-3", "event-2"), `${SESSION_ID}:event-3`)
        callbacks.onEvent(event("event-4", "session.completed"), `${SESSION_ID}:event-4`)
        return { kind: "closed", accepted: true }
      },
      async readEvents() {
        reads += 1
        return reads === 1
          ? eventsPage([], { scanThrough: 1 })
          : eventsPage(
              [
                record(2, "event-2", "session.resumed"),
                acceptanceRecord(3, "event-3", "event-2"),
                record(4, "event-4", "session.completed"),
              ],
              { scanThrough: 4 },
            )
      },
      async readState() {
        return reads === 1 ? state("running") : state("completed")
      },
    }),
  })

  const snapshot = await controller.start()

  assert.equal(snapshot.phase, "terminal")
  assert.equal(snapshot.failure, null)
  assert.equal(attemptRequests.length, 2)
  assert.equal(attemptRequests[0].lastEventId, null)
  assert.equal(attemptRequests[1].lastEventId, `${SESSION_ID}:event-baseline`)
  assert.ok(!phases.includes("request_failed"))
})

test("an HTTP failure after acceptance reconciles without reissuing the mutation", async () => {
  let attempts = 0
  let reads = 0
  const phases = []
  const controller = transportController({
    request: request(),
    onChange: (snapshot) => phases.push(snapshot.phase),
    policy: { reconnectMaxAttempts: 1 },
    io: immediateIo({
      async attempt(_request, callbacks) {
        attempts += 1
        if (attempts === 1) {
          callbacks.onAccepted()
          return {
            kind: "network_failed",
            accepted: true,
            error: new Error("socket closed"),
          }
        }
        return {
          kind: "http_failed",
          accepted: false,
          status: 409,
          error: new Error("HTTP 409"),
        }
      },
      async readEvents() {
        reads += 1
        return reads < 3
          ? eventsPage([], { scanThrough: 1 })
          : eventsPage(
              [
                record(2, "event-resumed", "session.resumed"),
                acceptanceRecord(3, "event-accepted", "event-resumed"),
                record(4, "event-terminal", "session.completed"),
              ],
              { scanThrough: 4 },
            )
      },
      async readState() {
        return reads < 3 ? state("running") : state("completed")
      },
    }),
  })

  const snapshot = await controller.start()
  assert.equal(snapshot.phase, "terminal")
  assert.equal(attempts, 2)
  assert.ok(phases.includes("polling_fallback"))
  assert.ok(!phases.includes("request_failed"))
})

test("a rejected attempt after acceptance remains inside durable recovery", async () => {
  let attempts = 0
  let reads = 0
  const controller = transportController({
    request: request(),
    io: immediateIo({
      async attempt(_request, callbacks) {
        attempts += 1
        callbacks.onAccepted()
        throw new Error("stream reader rejected")
      },
      async readEvents() {
        reads += 1
        return eventsPage(
          [
            record(2, "event-resumed", "session.resumed"),
            acceptanceRecord(3, "event-accepted", "event-resumed"),
            record(4, "event-terminal", "session.completed"),
          ],
          { scanThrough: 4 },
        )
      },
      async readState() {
        return state("completed")
      },
    }),
  })

  const snapshot = await controller.start()

  assert.equal(snapshot.phase, "terminal")
  assert.equal(snapshot.failure, null)
  assert.equal(attempts, 1)
  assert.equal(reads, 1)
})

test("runtime failure is terminal while a non-retryable observer failure uses REST fallback", async () => {
  let runtimeReads = 0
  const runtimeController = transportController({
    request: request(),
    io: immediateIo({
      async attempt(_request, callbacks) {
        callbacks.onAccepted()
        return { kind: "runtime_failed", accepted: true, error: runtimeError() }
      },
      async readEvents() {
        runtimeReads += 1
        return eventsPage()
      },
      async readState() {
        return state("failed")
      },
    }),
  })
  const runtimeSnapshot = await runtimeController.start()
  assert.equal(runtimeSnapshot.phase, "runtime_failed")
  assert.equal(runtimeSnapshot.failure.kind, "runtime")
  assert.equal(runtimeReads, 0)

  const phases = []
  let eventReads = 0
  const fallbackController = transportController({
    request: request(),
    onChange: (snapshot) => phases.push(snapshot.phase),
    io: immediateIo({
      async attempt(_request, callbacks) {
        callbacks.onAccepted()
        return { kind: "observer_failed", accepted: true, error: observerError(false) }
      },
      async readEvents() {
        eventReads += 1
        return eventReads === 1
          ? eventsPage([], { scanThrough: 1 })
          : eventsPage(
              [
                record(2, "event-resumed", "session.resumed"),
                acceptanceRecord(3, "event-accepted", "event-resumed"),
                record(4, "event-terminal", "session.failed"),
              ],
              { scanThrough: 4 },
            )
      },
      async readState() {
        return eventReads === 1 ? state("running") : state("failed")
      },
    }),
  })
  const fallbackSnapshot = await fallbackController.start()
  assert.equal(fallbackSnapshot.phase, "terminal")
  assert.ok(phases.includes("polling_fallback"))
  assert.equal(eventReads, 2)
})

test("retry exhaustion enters bounded polling fallback", async () => {
  let attempts = 0
  let eventReads = 0
  const phases = []
  const controller = transportController({
    request: request(),
    onChange: (snapshot) => phases.push(snapshot.phase),
    policy: {
      reconnectMaxAttempts: 1,
      reconnectMaxElapsedMs: 1_000,
      reconnectJitterRatio: 0,
    },
    io: immediateIo({
      async attempt(_request, callbacks) {
        attempts += 1
        callbacks.onAccepted()
        return {
          kind: "network_failed",
          accepted: true,
          error: new Error("offline"),
        }
      },
      async readEvents() {
        eventReads += 1
        return eventReads < 3
          ? eventsPage([], { scanThrough: 1 })
          : eventsPage(
              [
                record(2, "event-resumed", "session.resumed"),
                acceptanceRecord(3, "event-accepted", "event-resumed"),
                record(4, "event-terminal", "session.interrupted"),
              ],
              { scanThrough: 4 },
            )
      },
      async readState() {
        return eventReads < 3 ? state("running") : state("interrupted")
      },
    }),
  })

  const snapshot = await controller.start()

  assert.equal(snapshot.phase, "terminal")
  assert.equal(attempts, 2)
  assert.ok(phases.includes("polling_fallback"))
})

test("polling fallback stops after bounded consecutive reconciliation failures", async () => {
  let reads = 0
  const controller = transportController({
    request: request(),
    policy: {
      reconnectMaxAttempts: 1,
      reconnectJitterRatio: 0,
      fallbackMaxConsecutiveFailures: 3,
    },
    io: immediateIo({
      async attempt(_request, callbacks) {
        callbacks.onAccepted()
        return { kind: "observer_failed", accepted: true, error: observerError(false) }
      },
      async readEvents() {
        reads += 1
        throw new Error("durable store unavailable")
      },
      async readState() {
        assert.fail("state must not be read after an event-page failure")
      },
    }),
  })

  const snapshot = await controller.start()

  assert.equal(snapshot.phase, "transport_failed")
  assert.equal(snapshot.failure.kind, "reconciliation")
  assert.equal(reads, 4)
})

test("successful fallback reconciliation clears a prior acceptance failure", async () => {
  const snapshots = []
  let attempts = 0
  let eventReads = 0
  let fallbackDelays = 0
  let announceRecoveredPoll
  const recoveredPollStarted = new Promise((resolve) => {
    announceRecoveredPoll = resolve
  })
  const controller = transportController({
    request: request(),
    onChange: (snapshot) => snapshots.push(snapshot),
    policy: { reconnectMaxAttempts: 1, reconnectJitterRatio: 0 },
    io: immediateIo({
      async attempt(_request, callbacks) {
        attempts += 1
        if (attempts === 1) {
          return {
            kind: "network_failed",
            accepted: false,
            error: new Error("initial response lost"),
          }
        }
        callbacks.onAccepted()
        return { kind: "observer_failed", accepted: true, error: observerError(false) }
      },
      async readEvents() {
        eventReads += 1
        if (eventReads < 4) return eventsPage([], { scanThrough: 1 })
        return eventsPage(
          [
            record(2, "event-resumed", "session.resumed"),
            acceptanceRecord(3, "event-acceptance", "event-resumed"),
          ],
          { scanThrough: 3 },
        )
      },
      async readState() {
        return state("running")
      },
      async delay(milliseconds, signal) {
        if (milliseconds !== 2_000) return
        fallbackDelays += 1
        if (fallbackDelays === 1) return
        announceRecoveredPoll()
        await new Promise((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true },
          )
        })
      },
    }),
  })

  const running = controller.start()
  await recoveredPollStarted

  assert.ok(snapshots.some((snapshot) => snapshot.failure !== null))
  assert.equal(controller.snapshot().phase, "polling_fallback")
  assert.equal(controller.snapshot().failure, null)

  controller.dispose()
  const snapshot = await running
  assert.equal(snapshot.phase, "cancelled")
})

test("polling fallback bounds terminal state without a post-baseline boundary", async () => {
  let reads = 0
  const controller = transportController({
    request: request(),
    policy: { fallbackMaxConsecutiveFailures: 2 },
    io: immediateIo({
      async attempt(_request, callbacks) {
        callbacks.onAccepted()
        return { kind: "observer_failed", accepted: true, error: observerError(false) }
      },
      async readEvents() {
        reads += 1
        return reads === 1
          ? eventsPage(
              [
                record(2, "event-resumed", "session.resumed"),
                acceptanceRecord(3, "event-accepted", "event-resumed"),
              ],
              { scanThrough: 3 },
            )
          : eventsPage([], { scanThrough: 3 })
      },
      async readState() {
        return state("completed")
      },
    }),
  })

  const snapshot = await controller.start()
  assert.equal(snapshot.phase, "transport_failed")
  assert.equal(snapshot.failure.kind, "protocol")
  assert.equal(reads, 3)
})

test("polling fallback drains a bounded durable backlog without idle delays", async () => {
  let reads = 0
  let delays = 0
  const controller = transportController({
    request: request(),
    policy: { reconciliationMaxPages: 1 },
    io: immediateIo({
      async attempt(_request, callbacks) {
        callbacks.onAccepted()
        return { kind: "observer_failed", accepted: true, error: observerError(false) }
      },
      async readEvents(_sessionId, afterSequence) {
        reads += 1
        assert.equal(afterSequence, reads)
        const sequence = reads + 1
        const nextRecord =
          reads === 1
            ? record(sequence, `event-${sequence}`, "session.resumed")
            : reads === 2
              ? acceptanceRecord(sequence, `event-${sequence}`, "event-2")
              : record(sequence, `event-${sequence}`, "session.completed")
        return eventsPage([nextRecord], { hasMore: reads < 3, scanThrough: sequence })
      },
      async readState() {
        return reads === 3 ? state("completed") : state("running")
      },
      async delay() {
        delays += 1
      },
    }),
  })

  const snapshot = await controller.start()
  assert.equal(snapshot.phase, "terminal")
  assert.equal(reads, 3)
  assert.equal(delays, 0)
})

test("visibility pauses active work and disposal removes listeners and timers", async () => {
  const pageVisibility = visibility(false)
  let attempts = 0
  let activeSignal = null
  let activeCallbacks = null
  const phases = []
  const controller = transportController({
    request: request(),
    visibility: pageVisibility.source,
    onChange: (snapshot) => phases.push(snapshot.phase),
    io: immediateIo({
      attempt: async (_request, callbacks, signal) => {
        attempts += 1
        activeSignal = signal
        activeCallbacks = callbacks
        callbacks.onAccepted()
        return new Promise((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            {
              once: true,
            },
          )
        })
      },
      async readEvents() {
        return eventsPage()
      },
      async readState() {
        return state("running")
      },
    }),
  })

  const running = controller.start()
  await tick()
  assert.equal(attempts, 0)
  assert.equal(controller.snapshot().phase, "paused")

  pageVisibility.set(true)
  await tick()
  assert.equal(attempts, 1)
  assert.equal(controller.snapshot().phase, "streaming")

  pageVisibility.set(false)
  await tick()
  assert.equal(activeSignal.aborted, true)
  assert.equal(controller.snapshot().phase, "paused")

  controller.dispose()
  const snapshot = await running
  activeCallbacks.onAccepted()
  activeCallbacks.onEvent(event("event-late"), `${SESSION_ID}:event-late`)
  assert.equal(snapshot.phase, "cancelled")
  assert.equal(controller.snapshot().phase, "cancelled")
  assert.equal(controller.snapshot().events.length, 0)
  assert.equal(pageVisibility.listenerCount(), 0)
  assert.equal(phases.at(-1), "cancelled")
})

test("a late reconciliation response cannot mutate a cancelled transport", async () => {
  let announceRead
  let resolveRead
  let readSignal = null
  const readStarted = new Promise((resolve) => {
    announceRead = resolve
  })
  const controller = transportController({
    request: request(),
    io: immediateIo({
      async attempt(_request, callbacks) {
        callbacks.onAccepted()
        return {
          kind: "network_failed",
          accepted: true,
          error: new Error("socket closed"),
        }
      },
      async readEvents(_sessionId, _afterSequence, _limit, signal) {
        readSignal = signal
        announceRead()
        return new Promise((resolve) => {
          resolveRead = resolve
        })
      },
      async readState() {
        assert.fail("cancelled reconciliation must not continue to state")
      },
    }),
  })

  const running = controller.start()
  await readStarted
  controller.dispose()
  resolveRead(eventsPage([record(2, "event-late", "session.completed")], { scanThrough: 2 }))

  const snapshot = await running
  assert.equal(readSignal.aborted, true)
  assert.equal(snapshot.phase, "cancelled")
  assert.equal(snapshot.events.length, 0)
  assert.equal(snapshot.latestSequence, 1)
})

test("time spent in a hidden tab does not exhaust the active reconnect budget", async () => {
  const pageVisibility = visibility(true)
  let now = 0
  let attempts = 0
  let eventReads = 0
  let delayCalls = 0
  let announceFirstDelay
  const firstDelayStarted = new Promise((resolve) => {
    announceFirstDelay = resolve
  })
  const controller = transportController({
    request: request(),
    visibility: pageVisibility.source,
    policy: {
      reconnectMaxAttempts: 4,
      reconnectMaxElapsedMs: 1_000,
      reconnectJitterRatio: 0,
    },
    io: immediateIo({
      now: () => now,
      async attempt(_request, callbacks) {
        attempts += 1
        callbacks.onAccepted()
        if (attempts < 3) {
          return {
            kind: "network_failed",
            accepted: true,
            error: new Error("offline"),
          }
        }
        callbacks.onEvent(event("event-resumed", "session.resumed"), `${SESSION_ID}:event-resumed`)
        callbacks.onEvent(
          acceptanceEvent("event-accepted", "event-resumed"),
          `${SESSION_ID}:event-accepted`,
        )
        callbacks.onEvent(
          event("event-terminal", "session.completed"),
          `${SESSION_ID}:event-terminal`,
        )
        return { kind: "closed", accepted: true }
      },
      async readEvents() {
        eventReads += 1
        if (attempts < 3 && eventReads >= 3) {
          return eventsPage(
            [
              record(2, "fallback-resumed", "session.resumed"),
              acceptanceRecord(3, "fallback-accepted", "fallback-resumed"),
              record(4, "fallback-terminal", "session.completed"),
            ],
            { scanThrough: 4 },
          )
        }
        return eventsPage([], { scanThrough: 1 })
      },
      async readState() {
        return attempts >= 3 || eventReads >= 3 ? state("completed") : state("running")
      },
      async delay(_milliseconds, signal) {
        delayCalls += 1
        if (delayCalls > 1) return
        announceFirstDelay()
        await new Promise((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true },
          )
        })
      },
    }),
  })

  const running = controller.start()
  await firstDelayStarted
  pageVisibility.set(false)
  await tick()
  now = 10_000
  pageVisibility.set(true)

  const snapshot = await running
  assert.equal(snapshot.phase, "terminal")
  assert.equal(attempts, 3)
  assert.equal(pageVisibility.listenerCount(), 0)
})

test("already-aborted controllers perform no transport work", async () => {
  let attempts = 0
  const abortController = new AbortController()
  abortController.abort()
  const controller = transportController({
    request: request(),
    io: immediateIo({
      async attempt() {
        attempts += 1
        return { kind: "closed", accepted: true }
      },
      async readEvents() {
        return eventsPage()
      },
      async readState() {
        return state("running")
      },
    }),
  })

  const snapshot = await controller.start(abortController.signal)
  assert.equal(snapshot.phase, "cancelled")
  assert.equal(attempts, 0)
})

test("retry policy clamps jitter and run identities satisfy replay marker rules", () => {
  const policy = {
    reconnectInitialDelayMs: 250,
    reconnectBackoffFactor: 2,
    reconnectMaxDelayMs: 10_000,
    reconnectJitterRatio: 0.2,
    reconnectMaxAttempts: 8,
    reconnectMaxElapsedMs: 30_000,
    connectionTimeoutMs: 15_000,
    confirmationTimeoutMs: 30_000,
    reconciliationRequestTimeoutMs: 15_000,
    reconciliationPageLimit: 100,
    reconciliationMaxPages: 10,
    fallbackPollIntervalMs: 2_000,
    fallbackMaxConsecutiveFailures: 8,
  }
  assert.equal(mutationReconnectDelay(1, policy, 0), 200)
  assert.equal(mutationReconnectDelay(1, policy, 1), 300)
  assert.equal(mutationReconnectDelay(8, policy, 1), 10_000)
  assert.throws(() => mutationReconnectDelay(1, policy, Number.NaN), /must be finite/)

  const sessionId = createRunSessionId(() => "12345678-1234-1234-1234-123456789abc")
  assert.equal(sessionId, "session-12345678-1234-1234-1234-123456789abc")
  assert.deepEqual(newRunReplayBaseline(sessionId), { marker: `${sessionId}:`, sequence: 0 })
  assert.throws(() => createRunSessionId(() => "invalid id"), /server contract/)
  assert.equal(
    createMutationId(() => "12345678-1234-1234-1234-123456789abc"),
    "mutation-12345678-1234-1234-1234-123456789abc",
  )
  assert.throws(() => createMutationId(() => "invalid id"), /server contract/)

  assert.throws(
    () =>
      new MutationTransportController({
        request: request({ mutationId: "invalid id" }),
        io: immediateIo(),
      }),
    /mutationId/,
  )

  assert.throws(
    () =>
      new MutationTransportController({
        request: request({ path: "//external.invalid/run" }),
        io: immediateIo(),
      }),
    /root-relative path/,
  )
  assert.throws(
    () =>
      new MutationTransportController({
        request: request({ baseline: { marker: `${SESSION_ID}:event\n1`, sequence: 1 } }),
        io: immediateIo(),
      }),
    /invalid event id/,
  )
  assert.throws(
    () =>
      new MutationTransportController({
        request: request({ baseline: { marker: `${SESSION_ID}:`, sequence: 1 } }),
        io: immediateIo(),
      }),
    /start markers require sequence zero/,
  )
  assert.throws(
    () =>
      new MutationTransportController({
        request: request({ baseline: { marker: `${SESSION_ID}:event-1`, sequence: 0 } }),
        io: immediateIo(),
      }),
    /durable event markers require a positive sequence/,
  )
})
