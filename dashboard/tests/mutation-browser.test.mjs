import assert from "node:assert/strict"
import test from "node:test"

globalThis.window = { __CAYU_DASHBOARD_CONFIG__: { apiBaseUrl: "/control/api" } }

const {
  createMutationBrowserIO,
  executeResumeMutation,
  fetchMutationReplayBaseline,
  MUTATION_SSE_DATA_MAX_BYTES,
} = await import("../src/lib/mutation-browser.ts")

const SESSION_ID = "session-browser"
const MUTATION_ID = "mutation-browser"

function attemptRequest(lastEventId = `${SESSION_ID}:event-1`) {
  return {
    path: "/resume",
    body: { session_id: SESSION_ID, prompt: "continue" },
    sessionId: SESSION_ID,
    mutationId: MUTATION_ID,
    lastEventId,
  }
}

function event(id = "event-2", type = "session.resumed", payload = {}) {
  return {
    id,
    type,
    session_id: SESSION_ID,
    agent_name: "assistant",
    environment_name: null,
    workflow_name: null,
    tool_name: null,
    payload,
    timestamp: "2026-07-15T00:00:00Z",
  }
}

function streamResponse(status = 200, contentType = "text/event-stream; charset=utf-8") {
  return new Response(status === 200 ? null : JSON.stringify({ detail: "request rejected" }), {
    status,
    headers: { "content-type": status === 200 ? contentType : "application/json" },
  })
}

function callbacks() {
  const accepted = []
  const events = []
  return {
    accepted,
    events,
    value: {
      onAccepted: () => accepted.push(true),
      onEvent: (value, id) => events.push({ value, id }),
    },
  }
}

test("browser adapter performs one explicit replay attempt with durable headers", async () => {
  let capturedUrl = null
  let capturedInit = null
  let fetchCalls = 0
  const observed = callbacks()
  const io = createMutationBrowserIO({
    fetchImpl: async () => {
      fetchCalls += 1
      return streamResponse()
    },
    async fetchEventSource(input, init) {
      capturedUrl = input
      capturedInit = init
      await init.onopen(streamResponse())
      init.onmessage({
        id: `${SESSION_ID}:event-2`,
        event: "",
        data: JSON.stringify(event()),
      })
    },
  })

  const result = await io.attempt(attemptRequest(), observed.value, new AbortController().signal)

  assert.deepEqual(result, { kind: "closed", accepted: true })
  assert.equal(capturedUrl, "/control/api/resume")
  assert.equal(capturedInit.method, "POST")
  assert.equal(capturedInit.openWhenHidden, true)
  assert.equal(capturedInit.headers["Cayu-Mutation-ID"], MUTATION_ID)
  assert.equal(capturedInit.headers["Last-Event-ID"], `${SESSION_ID}:event-1`)
  assert.equal(capturedInit.headers.Accept, "text/event-stream")
  assert.deepEqual(JSON.parse(capturedInit.body), {
    session_id: SESSION_ID,
    prompt: "continue",
  })
  assert.deepEqual(observed.accepted, [true])
  assert.deepEqual(observed.events, [{ value: event(), id: `${SESSION_ID}:event-2` }])
  assert.equal(fetchCalls, 0, "the injected fetch is passed through, not called by the test stub")
})

test("browser adapter omits Last-Event-ID only for the initial mutation attempt", async () => {
  let headers = null
  const io = createMutationBrowserIO({
    async fetchEventSource(_input, init) {
      headers = init.headers
      await init.onopen(streamResponse())
    },
  })

  const result = await io.attempt(
    attemptRequest(null),
    callbacks().value,
    new AbortController().signal,
  )

  assert.deepEqual(result, { kind: "closed", accepted: true })
  assert.equal("Last-Event-ID" in headers, false)
  assert.equal(headers["Cayu-Mutation-ID"], MUTATION_ID)
})

test("browser adapter preserves typed runtime and observer outcomes", async () => {
  for (const [kind, code, retryable, expectedKind] of [
    ["runtime", "runtime_failed", false, "runtime_failed"],
    ["observer", "observer_lagged", true, "observer_failed"],
  ]) {
    const io = createMutationBrowserIO({
      async fetchEventSource(_input, init) {
        await init.onopen(streamResponse())
        init.onmessage({
          id: "",
          event: "error",
          data: JSON.stringify({
            type: "stream.error",
            kind,
            code,
            error: `${kind} failed`,
            error_type: "ContractError",
            retryable,
            session_id: SESSION_ID,
          }),
        })
      },
    })

    const result = await io.attempt(
      attemptRequest(),
      callbacks().value,
      new AbortController().signal,
    )

    assert.equal(result.kind, expectedKind)
    assert.equal(result.accepted, true)
    assert.equal(result.error.kind, kind)
    assert.equal(result.error.retryable, retryable)
  }
})

test("browser adapter separates HTTP, protocol, network, and cancellation failures", async () => {
  const cases = [
    {
      expected: "http_failed",
      adapter: async (_input, init) => init.onopen(streamResponse(409)),
    },
    {
      expected: "protocol_failed",
      adapter: async (_input, init) => init.onopen(streamResponse(200, "application/json")),
    },
    {
      expected: "network_failed",
      adapter: async () => {
        throw new TypeError("socket closed")
      },
    },
  ]

  for (const { expected, adapter } of cases) {
    const io = createMutationBrowserIO({ fetchEventSource: adapter })
    const result = await io.attempt(
      attemptRequest(),
      callbacks().value,
      new AbortController().signal,
    )
    assert.equal(result.kind, expected)
  }

  const controller = new AbortController()
  controller.abort()
  let attempted = false
  const io = createMutationBrowserIO({
    async fetchEventSource() {
      attempted = true
    },
  })
  const result = await io.attempt(attemptRequest(), callbacks().value, controller.signal)
  assert.deepEqual(result, { kind: "aborted", accepted: false })
  assert.equal(attempted, false)

  const activeController = new AbortController()
  const activeIo = createMutationBrowserIO({
    async fetchEventSource(_input, init) {
      await init.onopen(streamResponse())
      activeController.abort()
    },
  })
  const activeResult = await activeIo.attempt(
    attemptRequest(),
    callbacks().value,
    activeController.signal,
  )
  assert.deepEqual(activeResult, { kind: "aborted", accepted: true })
})

test("browser adapter requires the exact event-stream media type", async () => {
  for (const contentType of ["text/event-streaming", "text/event-stream+json"]) {
    const io = createMutationBrowserIO({
      fetchEventSource: async (_input, init) => init.onopen(streamResponse(200, contentType)),
    })
    const result = await io.attempt(
      attemptRequest(),
      callbacks().value,
      new AbortController().signal,
    )
    assert.equal(result.kind, "protocol_failed")
    assert.equal(result.accepted, false)
  }

  const io = createMutationBrowserIO({
    fetchEventSource: async (_input, init) =>
      init.onopen(streamResponse(200, "Text/Event-Stream; charset=utf-8")),
  })
  const result = await io.attempt(attemptRequest(), callbacks().value, new AbortController().signal)
  assert.deepEqual(result, { kind: "closed", accepted: true })
})

test("browser adapter rejects oversized and malformed SSE data before accumulation", async () => {
  for (const data of [
    "x".repeat(MUTATION_SSE_DATA_MAX_BYTES + 1),
    JSON.stringify({ ...event(), payload: [] }),
  ]) {
    const io = createMutationBrowserIO({
      async fetchEventSource(_input, init) {
        await init.onopen(streamResponse())
        init.onmessage({ id: `${SESSION_ID}:event-2`, event: "", data })
      },
    })
    const result = await io.attempt(
      attemptRequest(),
      callbacks().value,
      new AbortController().signal,
    )
    assert.equal(result.kind, "protocol_failed")
    assert.equal(result.accepted, true)
  }
})

test("browser adapter bounds unterminated raw SSE frames before parser accumulation", async () => {
  let fetchCalls = 0
  const io = createMutationBrowserIO({
    fetchImpl: async () => {
      fetchCalls += 1
      return new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new Uint8Array(MUTATION_SSE_DATA_MAX_BYTES).fill(0x78))
            controller.enqueue(new Uint8Array([0x78]))
            controller.close()
          },
        }),
        { headers: { "content-type": "text/event-stream" } },
      )
    },
    async fetchEventSource(input, init) {
      const response = await init.fetch(input, init)
      await init.onopen(response)
      const reader = response.body.getReader()
      while (!(await reader.read()).done) {
        // The bounded response must fail before an SSE parser can retain the
        // complete unterminated line.
      }
    },
  })

  const result = await io.attempt(attemptRequest(), callbacks().value, new AbortController().signal)

  assert.equal(fetchCalls, 1)
  assert.equal(result.kind, "protocol_failed")
  assert.equal(result.accepted, true)
  assert.match(result.error.message, /frame exceeded/)
})

test("browser adapter resets its raw byte budget at SSE frame boundaries", async () => {
  const frameSize = Math.floor(MUTATION_SSE_DATA_MAX_BYTES / 2) + 1
  const io = createMutationBrowserIO({
    fetchImpl: async () =>
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new Uint8Array(frameSize).fill(0x78))
            controller.enqueue(new Uint8Array([0x0a, 0x0a]))
            controller.enqueue(new Uint8Array(frameSize).fill(0x79))
            controller.enqueue(new Uint8Array([0x0d, 0x0a, 0x0d, 0x0a]))
            controller.close()
          },
        }),
        { headers: { "content-type": "text/event-stream" } },
      ),
    async fetchEventSource(input, init) {
      const response = await init.fetch(input, init)
      await init.onopen(response)
      const reader = response.body.getReader()
      while (!(await reader.read()).done) {
        // The combined stream exceeds the ceiling, but each frame remains bounded.
      }
    },
  })

  const result = await io.attempt(attemptRequest(), callbacks().value, new AbortController().signal)

  assert.deepEqual(result, { kind: "closed", accepted: true })
})

test("browser adapter opts out of fetch-event-source implicit retries", async () => {
  let onerrorThrew = false
  const failure = new TypeError("connection reset")
  const io = createMutationBrowserIO({
    async fetchEventSource(_input, init) {
      try {
        init.onerror(failure)
      } catch (error) {
        onerrorThrew = error === failure
        throw error
      }
    },
  })

  const result = await io.attempt(attemptRequest(), callbacks().value, new AbortController().signal)

  assert.equal(onerrorThrew, true)
  assert.equal(result.kind, "network_failed")
  assert.equal(result.accepted, false)
})

test("existing-session replay baseline reads are bounded and cancel their request", async () => {
  let requestSignal = null
  await assert.rejects(
    fetchMutationReplayBaseline(SESSION_ID, {
      timeoutMs: 5,
      readEvents: async (_sessionId, _query, signal) => {
        requestSignal = signal
        return new Promise(() => undefined)
      },
    }),
    {
      name: "MutationReplayBaselineTimeoutError",
      message: "Timed out while reading the durable mutation replay baseline.",
    },
  )
  assert.equal(requestSignal.aborted, true)
})

test("existing-session replay baseline honors caller cancellation before reading", async () => {
  const controller = new AbortController()
  controller.abort()
  let reads = 0

  await assert.rejects(
    fetchMutationReplayBaseline(SESSION_ID, {
      signal: controller.signal,
      readEvents: async () => {
        reads += 1
        throw new Error("must not read")
      },
    }),
    { name: "AbortError" },
  )
  assert.equal(reads, 0)
})

test("existing-session replay baseline cancels an active read with its caller", async () => {
  const controller = new AbortController()
  let requestSignal = null
  const baseline = fetchMutationReplayBaseline(SESSION_ID, {
    signal: controller.signal,
    readEvents: async (_sessionId, _query, signal) => {
      requestSignal = signal
      return new Promise(() => undefined)
    },
  })

  controller.abort()
  await assert.rejects(baseline, { name: "AbortError" })
  assert.equal(requestSignal.aborted, true)
})

test("a bounded baseline failure prevents an existing-session mutation POST", async () => {
  let mutationPosts = 0
  await assert.rejects(
    executeResumeMutation(
      { session_id: SESSION_ID, prompt: "continue" },
      {
        policy: { reconciliationRequestTimeoutMs: 5 },
        dependencies: {
          async readSessionEvents() {
            return new Promise(() => undefined)
          },
          async fetchEventSource() {
            mutationPosts += 1
          },
        },
      },
    ),
    { name: "MutationReplayBaselineTimeoutError" },
  )
  assert.equal(mutationPosts, 0)
})

test("existing-session replay baseline preserves the latest durable marker", async () => {
  let captured = null
  const baseline = await fetchMutationReplayBaseline(SESSION_ID, {
    readEvents: async (sessionId, query, signal) => {
      captured = { sessionId, query, signal }
      return {
        session_id: sessionId,
        events: [{ id: "event-latest", sequence: 42 }],
        next_cursor: null,
        has_more: false,
        scan_through_sequence: 42,
      }
    },
  })

  assert.deepEqual(baseline, { marker: `${SESSION_ID}:event-latest`, sequence: 42 })
  assert.equal(captured.sessionId, SESSION_ID)
  assert.deepEqual(captured.query, { order_by: "sequence_desc", limit: 1 })
  assert.equal(captured.signal.aborted, true, "completed reads release their internal signal")
})
