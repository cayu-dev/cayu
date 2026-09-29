// Behavior tests for the browser client served by mount_cayu at {path}/client.js.
// Run with: node --test tests/browser_client/*.test.mjs
import assert from "node:assert/strict"
import { afterEach, beforeEach, mock, test } from "node:test"

import {
  CayuClientError,
  CLIENT_VERSION,
  CONTRACT_VERSION,
  connect,
} from "../../src/cayu/server/browser_client/client.js"

const API = "http://cayu.test/cayu/api"
const START = 1_000_000

class FakeDocument extends EventTarget {
  hidden = false

  setHidden(hidden) {
    this.hidden = hidden
    this.dispatchEvent(new Event("visibilitychange"))
  }
}

function enabled() {
  return { configured: true, read: { enabled: true, unavailable_reason: null } }
}

// Mirrors the session follow contract advertised by the server (#1926).
function sessionFollowContract(pathTemplate = "/cayu/api/sessions/{session_id}/events/stream") {
  return {
    method: "GET",
    path_template: pathTemplate,
    event_id_format: "session_id:cayu_event_<sequence>",
    start_query_param: "after_sequence",
    resume_header: "Last-Event-ID",
    unknown_event_marker_behavior: "reject",
    filter_query_params: ["event_type", "exclude_event_type", "interaction_id"],
    event_data_schema: "SseEventEnvelope",
    end_event_name: "end",
    end_data_schema: "SseSessionFollowEndEnvelope",
    terminal_behavior: "end_after_terminal_event",
    limit_exceeded_status: 429,
  }
}

function makeContract({
  stream = false,
  follow = sessionFollowContract(),
  usage = true,
  contractVersion = CONTRACT_VERSION,
  clientVersion = CLIENT_VERSION,
} = {}) {
  const sse = { content_type: "text/event-stream" }
  const surfaces = {
    usage: usage
      ? enabled()
      : { configured: false, read: { enabled: false, unavailable_reason: "not_configured" } },
  }
  if (stream) {
    sse.session_follow = follow
    surfaces.session_follow = enabled()
  }
  return {
    api_prefix: "/cayu/api",
    contract_version: contractVersion,
    sse,
    client: {
      module_url: "/cayu/client.js",
      types_url: "/cayu/client.d.ts",
      version: clientVersion,
      guide_topic: "app-ui",
    },
    capabilities: { surfaces },
  }
}

function record(sequence, type = "model.text.delta") {
  return {
    sequence,
    id: `event_${sequence}`,
    type,
    session_id: "s1",
    interaction_id: null,
    agent_name: "assistant",
    environment_name: null,
    workflow_name: null,
    tool_name: null,
    payload: {},
    timestamp: "2026-09-01T00:00:00+00:00",
  }
}

function json(body, { status = 200, headers = {} } = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  })
}

/** SSE response body following the session follow stream frame format. */
class FakeStream {
  constructor(request) {
    this.request = request
    this.aborted = false
    this.closed = false
    const encoder = new TextEncoder()
    this.body = new ReadableStream({
      start: (controller) => {
        this.push = (text) => this.pushBytes(encoder.encode(text))
        this.pushBytes = (bytes) => {
          if (!this.closed && !this.aborted) controller.enqueue(bytes)
        }
        this.finish = () => {
          if (this.closed || this.aborted) return
          this.closed = true
          controller.close()
        }
        request.init.signal?.addEventListener("abort", () => {
          if (this.closed) return
          this.aborted = true
          controller.error(new DOMException("The operation was aborted.", "AbortError"))
        })
      },
      cancel: () => {
        this.closed = true
      },
    })
  }

  event(sequence, type = "model.text.delta") {
    const { sequence: _sequence, ...envelope } = record(sequence, type)
    this.push(`id: s1:cayu_event_${sequence}\ndata: ${JSON.stringify(envelope)}\n\n`)
  }

  heartbeat() {
    this.push(": heartbeat\n\n")
  }

  end(status) {
    const data = { type: "session.follow.end", session_id: "s1", status }
    this.push(`event: end\ndata: ${JSON.stringify(data)}\n\n`)
    this.finish()
  }
}

class FakeServer {
  constructor(contract = makeContract()) {
    this.contract = contract
    this.requests = []
    this.overrides = []
    this.streams = []
    this.events = []
    this.status = "running"
    this.sessions = []
    this.usage = { session_id: "s1", total_tokens: "12" }
    this.usageEtag = '"usage-1"'
  }

  /** Answer the next request whose path ends with `suffix` with `respond()`. */
  once(suffix, respond) {
    this.overrides.push({ suffix, respond })
  }

  fetch = async (input, init = {}) => {
    const url = new URL(String(input))
    const request = { url, path: url.pathname, query: url.searchParams, init, at: Date.now() }
    request.headers = new Headers(init.headers)
    this.requests.push(request)
    if (init.signal?.aborted) throw new DOMException("The operation was aborted.", "AbortError")
    const override = this.overrides.findIndex((item) => request.path.endsWith(item.suffix))
    if (override !== -1) {
      const [item] = this.overrides.splice(override, 1)
      return item.respond(request)
    }
    const path = request.path.slice("/cayu/api".length)
    if (path === "/contract") return json(this.contract)
    if (path === "/sessions/s1") return json({ id: "s1", status: this.status })
    if (path === "/sessions/s1/events") return json(this.eventPage(request.query))
    if (path === "/sessions/s1/events/stream" || path === "/custom/s1/follow") {
      const stream = new FakeStream(request)
      this.streams.push(stream)
      return new Response(stream.body, { headers: { "content-type": "text/event-stream" } })
    }
    if (path === "/sessions/s1/usage") {
      if (request.headers.get("If-None-Match") === this.usageEtag) {
        return new Response(null, { status: 304, headers: { ETag: this.usageEtag } })
      }
      return json(this.usage, { headers: { ETag: this.usageEtag } })
    }
    if (path === "/sessions") {
      const status = request.query.get("status")
      return json({
        sessions: this.sessions.filter((session) => status === null || session.status === status),
        next_cursor: null,
        total_count: null,
      })
    }
    return json({ detail: "Not Found" }, { status: 404 })
  }

  eventPage(query) {
    const after = query.has("after_sequence") ? Number(query.get("after_sequence")) : null
    const limit = Number(query.get("limit") ?? 100)
    const excluded = query.get("exclude_event_type")
    const unseen = this.events.filter((event) => after === null || event.sequence > after)
    const matching = unseen.filter((event) => event.type !== excluded)
    const page = matching.slice(0, limit)
    const hasMore = matching.length > limit
    const scanned = hasMore
      ? page.at(-1).sequence
      : Math.max(after ?? 0, ...unseen.map((event) => event.sequence))
    return {
      session_id: "s1",
      events: page,
      order_by: "sequence_asc",
      next_sequence: page.at(-1)?.sequence ?? after,
      scan_through_sequence: after === null && unseen.length === 0 ? null : scanned,
      has_more: hasMore,
    }
  }

  requestsTo(suffix) {
    return this.requests.filter((request) => request.path.endsWith(suffix))
  }
}

async function flush() {
  for (let round = 0; round < 30; round += 1) {
    await new Promise((resolve) => setImmediate(resolve))
  }
}

async function advance(milliseconds, step = 250) {
  for (let elapsed = 0; elapsed < milliseconds; elapsed += step) {
    mock.timers.tick(Math.min(step, milliseconds - elapsed))
    await flush()
  }
}

async function assertQuiet(server, milliseconds = 10 * 60_000) {
  const count = server.requests.length
  await advance(milliseconds, 5_000)
  assert.equal(server.requests.length, count, "no request may follow")
}

let document

beforeEach(() => {
  mock.timers.enable({ apis: ["setTimeout", "Date"], now: START })
  mock.method(Math, "random", () => 0.5)
  document = new FakeDocument()
  globalThis.document = document
})

afterEach(() => {
  mock.timers.reset()
  mock.restoreAll()
  delete globalThis.document
})

test("connect reads the contract with same-origin credentials", async () => {
  const server = new FakeServer()
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })

  assert.equal(server.requests.length, 1)
  assert.equal(server.requests[0].url.href, `${API}/contract`)
  assert.equal(server.requests[0].init.credentials, "same-origin")
  assert.equal(client.contract.contract_version, CONTRACT_VERSION)
  assert.equal(client.streamAvailable, false)
})

test("connect defaults to the API next to the module", async () => {
  const urls = []
  await connect({
    fetch: async (input) => {
      urls.push(String(input))
      return json(makeContract())
    },
  })

  assert.match(urls[0], /\/src\/cayu\/server\/browser_client\/api\/contract$/)
})

test("connect rejects a different contract or client version", async () => {
  for (const contract of [
    makeContract({ contractVersion: "999" }),
    makeContract({ clientVersion: "999" }),
  ]) {
    const server = new FakeServer(contract)
    await assert.rejects(
      connect({ apiBaseUrl: API, fetch: server.fetch }),
      (error) =>
        error instanceof CayuClientError && error.kind === "contract" && /v999/.test(error.message),
    )
  }
})

test("connect reports an authentication failure", async () => {
  const server = new FakeServer()
  server.once("/contract", () => json({ detail: "Not authenticated" }, { status: 401 }))

  await assert.rejects(
    connect({ apiBaseUrl: API, fetch: server.fetch }),
    (error) =>
      error.kind === "auth" && error.status === 401 && /Not authenticated/.test(error.message),
  )
})

test("followSession streams events once, in order, and stops at the end frame", async () => {
  const server = new FakeServer(makeContract({ stream: true }))
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  const seen = []
  const statuses = []
  const follow = client.followSession("s1", {
    excludeEventTypes: ["model.text.delta"],
    onEvent: (event) => seen.push(`${event.sequence}:${event.type}`),
    onStatus: (status) => statuses.push(status),
  })
  await flush()

  assert.deepEqual(
    server.requests.map((request) => request.path),
    ["/cayu/api/contract", "/cayu/api/sessions/s1", "/cayu/api/sessions/s1/events/stream"],
  )
  const [stream] = server.streams
  assert.equal(stream.request.headers.get("Last-Event-ID"), null)
  assert.equal(stream.request.query.get("after_sequence"), null)
  assert.equal(stream.request.query.get("exclude_event_type"), "model.text.delta")
  assert.equal(stream.request.headers.get("Accept"), "text/event-stream")
  stream.event(1, "session.started")
  stream.heartbeat()
  stream.event(2)
  stream.event(2)
  stream.event(3, "tool.completed")
  stream.event(4, "session.completed")
  stream.end("completed")

  assert.deepEqual(await follow.done, { reason: "terminal", status: "completed", lastSequence: 4 })
  assert.deepEqual(seen, ["1:session.started", "3:tool.completed", "4:session.completed"])
  assert.deepEqual(statuses, ["running", "completed"])
  assert.equal(follow.transport, "stream")
  await assertQuiet(server)
})

test("a hidden tab closes the stream, sends nothing, and resumes after the last sequence", async () => {
  const server = new FakeServer(makeContract({ stream: true }))
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  const seen = []
  const follow = client.followSession("s1", { onEvent: (event) => seen.push(event.sequence) })
  await flush()
  server.streams[0].event(1, "session.started")
  server.streams[0].event(2)
  await flush()

  document.setHidden(true)
  await flush()
  assert.equal(server.streams[0].aborted, true)
  await assertQuiet(server, 5 * 60_000)

  document.setHidden(false)
  await flush()
  assert.equal(server.streams.length, 2)
  const resumed = server.streams[1]
  assert.equal(resumed.request.query.get("after_sequence"), "2")
  assert.equal(resumed.request.headers.get("Last-Event-ID"), null)
  resumed.event(2)
  resumed.event(3)
  resumed.event(4, "session.completed")
  resumed.end("completed")

  assert.equal((await follow.done).reason, "terminal")
  assert.deepEqual(seen, [1, 2, 3, 4])
  await assertQuiet(server)
})

test("followSession polls after_sequence when the stream is not advertised", async () => {
  const server = new FakeServer()
  server.events = [record(1, "session.started"), record(2), record(3, "tool.completed")]
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  const seen = []
  const follow = client.followSession("s1", {
    excludeEventTypes: ["model.text.delta", "session.completed"],
    onEvent: (event) => seen.push(`${event.sequence}:${event.type}`),
  })
  await flush()

  const [first, ...rest] = server.requestsTo("/events")
  assert.equal(first.query.get("after_sequence"), null)
  // One non-lifecycle type is excluded on the server; lifecycle types never are.
  assert.equal(first.query.get("exclude_event_type"), "model.text.delta")
  assert.ok(rest.every((request) => request.query.get("after_sequence") === "3"))
  assert.deepEqual(seen, ["1:session.started", "3:tool.completed"])

  server.events.push(record(4), record(5, "session.completed"))
  server.status = "completed"
  await advance(2_000)

  assert.deepEqual(await follow.done, { reason: "terminal", status: "completed", lastSequence: 5 })
  assert.deepEqual(seen, ["1:session.started", "3:tool.completed"])
  assert.equal(follow.transport, "poll")
  await assertQuiet(server)
})

test("a terminal session drains its remaining events without opening a stream", async () => {
  const server = new FakeServer(makeContract({ stream: true }))
  server.status = "completed"
  server.events = [record(1, "session.started"), record(2), record(3, "session.completed")]
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  const seen = []
  const follow = client.followSession("s1", {
    afterSequence: 1,
    onEvent: (event) => seen.push(event.sequence),
  })

  assert.deepEqual(await follow.done, { reason: "terminal", status: "completed", lastSequence: 3 })
  assert.deepEqual(seen, [2, 3])
  assert.equal(server.streams.length, 0)
  assert.equal(server.requestsTo("/events")[0].query.get("after_sequence"), "1")
  await assertQuiet(server)
})

test("polling sends nothing while hidden and resumes without a gap", async () => {
  const server = new FakeServer()
  server.events = [record(1, "session.started")]
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  const seen = []
  const follow = client.followSession("s1", { onEvent: (event) => seen.push(event.sequence) })
  await flush()
  assert.deepEqual(seen, [1])

  document.setHidden(true)
  server.events.push(record(2), record(3))
  await assertQuiet(server, 5 * 60_000)

  document.setHidden(false)
  await flush()
  const resumed = server.requestsTo("/events").at(-1)
  assert.equal(resumed.query.get("after_sequence"), "1")
  assert.deepEqual(seen, [1, 2, 3])

  follow.stop()
  assert.equal((await follow.done).reason, "aborted")
  await assertQuiet(server)
})

test("polling backs off while nothing changes and resets on new events", async () => {
  const server = new FakeServer()
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  const follow = client.followSession("s1")
  await advance(100_000, 100)

  const times = server.requestsTo("/events").map((request) => request.at)
  const gaps = times.slice(1).map((time, index) => time - times[index])
  assert.deepEqual(gaps.slice(0, 6), [2_000, 4_000, 8_000, 16_000, 30_000, 30_000])

  server.events.push(record(1, "tool.completed"))
  const before = server.requestsTo("/events").length
  await advance(30_000, 100)
  const after = server.requestsTo("/events").slice(before - 1)
  assert.equal(after[1].at - after[0].at <= 30_000, true)
  assert.equal(after[2].at - after[1].at, 2_000)
  follow.stop()
  await follow.done
})

test("retries honor Retry-After and otherwise back off exponentially", async () => {
  const server = new FakeServer()
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  server.once("/events", () =>
    json({ detail: "busy" }, { status: 503, headers: { "Retry-After": "7" } }),
  )
  server.once("/events", () => json({ detail: "busy" }, { status: 503 }))
  const follow = client.followSession("s1")
  await flush()
  assert.equal(server.requestsTo("/events").length, 1)

  await advance(6_900, 100)
  assert.equal(server.requestsTo("/events").length, 1)
  await advance(200, 100)
  assert.equal(server.requestsTo("/events").length, 2)
  // Second consecutive failure without Retry-After: 2 s (jitter is centered by the mock).
  await advance(1_800, 100)
  assert.equal(server.requestsTo("/events").length, 2)
  await advance(200, 100)
  assert.equal(server.requestsTo("/events").length, 3)
  assert.equal(follow.lastError.status, 503)
  follow.stop()
  await follow.done
})

test("an authentication failure stops following and reports why", async () => {
  const server = new FakeServer()
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  server.once("/events", () => json({ detail: "Session expired" }, { status: 401 }))
  const errors = []
  const follow = client.followSession("s1", { onError: (error) => errors.push(error) })

  await assert.rejects(
    follow.done,
    (error) => error.kind === "auth" && /Session expired/.test(error.message),
  )
  assert.equal(errors.length, 1)
  assert.equal(errors[0].status, 401)
  await assertQuiet(server)
})

test("an invalid response is a contract failure that stops following", async () => {
  const server = new FakeServer()
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  server.once("/events", () => json({ unexpected: true }))
  const follow = client.followSession("s1")

  await assert.rejects(follow.done, (error) => error.kind === "contract")
  await assertQuiet(server)
})

test("an advertised stream that is unavailable falls back to polling", async () => {
  const server = new FakeServer(makeContract({ stream: true }))
  server.events = [record(1, "tool.completed")]
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  server.once("/events/stream", () => json({ detail: "Not Found" }, { status: 404 }))
  const seen = []
  const follow = client.followSession("s1", { onEvent: (event) => seen.push(event.sequence) })
  await flush()

  assert.equal(server.requestsTo("/events/stream").length, 1)
  assert.deepEqual(seen, [1])
  await advance(10_000)
  assert.equal(server.requestsTo("/events/stream").length, 1)
  assert.equal(follow.transport, "poll")
  follow.stop()
  await follow.done
})

/** Serialize one event frame the way the follow stream does, with a chosen line ending. */
function eventFrame(sequence, type, newline, { name = null, payload = {} } = {}) {
  const { sequence: _sequence, ...envelope } = { ...record(sequence, type), payload }
  const lines = [`id: s1:cayu_event_${sequence}`]
  if (name !== null) lines.push(`event: ${name}`)
  lines.push(`data: ${JSON.stringify(envelope)}`)
  return lines.join(newline) + newline + newline
}

function endFrame(status, newline) {
  const data = JSON.stringify({ type: "session.follow.end", session_id: "s1", status })
  return `event: end${newline}data: ${data}${newline}${newline}`
}

/** Follow s1 over a stream whose body is delivered as exactly `chunks`, then closed. */
async function followChunks(chunks) {
  const server = new FakeServer(makeContract({ stream: true }))
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  const seen = []
  const follow = client.followSession("s1", { onEvent: (event) => seen.push(event) })
  await flush()
  const [stream] = server.streams
  for (const chunk of chunks) {
    if (typeof chunk === "string") stream.push(chunk)
    else stream.pushBytes(chunk)
    await flush()
  }
  stream.finish()
  await flush()
  // A parser regression can leave the follow reconnecting instead of ending;
  // stop it so the assertions fail rather than the test hanging.
  follow.stop()
  return { result: await follow.done, seen, server }
}

function splitAfter(text, marker) {
  const index = text.indexOf(marker)
  assert.notEqual(index, -1, marker)
  return [text.slice(0, index + marker.length), text.slice(index + marker.length)]
}

test("a CRLF split right after the event id line keeps the event", async () => {
  const body =
    eventFrame(1, "model.text.delta", "\r\n", { name: "model.text.delta" }) +
    endFrame("completed", "\r\n")
  const { result, seen } = await followChunks(splitAfter(body, "cayu_event_1\r"))

  assert.deepEqual(result, { reason: "terminal", status: "completed", lastSequence: 1 })
  assert.deepEqual(
    seen.map((event) => event.sequence),
    [1],
  )
})

test("CRLF splits after event, data, and blank-line CRs keep every event", async () => {
  const body =
    eventFrame(1, "tool.completed", "\r\n") +
    eventFrame(2, "tool.completed", "\r\n", { name: "tool.completed" }) +
    endFrame("completed", "\r\n")
  // Split between CR and LF after the first data line, after the second
  // event's `event:` line, and inside the blank line that ends the second event.
  const cuts = [
    body.indexOf("}\r\n") + 2,
    body.indexOf("event: tool.completed\r") + "event: tool.completed\r".length,
    body.indexOf("\r\n\r\n", body.indexOf("cayu_event_2")) + 3,
  ]
  const chunks = [0, ...cuts].map((cut, index) => body.slice(cut, cuts[index]))
  assert.ok(chunks.slice(0, -1).every((chunk) => chunk.endsWith("\r")))
  assert.ok(chunks.slice(1).every((chunk) => chunk.startsWith("\n")))
  const { result, seen } = await followChunks(chunks)

  assert.deepEqual(result, { reason: "terminal", status: "completed", lastSequence: 2 })
  assert.deepEqual(
    seen.map((event) => event.sequence),
    [1, 2],
  )
})

test("a stream delivered one byte at a time parses the same as one chunk", async () => {
  const payload = { text: "héllo — ✓ 😀", nested: { values: [1, 2, 3] } }
  for (const newline of ["\r\n", "\n", "\r"]) {
    const body =
      ": heartbeat" +
      newline +
      newline +
      eventFrame(1, "tool.completed", newline, { payload }) +
      eventFrame(2, "tool.completed", newline, { name: "tool.completed", payload }) +
      endFrame("completed", newline)
    const bytes = new TextEncoder().encode(body)
    const chunks = [...bytes].map((byte) => Uint8Array.of(byte))
    const { result, seen } = await followChunks(chunks)

    assert.deepEqual(result, { reason: "terminal", status: "completed", lastSequence: 2 }, newline)
    assert.deepEqual(
      seen.map((event) => [event.sequence, event.payload]),
      [
        [1, payload],
        [2, payload],
      ],
    )
  }
})

test("a multi-byte character split across reads is decoded intact", async () => {
  const payload = { text: "😀 and ✓" }
  const body = eventFrame(1, "tool.completed", "\n", { payload }) + endFrame("completed", "\n")
  const bytes = new TextEncoder().encode(body)
  // The emoji is four bytes (F0 9F 98 80); split after its second byte.
  const emoji = bytes.indexOf(0xf0)
  const { seen } = await followChunks([bytes.slice(0, emoji + 2), bytes.slice(emoji + 2)])

  assert.deepEqual(seen[0].payload, payload)
})

test("a lone CR that is the stream's last byte still ends the final frame", async () => {
  const body = eventFrame(1, "tool.completed", "\r") + endFrame("completed", "\r")
  assert.ok(body.endsWith("}\r\r"))
  const { result, seen } = await followChunks(splitAfter(body, "}\r"))

  assert.deepEqual(result, { reason: "terminal", status: "completed", lastSequence: 1 })
  assert.equal(seen.length, 1)
})

test("a frame cut off by the end of the stream is not dispatched", async () => {
  const server = new FakeServer(makeContract({ stream: true }))
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  const follow = client.followSession("s1")
  await flush()
  const data = JSON.stringify({ type: "session.follow.end", session_id: "s1", status: "completed" })
  server.streams[0].push(`${eventFrame(1, "tool.completed", "\n")}event: end\ndata: ${data}\n`)
  server.streams[0].finish()
  await flush()

  // Without its blank line the end frame is incomplete, so the client reconnects.
  assert.equal(follow.status, "running")
  await advance(2_000)
  assert.equal(server.streams.length, 2)
  assert.equal(server.streams[1].request.query.get("after_sequence"), "1")
  follow.stop()
  assert.equal((await follow.done).reason, "aborted")
})

test("a stream error frame catches up by polling, then streams again", async () => {
  const server = new FakeServer(makeContract({ stream: true }))
  server.events = [record(1, "tool.completed"), record(2, "tool.completed")]
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  const seen = []
  const follow = client.followSession("s1", { onEvent: (event) => seen.push(event.sequence) })
  await flush()
  const [first] = server.streams
  first.event(1, "tool.completed")
  // For example an event too large for the live transport.
  first.push(
    `event: error\ndata: ${JSON.stringify({ type: "stream.error", error: "too large", retryable: false })}\n\n`,
  )
  await flush()

  const catchUp = server.requestsTo("/events")
  assert.equal(catchUp.length, 1)
  assert.equal(catchUp[0].query.get("after_sequence"), "1")
  assert.deepEqual(seen, [1, 2])
  assert.equal(server.streams.length, 2)
  assert.equal(server.streams[1].request.query.get("after_sequence"), "2")
  assert.equal(server.streams[1].request.headers.get("Last-Event-ID"), null)
  assert.equal(follow.lastError.kind, "protocol")
  follow.stop()
  await follow.done
})

test("the stream uses the advertised path_template and starts with after_sequence", async () => {
  const server = new FakeServer(
    makeContract({
      stream: true,
      follow: sessionFollowContract("/cayu/api/custom/{session_id}/follow"),
    }),
  )
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  const seen = []
  const follow = client.followSession("s1", {
    afterSequence: 0,
    excludeEventTypes: ["model.text.delta", "session.completed"],
    onEvent: (event) => seen.push(event.sequence),
  })
  await flush()

  const [stream] = server.streams
  assert.equal(stream.request.path, "/cayu/api/custom/s1/follow")
  // afterSequence 0 need not name an event, so it goes in the start query,
  // never in a synthesized Last-Event-ID marker. Lifecycle types stay unfiltered.
  assert.equal(stream.request.query.get("after_sequence"), "0")
  assert.equal(stream.request.query.get("exclude_event_type"), "model.text.delta")
  assert.equal(stream.request.headers.get("Last-Event-ID"), null)
  stream.event(1, "tool.completed")
  stream.event(2, "session.completed")
  stream.end("completed")

  assert.deepEqual(await follow.done, { reason: "terminal", status: "completed", lastSequence: 2 })
  assert.deepEqual(seen, [1])
  assert.equal(server.requestsTo("/events/stream").length, 0)
})

test("a session_follow entry without path_template uses the default stream path", async () => {
  const follow = sessionFollowContract()
  delete follow.path_template
  const server = new FakeServer(makeContract({ stream: true, follow }))
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  const handle = client.followSession("s1", { afterSequence: 4 })
  await flush()

  assert.equal(server.streams[0].request.path, "/cayu/api/sessions/s1/events/stream")
  assert.equal(server.streams[0].request.query.get("after_sequence"), "4")
  handle.stop()
  await handle.done
})

test("a stream limit response waits for Retry-After before reconnecting", async () => {
  const server = new FakeServer(makeContract({ stream: true }))
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  server.once("/events/stream", () =>
    json({ detail: "Too many follow streams" }, { status: 429, headers: { "Retry-After": "5" } }),
  )
  const follow = client.followSession("s1")
  await flush()
  assert.equal(server.requestsTo("/events/stream").length, 1)

  await advance(4_900, 100)
  assert.equal(server.requestsTo("/events/stream").length, 1)
  await advance(200, 100)
  assert.equal(server.streams.length, 1)
  assert.equal(follow.lastError.status, 429)
  follow.stop()
  await follow.done
})

function session(id, status, updatedAt = "2026-09-01T00:00:00+00:00") {
  return { id, status, agent_name: "assistant", updated_at: updatedAt }
}

test("watchSessions refreshes no faster than 15 s and backs off while unchanged", async () => {
  const server = new FakeServer()
  server.sessions = [session("a", "running")]
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  const changes = []
  const stop = client.watchSessions(
    { status: ["pending", "running"] },
    { intervalMs: 1_000, onChange: (sessions) => changes.push(sessions.map((item) => item.id)) },
  )
  await flush()
  assert.equal(server.requestsTo("/sessions")[0].at, START)
  await advance(240_000, 1_000)

  const refreshes = server
    .requestsTo("/sessions")
    .filter((r) => r.query.get("status") === "pending")
  assert.ok(
    server
      .requestsTo("/sessions")
      .every((r) => ["pending", "running"].includes(r.query.get("status"))),
  )
  const gaps = refreshes.slice(1).map((request, index) => request.at - refreshes[index].at)
  assert.deepEqual(gaps, [15_000, 30_000, 60_000, 120_000])
  assert.deepEqual(changes, [["a"]])

  // The next unchanged-backoff refresh, at 345 s, sees the change and resets to 15 s.
  server.sessions = [session("a", "running", "2026-09-01T00:05:00+00:00"), session("b", "pending")]
  await advance(105_000, 1_000)
  assert.deepEqual(changes, [["a"], ["a", "b"]])
  const count = server.requestsTo("/sessions").length
  await advance(14_000, 1_000)
  assert.equal(server.requestsTo("/sessions").length, count)
  await advance(1_000, 1_000)
  assert.equal(server.requestsTo("/sessions").length, count + 2)
  stop()
  await assertQuiet(server)
})

test("watchSessions stops when nothing is active", async () => {
  const server = new FakeServer()
  server.sessions = [session("a", "completed"), session("b", "failed")]
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  const changes = []
  const stops = []
  client.watchSessions(
    {},
    { onChange: (sessions) => changes.push(sessions), onStop: (reason) => stops.push(reason) },
  )
  await flush()

  assert.equal(changes.length, 1)
  assert.deepEqual(stops, ["idle"])
  assert.equal(server.requestsTo("/sessions")[0].query.get("status"), null)
  await assertQuiet(server)
})

test("watchSessions sends nothing while hidden", async () => {
  const server = new FakeServer()
  server.sessions = [session("a", "running")]
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  document.setHidden(true)
  const stop = client.watchSessions({ status: "running" }, {})
  await assertQuiet(server, 5 * 60_000)

  document.setHidden(false)
  await flush()
  assert.equal(server.requestsTo("/sessions").length, 1)
  document.setHidden(true)
  await assertQuiet(server, 5 * 60_000)
  stop()
})

test("watchSessions honors Retry-After and stops on authorization failures", async () => {
  const server = new FakeServer()
  server.sessions = [session("a", "running")]
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })
  server.once("/sessions", () =>
    json({ detail: "slow down" }, { status: 429, headers: { "Retry-After": "45" } }),
  )
  const errors = []
  const stops = []
  client.watchSessions(
    { status: "running" },
    { onError: (error) => errors.push(error), onStop: (reason) => stops.push(reason) },
  )
  await flush()
  assert.equal(server.requestsTo("/sessions").length, 1)
  await advance(44_000, 1_000)
  assert.equal(server.requestsTo("/sessions").length, 1)
  server.once("/sessions", () => json({ detail: "Forbidden" }, { status: 403 }))
  await advance(1_000, 1_000)
  assert.equal(server.requestsTo("/sessions").length, 2)

  assert.equal(errors.length, 1)
  assert.equal(errors[0].kind, "auth")
  assert.deepEqual(stops, ["error"])
  await assertQuiet(server)
})

test("getSessionUsage revalidates with the usage ETag", async () => {
  const server = new FakeServer()
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })

  assert.deepEqual(await client.getSessionUsage("s1"), server.usage)
  assert.deepEqual(await client.getSessionUsage("s1"), server.usage)
  const [first, second] = server.requestsTo("/usage")
  assert.equal(first.headers.get("If-None-Match"), null)
  assert.equal(second.headers.get("If-None-Match"), '"usage-1"')
})

test("getSessionUsage does not request a surface the contract reports unavailable", async () => {
  const server = new FakeServer(makeContract({ usage: false }))
  const client = await connect({ apiBaseUrl: API, fetch: server.fetch })

  await assert.rejects(client.getSessionUsage("s1"), (error) => error.kind === "unavailable")
  assert.equal(server.requestsTo("/usage").length, 0)
})
