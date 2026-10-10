// @ts-self-types="./client.d.ts"
/**
 * Cayu browser client.
 *
 * A dependency-free ES module served by `mount_cayu` at `{path}/client.js`.
 * It follows sessions and watches session lists with bounded, visibility-aware
 * request behavior so custom UIs do not have to hand-write polling loops.
 * `cayu guide app-ui` describes the rules this module implements.
 */

/** Version of this module's public API, advertised as `client.version` by `/api/contract`. */
export const CLIENT_VERSION = "1"
/** Exact server contract version this module was released with. */
export const CONTRACT_VERSION = "49"

const TERMINAL_STATUSES = new Set(["completed", "failed", "interrupted"])
const ACTIVE_STATUSES = new Set(["pending", "running", "interrupting"])
const TERMINAL_EVENT_TYPES = new Set(["session.completed", "session.failed", "session.interrupted"])
const LIFECYCLE_EVENT_TYPES = new Set([
  ...TERMINAL_EVENT_TYPES,
  "session.started",
  "session.resumed",
])

/** Minimum spacing between `watchSessions` refreshes. It cannot be lowered. */
export const MIN_WATCH_INTERVAL_MS = 15_000

const STREAM_HEALTHY_AFTER_MS = 10_000

const DEFAULTS = Object.freeze({
  followPollIntervalMs: 2_000,
  followMaxPollIntervalMs: 30_000,
  watchIntervalMs: MIN_WATCH_INTERVAL_MS,
  watchMaxIntervalMs: 120_000,
  retryInitialDelayMs: 1_000,
  retryMaxDelayMs: 60_000,
  retryAfterMaxMs: 600_000,
  jitterRatio: 0.2,
  eventPageLimit: 200,
  watchPageLimit: 100,
  maxStreamFailures: 3,
})

/**
 * Error reported by the client. `kind` is one of `auth`, `contract`,
 * `not_found`, `unavailable`, `http`, `network`, or `protocol`. Only
 * `retryable` errors are retried; the others stop the operation.
 */
export class CayuClientError extends Error {
  constructor(
    kind,
    message,
    { status = null, retryable = false, retryAfterMs = null, cause } = {},
  ) {
    super(message, cause === undefined ? undefined : { cause })
    this.name = "CayuClientError"
    this.kind = kind
    this.status = status
    this.retryable = retryable
    this.retryAfterMs = retryAfterMs
  }
}

/**
 * Read `/api/contract`, check it against this module, and return a client.
 *
 * The API base defaults to `./api` next to this module, which is where
 * `mount_cayu` places it. Requests use same-origin credentials.
 */
export async function connect(options = {}) {
  const fetchImpl = options.fetch ?? ((input, init) => globalThis.fetch(input, init))
  const apiBase = trimSlash(String(options.apiBaseUrl ?? new URL("./api", import.meta.url)))
  const context = {
    apiBase,
    fetch: fetchImpl,
    headers: { ...(options.headers ?? {}) },
    credentials: options.credentials ?? "same-origin",
    contract: null,
    streamPath: null,
  }
  const contract = await requestJson(context, "/contract", { signal: options.signal })
  if (contract === null || typeof contract !== "object") {
    throw new CayuClientError("contract", "The server returned an invalid contract.")
  }
  if (contract.contract_version !== CONTRACT_VERSION) {
    throw new CayuClientError(
      "contract",
      `This client expects Cayu server contract v${CONTRACT_VERSION}, but the server reports ` +
        `v${String(contract.contract_version)}. Reload the page to load the server's client.js.`,
    )
  }
  const advertised = contract.client
  if (advertised && typeof advertised === "object" && advertised.version !== CLIENT_VERSION) {
    throw new CayuClientError(
      "contract",
      `The server advertises client.js v${String(advertised.version)}, but v${CLIENT_VERSION} ` +
        "is loaded. Reload the page to load the server's client.js.",
    )
  }
  context.contract = contract
  context.streamPath = sessionFollowStreamPath(contract)
  return new CayuClient(context)
}

class CayuClient {
  #context
  #usageCache = new Map()

  constructor(context) {
    this.#context = context
  }

  /** The `/api/contract` response this client was checked against. */
  get contract() {
    return this.#context.contract
  }

  /** Whether `followSession` will try the session follow stream first. */
  get streamAvailable() {
    return this.#context.streamPath !== null
  }

  followSession(sessionId, options = {}) {
    return followSession(this.#context, requireSessionId(sessionId), options)
  }

  watchSessions(filter = {}, options = {}) {
    return watchSessions(this.#context, filter ?? {}, options)
  }

  async getSessionUsage(sessionId, options = {}) {
    const id = requireSessionId(sessionId)
    const usage = this.#context.contract?.capabilities?.surfaces?.usage
    if (usage?.read?.enabled === false) {
      throw new CayuClientError(
        "unavailable",
        `Session usage is unavailable: ${String(usage.read.unavailable_reason ?? "disabled")}.`,
      )
    }
    const cached = this.#usageCache.get(id)
    const headers = cached ? { "If-None-Match": cached.etag } : {}
    const response = await send(this.#context, `/sessions/${encodeURIComponent(id)}/usage`, {
      signal: options.signal,
      headers,
    })
    if (response.status === 304 && cached) return cached.body
    if (!response.ok) throw await httpError(response)
    const body = await readJson(response)
    const etag = response.headers.get("ETag")
    if (etag) {
      this.#usageCache.delete(id)
      this.#usageCache.set(id, { etag, body })
      if (this.#usageCache.size > 100) this.#usageCache.delete(this.#usageCache.keys().next().value)
    }
    return body
  }
}

function followSession(context, sessionId, options) {
  const excluded = new Set(options.excludeEventTypes ?? [])
  // Lifecycle events are always fetched so the client can see a session end;
  // excluded lifecycle types are filtered here instead of on the server.
  const serverExclude = [...excluded].find((type) => !LIFECYCLE_EVENT_TYPES.has(type)) ?? null
  const basePollMs = positive(options.pollIntervalMs, DEFAULTS.followPollIntervalMs)
  const maxPollMs = Math.max(basePollMs, DEFAULTS.followMaxPollIntervalMs)
  const runner = new Runner(options.signal)
  const state = {
    lastSequence: normalizeSequence(options.afterSequence),
    status: null,
    transport: null,
    lastError: null,
  }

  const setStatus = (status) => {
    if (typeof status !== "string") {
      throw new CayuClientError("contract", "The server returned a session without a status.")
    }
    if (status === state.status) return
    state.status = status
    invoke(options.onStatus, status)
  }

  const acceptEvent = (event) => {
    if (state.lastSequence !== null && event.sequence <= state.lastSequence) return false
    state.lastSequence = event.sequence
    if (!excluded.has(event.type)) invoke(options.onEvent, event)
    return true
  }

  const readStatus = async () => {
    const session = await runner.run((signal) =>
      requestJson(context, `/sessions/${encodeURIComponent(sessionId)}`, { signal }),
    )
    setStatus(session?.status)
  }

  const pollPage = async () => {
    const params = new URLSearchParams({ limit: String(DEFAULTS.eventPageLimit) })
    if (state.lastSequence !== null) params.set("after_sequence", String(state.lastSequence))
    if (serverExclude !== null) params.set("exclude_event_type", serverExclude)
    const page = await runner.run((signal) =>
      requestJson(context, `/sessions/${encodeURIComponent(sessionId)}/events?${params}`, {
        signal,
      }),
    )
    if (!page || !Array.isArray(page.events)) {
      throw new CayuClientError("contract", "The server returned an invalid event page.")
    }
    let received = false
    let lifecycle = null
    for (const record of page.events) {
      if (!record || !Number.isSafeInteger(record.sequence) || typeof record.type !== "string") {
        throw new CayuClientError("contract", "The server returned an invalid event record.")
      }
      if (acceptEvent(record)) {
        received = true
        if (LIFECYCLE_EVENT_TYPES.has(record.type)) lifecycle = record.type
      }
    }
    const scanned = page.scan_through_sequence
    if (
      Number.isSafeInteger(scanned) &&
      (state.lastSequence === null || scanned > state.lastSequence)
    ) {
      state.lastSequence = scanned
    }
    return { hasMore: page.has_more === true, received, lifecycle }
  }

  const readStream = async () => {
    let lifecycle = null
    let receivedEvent = false
    let openedAt = null
    const outcome = await runner.run(async (signal) => {
      // Resume with the start query rather than Last-Event-ID: the stream
      // rejects markers that do not name an existing event, and lastSequence
      // may come from afterSequence or scan_through_sequence.
      const params = new URLSearchParams()
      if (state.lastSequence !== null) params.set("after_sequence", String(state.lastSequence))
      if (serverExclude !== null) params.set("exclude_event_type", serverExclude)
      const query = params.toString() ? `?${params}` : ""
      const path = context.streamPath.replace("{session_id}", encodeURIComponent(sessionId))
      const response = await send(context, `${path}${query}`, {
        signal,
        headers: { Accept: "text/event-stream" },
      })
      if (!response.ok) throw await httpError(response)
      if (!response.body) throw new CayuClientError("protocol", "The follow stream has no body.")
      openedAt = Date.now()
      return readSse(response.body, (frame) => {
        if (frame.event === "end") {
          const data = parseJson(frame.data)
          return { type: "end", status: data?.status }
        }
        if (frame.event === "error") {
          const data = parseJson(frame.data)
          return {
            type: "error",
            error: new CayuClientError("protocol", String(data?.error ?? "The stream failed."), {
              retryable: data?.retryable === true,
            }),
          }
        }
        const sequence = sequenceFromEventId(frame.id)
        // Event frames carry a `session_id:cayu_event_<sequence>` id; other
        // named frames without one are ignored.
        if (frame.event !== "message" && sequence === null) return null
        const envelope = parseJson(frame.data)
        if (sequence === null || !envelope || typeof envelope.type !== "string") {
          return {
            type: "error",
            error: new CayuClientError("protocol", "The follow stream sent an invalid frame."),
          }
        }
        receivedEvent = true
        if (acceptEvent({ ...envelope, sequence }) && LIFECYCLE_EVENT_TYPES.has(envelope.type)) {
          lifecycle = envelope.type
        }
        return null
      })
    })
    // A stream that delivered events, or stayed open on heartbeats, was healthy
    // even if the server later closed it.
    const healthy =
      receivedEvent || (openedAt !== null && Date.now() - openedAt >= STREAM_HEALTHY_AFTER_MS)
    return { ...(outcome ?? { type: "closed" }), lifecycle, healthy }
  }

  const done = (async () => {
    let streamUsable = context.streamPath !== null
    let streamFailures = 0
    let catchUp = false
    let statusCheck = true
    let terminalSeen = false
    let attempt = 0
    let idlePolls = 0
    let lastRequestAt = Number.NEGATIVE_INFINITY
    let dueAt = 0
    let retryAt = 0
    let hideCount = runner.hideCount

    const noteLifecycle = (type) => {
      if (type === null) return
      statusCheck = true
      terminalSeen = TERMINAL_EVENT_TYPES.has(type)
    }
    const scheduleRetry = (failure) => {
      attempt += 1
      retryAt = Date.now() + Math.max(backoffDelay(attempt), failure?.retryAfterMs ?? 0)
      dueAt = retryAt
    }

    for (;;) {
      const scheduledHideCount = hideCount
      // After the page was hidden, resume promptly from the last sequence, but
      // never sooner than a pending retry or one poll interval after the last request.
      const ready = await runner.waitUntil(() =>
        runner.hideCount === scheduledHideCount
          ? dueAt
          : Math.max(retryAt, lastRequestAt + basePollMs),
      )
      if (!ready) return finish("aborted")
      hideCount = runner.hideCount
      let phase = "status"
      try {
        if (statusCheck) {
          lastRequestAt = Date.now()
          await readStatus()
          // A terminal event can become visible just before the status does;
          // keep checking at the poll cadence until they agree.
          statusCheck = terminalSeen && !TERMINAL_STATUSES.has(state.status)
        }
        if (TERMINAL_STATUSES.has(state.status)) {
          phase = "poll"
          state.transport = "poll"
          lastRequestAt = Date.now()
          const page = await pollPage()
          attempt = 0
          if (!page.hasMore) return finish("terminal")
          dueAt = 0
          continue
        }
        if (streamUsable && !catchUp) {
          phase = "stream"
          state.transport = "stream"
          lastRequestAt = Date.now()
          const outcome = await readStream()
          noteLifecycle(outcome.lifecycle)
          if (outcome.healthy) {
            attempt = 0
            streamFailures = 0
          } else if (outcome.type !== "error") {
            streamFailures += 1
            if (streamFailures >= DEFAULTS.maxStreamFailures) streamUsable = false
          }
          if (outcome.type === "error") throw outcome.error
          if (outcome.type === "end") {
            setStatus(outcome.status)
            statusCheck = false
            if (TERMINAL_STATUSES.has(state.status)) return finish("terminal")
          }
          // The stream closed before the session ended: reconnect with backoff.
          scheduleRetry(null)
          continue
        }
        phase = "poll"
        state.transport = "poll"
        lastRequestAt = Date.now()
        const page = await pollPage()
        attempt = 0
        noteLifecycle(page.lifecycle)
        if (page.hasMore || page.lifecycle !== null) {
          dueAt = 0
          continue
        }
        if (catchUp) {
          // Caught up past what the stream could not send; stream again.
          catchUp = false
          if (streamUsable) {
            dueAt = 0
            continue
          }
        }
        // Back off while nothing new arrives: base, 2x, 4x ... up to the ceiling.
        if (page.received) idlePolls = 0
        dueAt = lastRequestAt + Math.min(maxPollMs, basePollMs * 2 ** idlePolls)
        if (!page.received) idlePolls += 1
      } catch (error) {
        if (isAbortError(error)) {
          if (runner.stopped) return finish("aborted")
          continue
        }
        const failure = toClientError(error)
        state.lastError = failure
        if (failure.kind === "auth") return fail(failure)
        if (phase === "stream") {
          streamFailures += 1
          if (!failure.retryable || [404, 405, 501].includes(failure.status)) {
            // The advertised stream is not usable here. Polling stays correct,
            // so catch up by polling instead of stopping.
            catchUp = true
            if (failure.status !== null || streamFailures >= DEFAULTS.maxStreamFailures) {
              streamUsable = false
            }
            dueAt = 0
            continue
          }
          if (streamFailures >= DEFAULTS.maxStreamFailures) streamUsable = false
        } else if (!failure.retryable) {
          return fail(failure)
        }
        scheduleRetry(failure)
      }
    }
  })()

  function finish(reason) {
    runner.stop()
    return { reason, status: state.status, lastSequence: state.lastSequence }
  }

  function fail(error) {
    runner.stop()
    invoke(options.onError, error)
    throw error
  }

  return {
    done,
    stop: () => runner.stop(),
    get lastSequence() {
      return state.lastSequence
    },
    get status() {
      return state.status
    },
    get transport() {
      return state.transport
    },
    get lastError() {
      return state.lastError
    },
  }
}

function watchSessions(context, filter, options) {
  const base = Math.max(
    MIN_WATCH_INTERVAL_MS,
    positive(options.intervalMs, DEFAULTS.watchIntervalMs),
  )
  const maxInterval = Math.max(base, positive(options.maxIntervalMs, DEFAULTS.watchMaxIntervalMs))
  const statuses =
    filter.status === undefined || filter.status === null
      ? [null]
      : [...new Set(Array.isArray(filter.status) ? filter.status : [filter.status])]
  const runner = new Runner(options.signal)
  let stopReason = null

  const stop = (reason = "stopped") => {
    if (runner.stopped) return
    stopReason = reason
    runner.stop()
  }

  const refresh = async () => {
    const byId = new Map()
    for (const status of statuses) {
      const params = new URLSearchParams({
        limit: String(positive(filter.limit, DEFAULTS.watchPageLimit)),
      })
      if (status !== null) params.set("status", status)
      if (filter.agentName) params.set("agent_name", filter.agentName)
      if (filter.environmentName) params.set("environment_name", filter.environmentName)
      if (filter.parentSessionId) params.set("parent_session_id", filter.parentSessionId)
      for (const label of filter.labels ?? []) params.append("label", label)
      const page = await runner.run((signal) =>
        requestJson(context, `/sessions?${params}`, { signal }),
      )
      if (!page || !Array.isArray(page.sessions)) {
        throw new CayuClientError("contract", "The server returned an invalid session list.")
      }
      for (const session of page.sessions) byId.set(session.id, session)
    }
    return [...byId.values()].sort((left, right) =>
      String(right.updated_at).localeCompare(String(left.updated_at)),
    )
  }

  void (async () => {
    let interval = base
    let attempt = 0
    let dueAt = 0
    let fingerprint = null
    for (;;) {
      if (!(await runner.waitUntil(() => dueAt))) break
      const startedAt = Date.now()
      try {
        const sessions = await refresh()
        attempt = 0
        const next = JSON.stringify(sessions.map((s) => [s.id, s.status, s.updated_at]))
        if (next !== fingerprint) {
          fingerprint = next
          interval = base
          invoke(options.onChange, sessions)
        } else {
          interval = Math.min(maxInterval, interval * 2)
        }
        if (!sessions.some((session) => ACTIVE_STATUSES.has(session.status))) {
          stop("idle")
          break
        }
        dueAt = startedAt + interval
      } catch (error) {
        if (isAbortError(error)) {
          if (runner.stopped) break
          // Hidden mid-refresh: the aborted request still counts toward spacing.
          dueAt = startedAt + base
          continue
        }
        const failure = toClientError(error)
        if (!failure.retryable) {
          invoke(options.onError, failure)
          stop("error")
          break
        }
        attempt += 1
        dueAt = Date.now() + Math.max(base, backoffDelay(attempt), failure.retryAfterMs ?? 0)
      }
    }
    invoke(options.onStop, stopReason ?? "stopped")
  })()

  return () => stop("stopped")
}

/**
 * Owns the visibility subscription, the in-flight request, and delays for one
 * follow or watch. Hiding the page aborts the in-flight request or delay, and
 * nothing is sent until the page is visible again.
 */
class Runner {
  constructor(signal) {
    this.stopped = false
    this.hideCount = 0
    this.active = null
    this.waiters = new Set()
    this.document = globalThis.document
    this.onVisibility = () => {
      if (this.hidden()) {
        this.hideCount += 1
        this.active?.abort()
      } else {
        this.wake()
      }
    }
    this.document?.addEventListener?.("visibilitychange", this.onVisibility)
    this.signal = signal ?? null
    this.onAbort = () => this.stop()
    if (this.signal?.aborted) this.stop()
    else this.signal?.addEventListener("abort", this.onAbort, { once: true })
  }

  hidden() {
    return this.document?.hidden === true
  }

  wake() {
    const waiters = [...this.waiters]
    this.waiters.clear()
    for (const resolve of waiters) resolve()
  }

  stop() {
    if (this.stopped) return
    this.stopped = true
    this.active?.abort()
    this.document?.removeEventListener?.("visibilitychange", this.onVisibility)
    this.signal?.removeEventListener("abort", this.onAbort)
    this.wake()
  }

  async untilVisible() {
    while (!this.stopped && this.hidden()) {
      await new Promise((resolve) => this.waiters.add(resolve))
    }
    return !this.stopped
  }

  async run(operation) {
    const controller = new AbortController()
    this.active = controller
    if (this.stopped || this.hidden()) controller.abort()
    try {
      if (controller.signal.aborted) throw abortError()
      return await operation(controller.signal)
    } finally {
      if (this.active === controller) this.active = null
    }
  }

  /** Wait until visible and `dueAt()` has passed. Returns false once stopped. */
  async waitUntil(dueAt) {
    for (;;) {
      if (!(await this.untilVisible())) return false
      const remaining = dueAt() - Date.now()
      if (remaining <= 0) return !this.stopped
      try {
        await this.run((signal) => delay(remaining, signal))
      } catch (error) {
        if (!isAbortError(error)) throw error
        if (this.stopped) return false
      }
    }
  }
}

async function readSse(body, onFrame) {
  const reader = body.getReader()
  // Streaming decode keeps a multi-byte character split across reads intact.
  const decoder = new TextDecoder()
  let buffer = ""
  // CR, LF, and CRLF each end one line. A CR that ends one read has already
  // ended its line, so an LF that starts the next read belongs to it.
  let pendingCR = false
  let frame = { event: "", data: [], id: null }
  try {
    for (;;) {
      const { value, done } = await reader.read()
      let text = done ? decoder.decode() : decoder.decode(value, { stream: true })
      if (text !== "") {
        if (pendingCR && text.startsWith("\n")) text = text.slice(1)
        pendingCR = text.endsWith("\r")
      }
      buffer += text
      const lines = buffer.split(/\r\n|\r|\n/)
      // The last piece has no line ending yet. At the end of the stream it is
      // an incomplete line, which the SSE format discards.
      buffer = lines.pop() ?? ""
      for (const line of lines) {
        if (line === "") {
          if (frame.data.length > 0 || frame.event !== "") {
            const result = onFrame({
              event: frame.event || "message",
              data: frame.data.join("\n"),
              id: frame.id,
            })
            if (result) return result
          }
          frame = { event: "", data: [], id: null }
          continue
        }
        if (line.startsWith(":")) continue
        const colon = line.indexOf(":")
        const field = colon === -1 ? line : line.slice(0, colon)
        let value = colon === -1 ? "" : line.slice(colon + 1)
        if (value.startsWith(" ")) value = value.slice(1)
        if (field === "event") frame.event = value
        else if (field === "data") frame.data.push(value)
        else if (field === "id") frame.id = value
      }
      if (done) return null
    }
  } finally {
    reader.cancel().catch(() => {})
  }
}

function sequenceFromEventId(id) {
  if (typeof id !== "string") return null
  const marker = ":cayu_event_"
  const index = id.lastIndexOf(marker)
  if (index <= 0) return null
  const digits = id.slice(index + marker.length)
  if (!/^\d+$/.test(digits)) return null
  const sequence = Number(digits)
  return Number.isSafeInteger(sequence) ? sequence : null
}

function sessionFollowStreamPath(contract) {
  const advertised = contract?.sse?.session_follow
  if (!advertised) return null
  const capability = contract?.capabilities?.surfaces?.session_follow
  if (capability?.read?.enabled === false) return null
  // `path_template` includes the API prefix, e.g. `/api/sessions/{session_id}/events/stream`.
  const template = typeof advertised.path_template === "string" ? advertised.path_template : ""
  const prefix = typeof contract.api_prefix === "string" ? contract.api_prefix : ""
  if (prefix && template.startsWith(`${prefix}/`) && template.includes("{session_id}")) {
    return template.slice(prefix.length)
  }
  return "/sessions/{session_id}/events/stream"
}

async function send(context, path, { signal, headers = {} } = {}) {
  try {
    return await context.fetch(`${context.apiBase}${path}`, {
      method: "GET",
      credentials: context.credentials,
      headers: { Accept: "application/json", ...context.headers, ...headers },
      signal,
    })
  } catch (error) {
    if (isAbortError(error)) throw error
    throw new CayuClientError("network", "The request did not reach the server.", {
      retryable: true,
      cause: error,
    })
  }
}

async function requestJson(context, path, options) {
  const response = await send(context, path, options)
  if (!response.ok) throw await httpError(response)
  return readJson(response)
}

async function readJson(response) {
  try {
    return await response.json()
  } catch (error) {
    if (isAbortError(error)) throw error
    throw new CayuClientError("protocol", "The server returned invalid JSON.", {
      status: response.status,
      retryable: true,
      cause: error,
    })
  }
}

async function httpError(response) {
  const status = response.status
  let detail = ""
  try {
    const body = await response.json()
    if (typeof body?.detail === "string") detail = body.detail.slice(0, 500)
  } catch {
    // The status code is enough when the body is not JSON.
  }
  const suffix = detail ? `: ${detail}` : "."
  if (status === 401 || status === 403) {
    return new CayuClientError("auth", `The server rejected the request (${status})${suffix}`, {
      status,
    })
  }
  if (status === 404) {
    return new CayuClientError("not_found", `Not found (404)${suffix}`, { status })
  }
  const retryable = status === 408 || status === 425 || status === 429 || status >= 500
  return new CayuClientError("http", `The server responded ${status}${suffix}`, {
    status,
    retryable,
    retryAfterMs: retryable ? retryAfterMs(response.headers.get("Retry-After")) : null,
  })
}

function retryAfterMs(value) {
  if (!value) return null
  const trimmed = value.trim()
  let milliseconds = null
  if (/^\d+$/.test(trimmed)) milliseconds = Number(trimmed) * 1000
  else {
    const date = Date.parse(trimmed)
    if (!Number.isNaN(date)) milliseconds = Math.max(0, date - Date.now())
  }
  return milliseconds === null ? null : Math.min(milliseconds, DEFAULTS.retryAfterMaxMs)
}

function backoffDelay(attempt) {
  const exponential = DEFAULTS.retryInitialDelayMs * 2 ** Math.max(0, attempt - 1)
  const capped = Math.min(DEFAULTS.retryMaxDelayMs, exponential)
  const jitter = 1 + (Math.random() * 2 - 1) * DEFAULTS.jitterRatio
  return Math.round(capped * jitter)
}

function toClientError(error) {
  if (error instanceof CayuClientError) return error
  return new CayuClientError("protocol", error instanceof Error ? error.message : String(error), {
    retryable: true,
    cause: error,
  })
}

function delay(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort)
      resolve()
    }, milliseconds)
    const onAbort = () => {
      clearTimeout(timer)
      reject(abortError())
    }
    signal.addEventListener("abort", onAbort, { once: true })
  })
}

function invoke(callback, value) {
  if (typeof callback !== "function") return
  try {
    callback(value)
  } catch (error) {
    // A failing UI callback must not stop the follow loop; surface it the way
    // browsers report listener errors.
    if (typeof globalThis.reportError === "function") globalThis.reportError(error)
    else
      queueMicrotask(() => {
        throw error
      })
  }
}

function abortError() {
  return new DOMException("The operation was aborted.", "AbortError")
}

function isAbortError(error) {
  return error instanceof Error && error.name === "AbortError"
}

function parseJson(text) {
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

function normalizeSequence(value) {
  if (value === undefined || value === null) return null
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new TypeError("afterSequence must be a non-negative integer.")
  }
  return value
}

function requireSessionId(sessionId) {
  if (typeof sessionId !== "string" || sessionId.trim() === "") {
    throw new TypeError("sessionId must be a non-empty string.")
  }
  return sessionId
}

function positive(value, fallback) {
  return Number.isFinite(value) && value > 0 ? value : fallback
}

function trimSlash(value) {
  return value.endsWith("/") ? value.slice(0, -1) : value
}
